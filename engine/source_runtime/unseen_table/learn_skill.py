"""Cross-fit the whole two-stage learner; validation only checks frozen decisions."""
from collections import defaultdict, Counter
import time
from sklearn.model_selection import GroupKFold
from .common import normalize_bundle, validate_declaration, fingerprint, canonical, equal
from .learn_transforms import infer
from .learn_relations import fit as fit_relations
from .learn_value_modes import fit as fit_modes
from .source_consensus import fit as fit_sources
from .method_library import applicable
from .augment_examples import augment
from .numeric_context import fit as fit_context, evidence
from .trusted_context import overlay
from .detect_errors import fit as fit_detector
from .rank_candidates import fit as fit_selector, predict_stage, predict_rows
from .validate_skill import choose_policy, measure, quality_gate, qualify
from .features import FEATURE_SCHEMA_VERSION
from .row_state import fit as fit_state, out_of_group
from .decision_check import crossfit as check_crossfit

def core_fit(records, declaration, seed, extra_rules=()):
    fitted = infer(records, declaration, extra_rules=[r for r in extra_rules if r['kind'] != 'relationship'])
    fitted['feature_schema'] = FEATURE_SCHEMA_VERSION
    fitted['training_cells'] = sum((len(r['truth']) for r in records))
    for col, p in fitted['profiles'].items():
        points = [r for r in records if col in r['truth']]
        groups = defaultdict(set)
        for r in points:
            groups[canonical(r['truth'][col])].add(r['group'])
        p['value_groups'] = {v: sorted(gs) for v, gs in groups.items()}
        p['learning_error_fraction'] = sum((not equal(r['dirty'][col], r['truth'][col]) for r in points)) / max(1, len(points))
    fitted['value_modes'] = fit_modes(records, fitted)
    fitted['relations'] = fit_relations(records, declaration, [r for r in extra_rules if r['kind'] == 'relationship'])
    fitted['source_consensus'] = fit_sources(records, declaration)
    fitted['method_usage'] = applicable(declaration, fitted['profiles'])
    fitted['numeric_context'] = fit_context(records, fitted)
    fitted['row_state'] = fit_state(records, fitted, seed)
    practice = augment(records, fitted, seed=seed)
    states, state_folds = out_of_group(records, practice, fitted, seed)
    fitted['state_folds'] = state_folds
    fitted['detector'] = fit_detector(practice, fitted, seed)
    fitted['selector'] = fit_selector(practice, fitted, seed, states=states)
    return fitted

def pack_observations(records, cells):
    return [{'cell': c, 'truth': records[c['row']]['truth'][c['column']], 'group': records[c['row']]['group']} for c in cells if c['column'] in records[c['row']]['truth']]

def observations(records, fitted):
    return pack_observations(records, predict_rows([r['dirty'] for r in records], fitted))

def fit(records, declaration, *, seed=20260927, extra_rules=()):
    started = time.monotonic()
    fitted = core_fit(records, declaration, seed, extra_rules)
    groups = [r['group'] for r in records]
    checks = []
    held_models = {}
    inner = []
    splitter = GroupKFold(n_splits=min(3, len(set(groups))))
    for fold, (train_ids, check_ids) in enumerate(splitter.split(records, groups=groups)):
        train = [records[i] for i in train_ids]
        check = [records[i] for i in check_ids]
        # Outer-fold model suggestions may have seen an inner held group.
        # Inner context models therefore reconstruct proposals solely from their own records.
        model = core_fit(train, declaration, seed, ())
        checks += pack_observations(check, predict_stage([r['dirty'] for r in check], model))
        for r in check:
            held_models[r['group']] = model
        inner.append({'fold': fold, 'training_groups': sorted({r['group'] for r in train}), 'check_groups': sorted({r['group'] for r in check})})
    stage1_decision_model, checker_folds, held_checkers = check_crossfit(checks, seed, return_models=True)
    stage1_policy, stage1_metrics = choose_policy(checks)
    fitted['stage1_decision_model'] = stage1_decision_model
    practice = augment(records, fitted, seed=seed)
    contexts = []
    by_group = defaultdict(list)
    for i, r in enumerate(practice):
        by_group[r['group']].append(i)
    contexts = [None for _ in practice]
    states = [None for _ in practice]
    for group, ids in by_group.items():
        model = held_models[group]
        if declaration.get('source_field'):
            variants = defaultdict(list)
            for local, index in enumerate(ids):
                r = practice[index]
                variant = r['row_key'].rsplit(':', 1)[-1] if r.get('synthetic') else 'original'
                variants[variant].append(local)
            first = []
            for local_ids in variants.values():
                for cell in predict_stage([practice[ids[i]]['dirty'] for i in local_ids], model):
                    cell['row'] = local_ids[cell['row']]
                    first.append(cell)
        else:
            first = predict_stage([practice[i]['dirty'] for i in ids], model)
        from .decision_check import apply as check_apply
        from .row_state import probabilities as state_predict
        check_apply(first, held_checkers[group])
        by_row = [[] for _ in ids]
        for c in first:
            by_row[c['row']].append(c)
        for local, index in enumerate(ids):
            trust = overlay(practice[index]['dirty'], model, by_row[local], stage1_policy)
            contexts[index] = evidence(practice[index]['dirty'], fitted, trust, exclude_group=group)
            states[index] = state_predict([practice[index]['dirty']], model, trusted=[trust])[0]
    fitted['stage1_selector'] = fitted['selector']
    fitted['stage1_policy'] = stage1_policy
    fitted['selector'] = fit_selector(practice, fitted, seed, contexts=contexts, states=states)
    fitted['training_summary'] = {'original_records': len(records), 'practice_records': len(practice) - len(records), 'practice_group_boundaries_preserved': True, 'stage2_contexts': 'held-source-group stage1 predictions', 'inner_proposal_source':'inner training records only', 'stage1_cross_check': stage1_metrics, 'inner_folds': inner, 'state_folds': fitted['state_folds'], 'decision_checker_folds': checker_folds, 'seconds': time.monotonic() - started, 'rule_kinds': dict(Counter((r['kind'] for r in fitted['rules']))), 'relations': len(fitted['relations'])}
    return fitted

def learn(bundle, declaration, scope, *, client=None, seed=20260927, wall_seconds=900, work=None):
    from .workflow_budget import WorkflowBudget, ScopedClient
    validate_declaration(declaration, list(declaration.get('fields', {})))
    if not isinstance(scope, dict) or not scope or any((not isinstance(v, str) or not v.strip() for v in scope.values())):
        raise ValueError('需要明确的适用范围')
    records, audit = normalize_bundle(bundle, declaration)
    learning = records['learning']
    budget = WorkflowBudget(wall_seconds, work=work)
    groups = [r['group'] for r in learning]
    folds = []
    cross = []
    model_audits = []
    final_proposals = []
    for fold, (train_ids, check_ids) in enumerate(GroupKFold(n_splits=3).split(learning, groups=groups)):
        budget.check()
        train = [learning[i] for i in train_ids]
        check = [learning[i] for i in check_ids]
        proposals = []
        if client:
            from .learn_with_9b import learn as model_learn
            proposals, info = model_learn(ScopedClient(client, budget, 2, f'outer-{fold}'), train, declaration, rounds=2)
            model_audits.append({'scope': f'outer-{fold}', 'training_groups': sorted({r['group'] for r in train}), **info})
        fitted = budget.fit(train, declaration, seed, proposals, f'outer-{fold}')
        cross += pack_observations(check, budget.observe(check, fitted, f'outer-{fold}-check'))
        folds.append({'fold': fold, 'training_groups': sorted({r['group'] for r in train}), 'check_groups': sorted({r['group'] for r in check}), 'parameters_refit': True, 'stage2_contexts_out_of_group': True})
    decision_model, checker_folds = check_crossfit(cross, seed)
    policy, cross_metrics = choose_policy(cross)
    if client:
        from .learn_with_9b import learn as model_learn
        final_proposals, info = model_learn(ScopedClient(client, budget, 2, 'final'), learning, declaration, rounds=2)
        model_audits.append({'scope': 'final', 'training_groups': sorted(set(groups)), **info})
    final = budget.fit(learning, declaration, seed, final_proposals, 'final')
    final['decision_model'] = decision_model
    final['training_summary']['final_checker_folds'] = checker_folds

    def strategy_hash():
        return fingerprint({**{k: v for k, v in final.items() if k != 'training_summary'}, 'policy': policy})
    strategy = strategy_hash()
    checks = pack_observations(records['validation'], budget.observe(records['validation'], final, 'validation-check'))
    if strategy_hash() != strategy:
        raise ValueError('校验不得修改已冻结策略')
    metrics = measure(checks, policy)
    gate = quality_gate(metrics)
    model_rules = [r for r in final['rules'] + final['relations'] if r.get('origin') == 'model']
    accepted = len(model_rules)
    summary = {'mode': '9b_executable_rules_accepted' if accepted else '9b_called_no_executable_rules' if client else 'offline_learning', 'accepted': accepted, 'model_calls': budget.calls, 'model_audits': model_audits, 'seconds': budget.elapsed, 'workflow_wall_limit': wall_seconds, 'model': getattr(client, 'model', None), 'final_model_rule_ids': [r['id'] for r in model_rules]}
    budget.check()
    budget.save()
    final.update(format_version=6, skill_id=declaration['id'], version='6.0.0', scope=scope, policy=policy, qualification=qualify(checks, policy), quality_gate=gate, status='validated_development' if gate['passed'] else 'development_target_failed', validation_summary=metrics, validation_before_qualification=metrics, learning_cross_check=cross_metrics, learning_summary=summary, audit=audit, folds=folds, seed=seed, learning_bundle_sha256=fingerprint(bundle), strategy_sha256_before_validation=strategy, validation_changes_strategy=False, validation_reused_during_development=True, formal_model_experiment=False)
    return final
