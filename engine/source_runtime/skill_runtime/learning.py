"""Cross-check all authorized development sources, refit their reusable skill.

Only the supplied development bundle is readable here. Cached base fits must
have exactly the same training groups/schema. A refitted checkpoint remains
pending independent validation; its fitting/calibration scores are not tests.
"""
from pathlib import Path
import time
from sklearn.model_selection import GroupKFold
from unseen_table.common import (load, dump, fresh, fingerprint, sha256,
                                normalize_bundle, equal)
from unseen_table.learn_skill import fit, pack_observations
from unseen_table.rank_candidates import predict_rows
from unseen_table.decision_check import crossfit
from unseen_table.validate_skill import choose_policy, measure
from unseen_table.save_skill import save
from unseen_table.save_skill import read as read_base
from .codec import fit as fit_codec, encode
from .row_family import fit as fit_family
from .execute import propose_from_cells, POLICY
from .qualification import qualify, NEW_CLAUSES


def metrics(decisions, records):
    c = dict(errors=0, normal=0, changes=0, correct=0, damaged=0)
    for p in decisions:
        truth = records[p['row']]['truth'][p['field']]
        err = not equal(p['before'], truth); changed = not equal(p['before'], p['after'])
        c['errors'] += err; c['normal'] += not err; c['changes'] += changed
        c['correct'] += err and changed and equal(p['after'], truth)
        c['damaged'] += not err and changed
    p = c['correct'] / c['changes'] if c['changes'] else 0
    r = c['correct'] / c['errors'] if c['errors'] else 0
    return {**c, 'P': p, 'R': r, 'F1': 2*p*r/(p+r) if p+r else 0,
            'damage': c['damaged'] / c['normal'] if c['normal'] else None}


def checked_fit(records, declaration, cache=None):
    model = None
    if cache is not None:
        alternate = cache.with_name(cache.name.replace('-fitted.json','-model.json'))
        if cache.is_file(): model=load(cache)
        elif alternate.is_file(): model=load(alternate)
        elif cache.name=='refitted_model.json' and (cache.parent/'base').is_dir(): model=read_base(cache.parent/'base')
    if model is not None:
        if set(model['numeric_context']['training_groups']) != {r['group'] for r in records}:
            raise ValueError('Cached model source boundary differs')
        if model['declaration'] != declaration:
            raise ValueError('Cached model declaration differs')
        return model
    return fit(records, declaration, seed=20260927)


def learn(bundle, declaration, scope, output, *, cache_dir=None):
    started = time.monotonic(); root = fresh(output)
    normalized, audit = normalize_bundle(bundle, declaration)
    records = normalized['learning'] + normalized['validation']
    if len({r['group'] for r in records}) < 20:
        raise ValueError('At least 20 independently labeled source groups required')
    dump(root / 'preregistration.json', {'source_bundle_sha256': fingerprint(bundle),
         'scope': scope, 'seed': 20260927, 'policy': POLICY, 'real9B_learning_calls': 0,
         'raw_learning_cases_visible_to_model': False, 'sources': len({r['group'] for r in records}),
         'uses_all_authorized20_percent_with_source_cross_checks': True,
         'needs_independent_validation_after_refit': True,
         'code': {str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name('execute.py')]}})
    checks = []; held = []; folds = []
    cache_root = Path(cache_dir) if cache_dir else None
    for fold, (a, b) in enumerate(GroupKFold(3).split(records, groups=[r['group'] for r in records])):
        train, check = [records[i] for i in a], [records[i] for i in b]
        cache = cache_root / f'fold-{fold}-fitted.json' if cache_root else None
        model = checked_fit(train, declaration, cache)
        dump(root / f'fold-{fold}-model.json', model)
        cells = predict_rows([r['dirty'] for r in check], model)
        checks += pack_observations(check, cells)
        held.append((train, check, model, cells))
        folds.append({'training_groups': sorted(r['group'] for r in train),
                      'held_groups': sorted(r['group'] for r in check)})
        dump(root / 'progress.json', {'fitted_outer_folds': fold + 1,
                                     'seconds': time.monotonic() - started})
        print('V9 learning fold', fold + 1, 'complete', flush=True)
    final_checker, checker_folds = crossfit(checks, 20260927)
    base_policy, base_calibration = choose_policy(checks)
    provisional={**POLICY,'enabled_new_clauses':{s:True for s in NEW_CLAUSES}}
    held_decisions=[]
    for train, check, model, cells in held:
        model['policy'] = base_policy
        ps = propose_from_cells([r['dirty'] for r in check], cells, model,
                    fit_codec(train, declaration), fit_family(train, declaration), provisional)
        held_decisions.append((ps,check))
    enabled,qualification=qualify(held_decisions)
    policy={**POLICY,'enabled_new_clauses':enabled}
    total = dict(errors=0, normal=0, changes=0, correct=0, damaged=0)
    for train, check, model, cells in held:
        ps = propose_from_cells([r['dirty'] for r in check],cells,model,
                   fit_codec(train,declaration),fit_family(train,declaration),policy)
        m=metrics(ps,check)
        for k in total:total[k]+=m[k]
    calibration = metrics_from_counts(total)
    dump(root / 'calibration.json', {'wrapper': calibration, 'base': base_calibration,
          'component_qualification':qualification,
          'source_folds': folds, 'checker_folds': checker_folds,
          'scope': 'reused learning calibration, never independent test performance'})
    final = checked_fit(records, declaration, cache_root / 'refitted_model.json' if cache_root else None)
    final.update(decision_model=final_checker, policy=base_policy, format_version=6,
         skill_id=declaration['id'], version='6.0.0', scope=scope, status='pending_independent_validation',
         seed=20260927, audit=audit, folds=folds, learning_bundle_sha256=fingerprint(bundle),
         validation_summary=base_calibration, validation_before_qualification=base_calibration,
         quality_gate={'passed': False, 'reason': 'refit requires independent validation; no75 recall target applied here'},
         qualification={'validation_selects_rules_or_thresholds': False}, learning_cross_check=base_calibration,
         learning_summary={'mode': 'local20_percent_learning_summary_only', 'model_calls': 0,
             'seconds': time.monotonic() - started, 'source_cases_visible_to_model': False})
    save(root / 'base', final)
    codec, family = fit_codec(records, declaration), fit_family(records, declaration)
    dump(root / 'configuration.json', {'format_version': 10, 'seed': 20260927,
         'raw_reference_visible_to_model': False, 'policy': policy, 'codec': encode(codec),
         'component_qualification':qualification,
         'row_family': family, 'scope': scope, 'calibration': calibration,
         'status': 'pending_independent_validation', 'quality_gate_v10': {'passed': False},
         'source_bundle_sha256': fingerprint(bundle), 'original_experiments_unchanged': True})
    dump(root / 'compiled_rules.json', {'rules': final['rules'], 'relations': final['relations'],
         'codec': encode(codec), 'row_family': family, 'policy': policy,
         'component_qualification':qualification,
         'language': 'bounded executable template configuration; no arbitrary generated Python'})
    (root / 'SKILL.md').write_text('''# Learned V10 repair skill — pending independent validation

Use only with confirmed source scope and exact field roles. Backend learning used the
authorized20% labeled groups and source-separated cross-checks. All labeled rows stay
local. The9B model receives only the current dirty target fields and derived summaries.

Tool: apply_verified_skill(batch). Execute the fixed learned transforms, candidate
selector and checker, absolute point-evidence guard and native-point preservation.
Every supplemental clause needs its own source-held teaching qualification.
Disable a discovered pattern when its exact repairs are inaccurate. A sole
candidate or rounded numerical coincidence does not establish correctness.
Changing a plausible value requires evidence beyond a possible inverse. Never infer
an exact missing continuous measurement merely from a mean. IDs/labels are protected.
This checkpoint is a research candidate. Auto-deployment stays closed until an
independent output-frozen evaluation qualifies it. Preserve all historical versions.
''')
    dump(root / 'manifest.json', {'files': {str(p.relative_to(root)): sha256(p)
       for p in sorted(root.rglob('*')) if p.is_file() and not p.name.startswith('fold-')}})
    return {'skill': str(root.resolve()), 'calibration': calibration,'component_qualification':qualification,
            'seconds': time.monotonic() - started, 'status': 'pending_independent_validation'}


def metrics_from_counts(c):
    p = c['correct']/c['changes'] if c['changes'] else 0
    r = c['correct']/c['errors'] if c['errors'] else 0
    return {**c, 'P': p, 'R': r, 'F1': 2*p*r/(p+r) if p+r else 0,
            'damage': c['damaged']/c['normal'] if c['normal'] else None}
