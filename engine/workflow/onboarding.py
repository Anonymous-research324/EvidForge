"""First10 learning, second10 held out: no refit or feedback after assessment.

The historical learner still merges its two INTERNAL folds. Only the first
external learning partition is split into those folds. The external validation
partition is stored outside every skill/native stage and never passed to fit.
"""
from pathlib import Path
from copy import deepcopy
import time
from learning.environment import ROOT
from unseen_table.common import (load, dump, fresh, normalize_bundle, fingerprint,
                                 validate_declaration, sha256, equal, write_table)
from .packages import describe, execute, code_files, verify_files


def split_learning(bundle, schema):
    validate_declaration(schema, list(schema['fields']))
    normalized, audit = normalize_bundle(bundle, schema)
    groups = sorted(bundle['learning'], key=lambda g: g['group'])
    if len(groups) < 20:
        raise ValueError('This learner needs at least20 independent LEARNING groups. '
                         'Add learning cases; held-out validation groups will never be borrowed.')
    # Fully confirmed duplicate observations must stay on one side even when
    # a caller gives different IDs or different corruption to the same source.
    signatures = {}
    targets = schema['targets']
    for zone, records in normalized.items():
        signatures[zone] = {fingerprint({c: r['truth'][c] for c in targets})
                            for r in records if set(targets) <= set(r['truth'])}
    if signatures['learning'] & signatures['validation']:
        raise ValueError('Confirmed duplicate observations cross learning and external validation')
    middle = len(groups) // 2
    internal = {'learning': deepcopy(groups[:middle]), 'validation': deepcopy(groups[middle:])}
    normalize_bundle(internal, schema)
    return internal, audit


def prepare(bundle, schema, context, out):
    internal, audit = split_learning(bundle, schema)
    root = fresh(out)
    dump(root / 'heldout_validation_local.json', {'validation': bundle['validation']})
    dump(root / 'declaration.json', schema)
    dump(root / 'context.json', context)
    dump(root / 'learning_only_bundle.json', internal)
    dump(root / 'partition_freeze.json', {
        'mode': 'external10_learning_plus10_validation',
        'authorized_bundle_sha256': fingerprint(bundle),
        'learning_only_bundle_sha256': fingerprint(internal),
        'heldout_sha256': sha256(root / 'heldout_validation_local.json'),
        'declaration_sha256': fingerprint(schema), 'audit': audit,
        'validation_sent_to_learning': False, 'refit_after_validation_allowed': False,
        'historical20_merge_entry_unchanged': True})
    from learning.prepare import prepare as prepare_source
    prepare_source(internal, schema, context, root / 'prior_prepared')
    records = load(root / 'prior_prepared/source_records_backend.json')['records']
    if {r['group'] for r in records} != {g['group'] for g in bundle['learning']}:
        raise ValueError('Learning source boundary changed')
    dump(root / 'prepare_boundary_receipt.json', {'only_external_learning_used': True,
         'external_learning_groups': len(bundle['learning']),
         'external_validation_groups': len(bundle['validation']), 'actual9B_calls': 0})
    return root


def freeze_assessment(workflow, skill):
    root = Path(workflow).resolve()
    info = describe(skill)
    partition = load(root / 'partition_freeze.json')
    if fingerprint(load(root / 'learning_only_bundle.json')) != partition['learning_only_bundle_sha256']:
        raise ValueError('Learning partition changed after preparation')
    if fingerprint(load(root / 'declaration.json')) != partition['declaration_sha256']:
        raise ValueError('Field roles changed after preparation')
    if info['scope'] != load(root / 'context.json'):
        raise ValueError('Source scope differs')
    source = load(Path(skill) / 'source_records_backend.json')['records']
    expected = load(root / 'learning_only_bundle.json')
    normalized, _ = normalize_bundle(expected, info['declaration'])
    if fingerprint(sorted(source, key=lambda r: r['row_key'])) != fingerprint(
            sorted(normalized['learning'] + normalized['validation'], key=lambda r: r['row_key'])):
        raise ValueError('Sealed skill was not learned exclusively from the declared learning partition')
    target = root / 'assessment_freeze.json'
    if target.exists():
        raise ValueError('Assessment already frozen; preserve it rather than retune on this validation')
    dump(target, {'skill': info, 'code': code_files(),
         'heldout_sha256': partition['heldout_sha256'], 'unix': time.time(),
         'validation_feedback_available_at_freeze': False,
         'fixed_quality_gate': {'precision': .98, 'normal_damage': .005,
                                'min_changes': 5, 'min_normal_controls': 10}})
    return target


def metrics(records, output, targets):
    if len(records) != len(output):
        raise ValueError('Assessment row counts differ')
    errors = normal = changes = correct = damaged = 0
    for r, after in zip(records, output):
        for c in targets:
            if c not in r['truth']:
                continue
            wrong = not equal(r['dirty'][c], r['truth'][c])
            change = not equal(r['dirty'][c], after[c])
            errors += wrong; normal += not wrong; changes += change
            correct += wrong and change and equal(after[c], r['truth'][c])
            damaged += not wrong and change
    p = correct / changes if changes else 0
    recall = correct / errors if errors else 0
    return {'errors': errors, 'clean': normal, 'changes': changes, 'correct': correct,
            'wrong': changes - correct, 'normal_damaged': damaged, 'P': p, 'R': recall,
            'F1': 2 * p * recall / (p + recall) if p + recall else 0,
            'damage': damaged / normal if normal else None,
            # Preserve the original two-valued edit-detection metric exactly.
            'AUC': (1 + (changes-damaged)/errors-damaged/normal)/2 if errors and normal else None}


def assess(workflow, out):
    root = Path(workflow).resolve()
    frozen = load(root / 'assessment_freeze.json')
    verify_files(frozen['code'])
    if sha256(root / 'heldout_validation_local.json') != frozen['heldout_sha256']:
        raise ValueError('External validation changed after freeze')
    if describe(frozen['skill']['path']) != frozen['skill']:
        raise ValueError('Skill changed after assessment freeze')
    once = root / 'assessment_started.json'
    if once.exists():
        raise ValueError('Validation was already consumed; keep failed receipts and use a new validation source')
    # Consume before execution so failed/restarted jobs cannot silently reuse
    # validation for a new model. Recovery must preserve this frozen checkpoint.
    dump(once, {'out': str(Path(out).resolve()), 'unix': time.time(),
                'frozen_skill_manifest': frozen['skill']['manifest_sha256']})
    groups = load(root / 'heldout_validation_local.json')['validation']
    records = [{**r, 'group': g['group']} for g in groups for r in g['records']]
    schema = frozen['skill']['declaration']; targets = schema['targets']
    dirty = [r['dirty'] for r in records]
    output, decisions = execute(dirty, frozen['skill']['path'])
    # Also test source-authorized confirmed normal values, separately. They do
    # not enlarge the denominator of the original dirty-case damage rate.
    controls = [{**r['dirty'], **r['truth']} for r in records]
    clean_output, _ = execute(controls, frozen['skill']['path'])
    control_records = [{**r, 'dirty': clean} for r, clean in zip(records, controls)]
    result = metrics(records, output, targets)
    normal_result = metrics(control_records, clean_output, targets)
    verify_files(frozen['code'])
    if describe(frozen['skill']['path']) != frozen['skill']:
        raise ValueError('Checkpoint changed during validation')
    gate = frozen['fixed_quality_gate']
    passed = bool(result['changes'] >= gate['min_changes'] and result['clean'] >= gate['min_normal_controls']
                  and result['P'] >= gate['precision'] and result['damage'] <= gate['normal_damage']
                  and normal_result['clean'] >= gate['min_normal_controls']
                  and normal_result['damage'] <= gate['normal_damage'])
    report = fresh(out)
    write_table(report / 'validation_output.csv', list(schema['fields']), output)
    dump(report / 'decisions.json', decisions)
    dump(report / 'assessment.json', {'metrics': result, 'confirmed_normal_scenario': normal_result,
         'passed': passed, 'external_validation_groups': len(groups),
         'scope': 'independent user-provided validation, not a formal benchmark result',
         'frozen_skill': frozen['skill'], 'code': frozen['code'],
         'assessment_freeze_sha256': sha256(root / 'assessment_freeze.json'),
         'formal_reference_read': False, 'actual9B_calls': 0,
         'refit_after_validation': False,
         'next': 'reuse exact checkpoint within confirmed scope' if passed else
                 'keep suggestions for human review; acquire new independent learning/validation before changing the checkpoint'})
    dump(report / 'outputs_freeze.json', {'files': {n: sha256(report / n) for n in
         ['validation_output.csv', 'decisions.json', 'assessment.json']}})
    return load(report / 'assessment.json')
