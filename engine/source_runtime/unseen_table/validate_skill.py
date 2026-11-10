"""Permission is selected on held learning groups; validation only checks the frozen policy."""
from collections import defaultdict
from .common import equal
DEFAULT_TARGETS = {'precision': 0.98, 'recall': 0.75, 'collateral': 0.005}

def region(cell):
    # Sample-count-dependent domain labels are selector evidence, never permission keys.
    # A 6-row fold and the final 10-row learner must address the same permission region.
    # A proven constant must not share a gate with a weaker numeric lookup simply
    # because canonical candidate merging gave lookup evidence higher display priority.
    kind = 'constant' if 'constant' in cell['top'].get('kinds', []) and cell['top']['kind'] != 'keep' else cell['top']['kind']
    return ':'.join([kind, cell.get('old_form', 'unknown')])

def permitted(cell, policy):
    if 'confidence_threshold' in policy:
        return cell['top']['kind'] != 'keep' and not cell.get('tied', False) and cell.get('repair_confidence', 0.0) >= policy['confidence_threshold'] and cell['margin'] >= policy.get('margin_threshold', 0.0)
    p = policy.get('by_region', {}).get(region(cell), policy if 'score_threshold' in policy else {})
    return cell['top']['kind'] != 'keep' and (not cell.get('tied', False)) and bool(p) and (cell['top']['score'] >= p.get('score_threshold', 1.01)) and (cell['margin'] >= p.get('margin_threshold', 1.01))

def measure(observations, policy, qualified=None):
    errors = clean = changes = correct = collateral = covered = 0
    for o in observations:
        c, truth = (o['cell'], o['truth'])
        error = not equal(c['original'], truth)
        errors += int(error)
        clean += int(not error)
        covered += int(error and any((equal(p['value'], truth) for p in c['options'])))
        if permitted(c, policy):
            changes += 1
            correct += int(error and equal(c['top']['value'], truth))
            collateral += int(not error)
    return {'cells': len(observations), 'errors': errors, 'clean_cells': clean, 'changes': changes, 'correct': correct, 'incorrect': changes - correct, 'clean_changes': collateral, 'precision': correct / changes if changes else 0.0, 'recall': correct / errors if errors else 0.0, 'collateral': collateral / clean if clean else 0.0, 'candidate_coverage': covered / errors if errors else 0.0, 'scope': 'development checks, not formal 9B results'}

def choose_policy(observations):
    if observations and all('repair_confidence' in o['cell'] for o in observations):
        best = None
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995]:
            for margin in [0.0, 0.01, 0.03, 0.05, 0.1]:
                p = {'confidence_threshold': threshold, 'margin_threshold': margin,
                     'selected_on': 'cross-fitted decision checker on learning groups only', 'ties': 'abstain'}
                m = measure(observations, p)
                groups = {o['group'] for o in observations if permitted(o['cell'], p)}
                if m['changes'] < 5 or len(groups) < 3 or m['precision'] < 0.98 or m['collateral'] > 0.005:
                    continue
                if best is None or (m['correct'], threshold, margin) > best[0]:
                    best = ((m['correct'], threshold, margin), p)
        policy = best[1] if best else {'confidence_threshold': 1.01, 'margin_threshold': 1.01, 'selected_on': 'learning held groups only', 'ties': 'abstain'}
        calibrated_metrics = measure(observations, policy)
        # The checker is optional: select it only when held LEARNING predictions
        # beat the simpler stable rule policy. Validation never chooses a fallback.
        raw_points = [{**o, 'cell': {k: v for k, v in o['cell'].items() if k != 'repair_confidence'}} for o in observations]
        raw_policy, raw_metrics = choose_policy(raw_points)
        selection = {'checker': calibrated_metrics, 'direct_evidence': raw_metrics,
                     'selected_on': 'learning held groups only; validation cannot select alternatives'}
        if raw_metrics['correct'] >= calibrated_metrics['correct']:
            raw_policy['selection_comparison'] = selection
            return raw_policy, raw_metrics
        policy['selection_comparison'] = selection
        return policy, calibrated_metrics
    pools = defaultdict(list)
    for o in observations:
        if not o['cell'].get('stage1_applied'):
            pools[region(o['cell'])].append(o)
    policies = {}
    for key, points in pools.items():
        best = None
        for score in [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995]:
            for margin in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
                p = {'score_threshold': score, 'margin_threshold': margin}
                m = measure(points, p)
                groups = {o['group'] for o in points if permitted(o['cell'], p)}
                if m['changes'] < 5 or len(groups) < 3 or m['precision'] < 0.98 or m['clean_changes']:
                    continue
                if best is None or (m['correct'], score, margin) > best[0]:
                    best = ((m['correct'], score, margin), {**p, 'empirical_precision': m['precision'], 'checked_changes': m['changes'], 'checked_groups': len(groups)})
        policies[key] = best[1] if best else {'score_threshold': 1.01, 'margin_threshold': 1.01}
    policy = {'by_region': policies, 'selected_on': 'learning held entity groups only', 'ties': 'abstain'}
    return (policy, measure(observations, policy))

def qualify(observations, policy):
    return {'frozen_policy_check': measure(observations, policy), 'validation_selects_rules_or_thresholds': False}

def quality_gate(metrics):
    passed = metrics['precision'] >= 0.98 and metrics['recall'] >= 0.75 and (metrics['collateral'] <= 0.005)
    return {'passed': bool(passed), 'targets': DEFAULT_TARGETS, 'empirical_only': True, 'reason': '通过开发目标；仍需冻结后正式评估' if passed else '未达到75%开发目标；不能自动启动整表正式实验'}
