"""Predict repeated numeric states from other fields; never predict a unique exact float."""
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from .common import canonical, number
from .models import fit_family, predict_family
from .trusted_context import bootstrap

OTHER = '__continuous_or_unseen__'

def discover(records, declaration):
    groups = defaultdict(set)
    counts = defaultdict(int)
    for r in records:
        for col, value in r['truth'].items():
            if col in declaration['targets'] and number(value) is not None:
                key = canonical(value)
                groups[key].add(r['group'])
                counts[key] += 1
    return [k for k in sorted(groups, key=lambda k: (-counts[k], k))
            if len(groups[k]) >= 3 and counts[k] >= 4][:8]

def vector(row, target, fitted, trusted=None):
    cols = fitted['declaration']['targets']
    trusted = trusted if trusted is not None else bootstrap(row, fitted)
    result = [float(target == c) for c in cols]
    for col in cols:
        if col == target:
            result.extend([0.0] * 6)
            continue
        original = row[col]
        value = trusted.get('values', {}).get(col, original)
        raw, clean = number(original), number(value)
        p = fitted['profiles'][col]
        scale = max(0.01, p['numeric_sd'])
        result.extend([float(raw is not None), float(not original.strip()),
                       float(clean is not None), float(trusted.get('trust', {}).get(col, 0)),
                       max(-20, min(20, (float(raw) - p['numeric_mean']) / scale)) if raw is not None else 0,
                       max(-20, min(20, (float(clean) - p['numeric_mean']) / scale)) if clean is not None else 0])
    return result

def fit(records, fitted, seed):
    points = discover(records, fitted['declaration'])
    if not points:
        return {}
    x, y, weights, fields = [], [], [], []
    for r in records:
        for row, weight in [(r['dirty'], 1.0), ({**r['dirty'], **r['truth']}, 0.15)]:
            trust = bootstrap(row, fitted)
            for col, truth in r['truth'].items():
                if not fitted['profiles'][col]['numeric'] or number(truth) is None:
                    continue
                value = canonical(truth)
                x.append(vector(row, col, fitted, trust))
                y.append(value if value in points else OTHER)
                weights.append(weight)
                fields.append(col)
    if not x:
        return {}
    model = fit_family(x, y, weights, seed=seed, iterations=40, class_weight='balanced_subsample')
    per_column = {}
    field_count = len(fitted['declaration']['targets'])
    for col in fitted['declaration']['targets']:
        ids = [i for i, field in enumerate(fields) if field == col]
        if len(ids) < 12:
            continue
        per_column[col] = fit_family([x[i][field_count:] for i in ids], [y[i] for i in ids],
                                    [weights[i] for i in ids], seed=seed, iterations=24,
                                    class_weight='balanced_subsample', min_leaf=2)
    return {'points': points, 'model': model, 'per_column': per_column,
            'field_model_weight': 0.8, 'scores_are_empirical_not_guaranteed_probabilities': True,
            'training_groups': sorted({r['group'] for r in records}),
            'self_target_masked': True, 'class_and_identity_excluded': True}

def probabilities(rows, fitted, model=None, trusted=None):
    state = fitted.get('row_state', {}) if model is None else model
    result = [{} for _ in rows]
    if not state:
        return result
    pairs, x = [], []
    for i, row in enumerate(rows):
        trust = trusted[i] if trusted is not None else bootstrap(row, fitted)
        for col, p in fitted['profiles'].items():
            if p['numeric']:
                pairs.append((i, col))
                x.append(vector(row, col, fitted, trust))
    if not x:
        return result
    values = predict_family(state['model'], x)
    for (i, col), scores in zip(pairs, values):
        result[i][col] = {k: 0.0 for k in state['points']}
        result[i][col].update({str(k): float(v) for k, v in zip(state['model']['classes'], scores)})
    field_count = len(fitted['declaration']['targets'])
    for col, model in state.get('per_column', {}).items():
        ids = [i for i, pair in enumerate(pairs) if pair[1] == col]
        if not ids:
            continue
        scores = predict_family(model, [x[i][field_count:] for i in ids])
        alpha = state['field_model_weight']
        for index, probs in zip(ids, scores):
            i, _ = pairs[index]
            local = dict(zip(model['classes'], probs))
            for key in set(result[i][col]) | set(local):
                result[i][col][str(key)] = (1 - alpha) * result[i][col].get(key, 0.0) + alpha * float(local.get(key, 0.0))
    return result

def out_of_group(records, practice, fitted, seed):
    """State features for ranking use models that never saw the source entity's truth."""
    from .learn_transforms import infer
    output = [None for _ in practice]
    groups = [r['group'] for r in records]
    audits = []
    for fold, (train_ids, held_ids) in enumerate(GroupKFold(n_splits=min(3, len(set(groups)))).split(records, groups=groups)):
        train = [records[i] for i in train_ids]
        held_groups = {records[i]['group'] for i in held_ids}
        local = infer(train, fitted['declaration'])
        state = fit(train, local, seed + fold)
        ids = [i for i, r in enumerate(practice) if r['group'] in held_groups]
        for i, values in zip(ids, probabilities([practice[i]['dirty'] for i in ids], local, state)):
            output[i] = values
        audits.append({'training_groups': sorted({r['group'] for r in train}), 'check_groups': sorted(held_groups)})
    assert all(v is not None for v in output)
    return output, audits
