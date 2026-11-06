"""Learn decision reliability on out-of-group predictions, including wrong alternatives."""
from sklearn.model_selection import GroupKFold
from .common import equal
from .models import fit_classifier, predict

def vector(cell):
    return cell['decision_vector'] + [cell['top']['score'], cell['margin'],
           cell['error_probability'], min(128, cell['candidate_count']) / 128,
           float(cell.get('stage1_agrees', False)), float(cell.get('stage1_disagrees', False)),
           float(cell.get('stage1_was_permitted', False))]

def fit(points, seed):
    usable = [o for o in points if o['cell']['top']['kind'] != 'keep']
    x = [vector(o['cell']) for o in usable]
    y = [int(not equal(o['cell']['original'], o['truth']) and equal(o['cell']['top']['value'], o['truth'])) for o in usable]
    model = fit_classifier(x, y, seed=seed, iterations=45)
    model['trained_on'] = 'held-source-group decisions; normal damage and wrong answers are negative'
    return model

def apply(cells, model):
    if not model:
        return cells
    scores = predict(model, [vector(c) for c in cells])
    for c, score in zip(cells, scores):
        c['repair_confidence'] = float(score) if c['top']['kind'] != 'keep' else 0.0
    return cells

def crossfit(points, seed, *, return_models=False):
    """Threshold training sees held predictions of the checker, not its fitting scores."""
    groups = [o['group'] for o in points]
    audits = []
    held_models = {}
    for train_ids, held_ids in GroupKFold(n_splits=min(3, len(set(groups)))).split(points, groups=groups):
        train = [points[i] for i in train_ids]
        held = [points[i] for i in held_ids]
        model = fit(train, seed)
        apply([o['cell'] for o in held], model)
        for o in held:
            held_models[o['group']] = model
        audits.append({'training_groups': sorted({o['group'] for o in train}), 'check_groups': sorted({o['group'] for o in held})})
    result = (fit(points, seed), audits)
    return (*result, held_models) if return_models else result
