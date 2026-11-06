"""Local clean learning neighborhoods; estimates rank exact answers, never replace them."""
import numpy as np
from .common import number

def fit(records, fitted):
    cols = [c for c, p in fitted['profiles'].items() if p['numeric'] and sum((number(r['truth'].get(c, '')) is not None for r in records)) >= 8]
    if len(cols) < 2:
        return {}
    matrix = [[float(number(r['truth'][c])) if number(r['truth'].get(c, '')) is not None else None for c in cols] for r in records]
    arr = np.array([[np.nan if v is None else v for v in row] for row in matrix])
    center = np.nanmedian(arr, axis=0)
    scale = np.maximum(np.nanstd(arr, axis=0), 0.01)
    return {'columns': cols, 'matrix': matrix, 'center': center.tolist(), 'scale': scale.tolist(), 'training_groups': [r['group'] for r in records], 'trained_on': 'confirmed learning truth only'}

def evidence(row, fitted, trusted=None, exclude_group=None):
    from .trusted_context import bootstrap
    trusted = trusted or bootstrap(row, fitted)
    model = fitted.get('numeric_context', {})
    if not model:
        return {'trusted': trusted, 'columns': {}}
    cols = model['columns']
    arr = np.array([[np.nan if v is None else v for v in r] for r in model['matrix']])
    scale = np.array(model['scale'])
    if exclude_group is not None:
        arr = arr[np.array([g != exclude_group for g in model['training_groups']])]
        if not len(arr):
            return {'trusted': trusted, 'columns': {}}
    observed = np.zeros(len(cols))
    mask = np.zeros(len(cols))
    for i, col in enumerate(cols):
        n = number(trusted['values'].get(col, ''))
        if n is not None and trusted['trust'].get(col, 0) >= 0.98:
            observed[i] = float(n)
            mask[i] = 1
    output = {}
    for i, col in enumerate(cols):
        use = mask.copy()
        use[i] = 0
        available = np.sum(np.isfinite(arr) * use, axis=1)
        if not np.any(available >= 2):
            continue
        errors = np.minimum(((arr - observed) / scale) ** 2, 25)
        distance = np.nansum(errors * use, axis=1) / np.maximum(1, available)
        distance[(available < 2) | ~np.isfinite(arr[:, i])] = np.inf
        ids = np.argsort(distance)[:min(7, len(arr))]
        ids = ids[np.isfinite(distance[ids])]
        if not len(ids):
            continue
        weights = np.exp(-np.minimum(40, distance[ids] * 3))
        weights /= weights.sum()
        values = arr[ids, i]
        estimate = float(weights @ values)
        uncertainty = max(0.01, float(np.sqrt(weights @ (values - estimate) ** 2)), float(scale[i]) * 0.05)
        output[col] = {'estimate': estimate, 'uncertainty': uncertainty, 'observed_context': int(mask.sum() - mask[i]), 'neighbors': [{'value': float(v), 'weight': float(w)} for v, w in zip(values, weights)], 'nearest_distance': float(distance[ids[0]]), 'trusted_only': True}
    return {'trusted': trusted, 'columns': output}
