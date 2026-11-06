"""Suspicion is separate from exact repair confidence."""
from .common import equal
from .features import detection_features
from .numeric_context import evidence
from .models import fit_classifier

def fit(records, fitted, seed):
    x = []
    y = []
    weights = []
    for r in records:
        ctx = evidence(r['dirty'], fitted, exclude_group=r['group'])
        for col, truth in r['truth'].items():
            x.append(detection_features(r['dirty'], col, fitted, ctx, r['group']))
            y.append(int(not equal(r['dirty'][col], truth)))
            weights.append(0.4 if r.get('synthetic') else 1.0)
    return fit_classifier(x, y, weights, seed=seed, iterations=60)
