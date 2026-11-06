"""Bounded pairwise boosting: compare alternatives of the SAME cell, including KEEP."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def fit(x, comparisons, *, seed=20260927, iterations=45):
    if not comparisons:
        return {'constant': 0.0, 'training_examples': len(x)}
    matrix = np.asarray(x, dtype=np.float32)
    positive = np.asarray([p[0] for p in comparisons], dtype=int)
    negative = np.asarray([p[1] for p in comparisons], dtype=int)
    weight = np.asarray([p[2] for p in comparisons], dtype=float)
    scores = np.zeros(len(x))
    trees = []
    rate = 0.12
    for iteration in range(iterations):
        loss = 1 / (1 + np.exp(np.clip(scores[positive] - scores[negative], -35, 35)))
        gradient = np.zeros(len(x))
        hessian = np.zeros(len(x))
        np.add.at(gradient, positive, weight * loss)
        np.add.at(gradient, negative, -weight * loss)
        curvature = weight * loss * (1 - loss)
        np.add.at(hessian, positive, curvature)
        np.add.at(hessian, negative, curvature)
        active = hessian > 1e-10
        target = np.clip(gradient[active] / hessian[active], -4, 4)
        estimator = DecisionTreeRegressor(max_depth=4, min_samples_leaf=6, random_state=seed + iteration)
        estimator.fit(matrix[active], target, sample_weight=hessian[active])
        scores += rate * estimator.predict(matrix)
        t = estimator.tree_
        trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(),
                      'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(),
                      'value': t.value[:, 0, 0].tolist()})
    model = {'pairwise': True, 'log_odds': 0.0, 'rate': rate, 'trees': trees,
             'feature_count': len(x[0]), 'training_examples': len(x), 'comparisons': len(comparisons),
             'objective': 'within-cell correct versus wrong alternatives; normal KEEP is positive',
             'score_is_calibrated_probability': False}
    from .models import predict
    expected = 1 / (1 + np.exp(-np.clip(scores, -40, 40)))
    assert np.max(np.abs(predict(model, x) - expected)) < 1e-10
    return model
