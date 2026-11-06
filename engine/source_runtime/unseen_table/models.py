"""Fit bounded classifiers; persist executable numeric trees without pickle."""
import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def fit_classifier(x, y, weights=None, *, seed=20260927, iterations=70):
    if not x:
        return {'constant': 0.0, 'training_examples': 0}
    if len(set(y)) == 1:
        return {'constant': float(y[0]), 'training_examples': len(y)}
    estimator = GradientBoostingClassifier(n_estimators=iterations, learning_rate=0.08, max_depth=4, min_samples_leaf=6, random_state=seed, loss='log_loss')
    estimator.fit(np.asarray(x), y, sample_weight=weights)
    prior = float(estimator.init_.class_prior_[1])
    trees = []
    for stage in estimator.estimators_:
        tree = stage[0].tree_
        trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(), 'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(), 'value': tree.value[:, 0, 0].tolist()})
    model = {'log_odds': math.log(prior / (1 - prior)), 'rate': estimator.learning_rate, 'trees': trees, 'training_examples': len(y), 'feature_count': len(x[0])}
    assert np.max(np.abs(predict(model, x) - estimator.predict_proba(x)[:, 1])) < 1e-10
    return model

def predict(model, x):
    if not len(x):
        return np.empty(0)
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    matrix = np.asarray(x, dtype=np.float32)
    scores = np.full(len(x), model['log_odds'])
    for tree in model['trees']:
        left = np.asarray(tree['left'])
        right = np.asarray(tree['right'])
        feature = np.asarray(tree['feature'])
        threshold = np.asarray(tree['threshold'])
        nodes = np.zeros(len(x), dtype=int)
        while True:
            active = np.flatnonzero(left[nodes] != -1)
            if not len(active):
                break
            old_nodes = nodes[active]
            nodes[active] = np.where(matrix[active, feature[old_nodes]] <= threshold[old_nodes], left[old_nodes], right[old_nodes])
        scores += model['rate'] * np.asarray(tree['value'])[nodes]
    return 1 / (1 + np.exp(-np.clip(scores, -40, 40)))

def fit_family(x, y, weights=None, *, seed=20260927, iterations=70, class_weight=None, min_leaf=3):
    estimator = RandomForestClassifier(n_estimators=iterations, max_depth=9, min_samples_leaf=min_leaf, max_features=0.8, random_state=seed, n_jobs=1, class_weight=class_weight)
    estimator.fit(np.asarray(x), y, sample_weight=weights)
    trees = []
    for fitted in estimator.estimators_:
        t = fitted.tree_
        values = t.value[:, 0, :]
        values = values / values.sum(axis=1, keepdims=True)
        trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(), 'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(), 'values': values.tolist()})
    model = {'classes': estimator.classes_.tolist(), 'trees': trees}
    assert np.max(np.abs(predict_family(model, x) - estimator.predict_proba(x))) < 1e-10
    return model

def predict_family(model, x):
    matrix = np.asarray(x, dtype=np.float32)
    probabilities = np.zeros((len(x), len(model['classes'])))
    for tree in model['trees']:
        left = np.asarray(tree['left'])
        right = np.asarray(tree['right'])
        feature = np.asarray(tree['feature'])
        threshold = np.asarray(tree['threshold'])
        nodes = np.zeros(len(x), dtype=int)
        while True:
            active = np.flatnonzero(left[nodes] != -1)
            if not len(active):
                break
            at = nodes[active]
            nodes[active] = np.where(matrix[active, feature[at]] <= threshold[at], left[at], right[at])
        probabilities += np.asarray(tree['values'])[nodes]
    return probabilities / len(model['trees'])
