"""Unchanged classification settings from v3; used only after output sealing."""
from .common import fingerprint, number

def classify(reference, dirty, repaired, holdout, targets, identity):
    from sklearn import __version__ as sklearn_version
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score
    if any('class' not in row for row in reference + holdout): raise ValueError('下游参考文件需要class列')
    if 'class' in targets: raise ValueError('分类标签不能作为修复目标')
    if {fingerprint([r[c] for c in identity]) for r in reference} & {fingerprint([r[c] for c in identity]) for r in holdout}:
        raise ValueError('下游训练与留出记录重复')
    def x(rows): return [[float(number(r[c])) if number(r[c]) is not None else float('nan') for c in targets] for r in rows]
    conditions = {}; y = [r['class'] for r in reference]; y_test = [r['class'] for r in holdout]
    for name, rows in [('dirty', dirty), ('repaired', repaired), ('original_reference', reference)]:
        estimator = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=20260927))
        estimator.fit(x(rows), y); prediction = estimator.predict(x(holdout))
        conditions[name] = {'accuracy': float(accuracy_score(y_test, prediction)),
            'macro_f1': float(f1_score(y_test, prediction, average='macro')), 'test_rows': len(holdout),
            'correct_predictions': int(sum(a == b for a, b in zip(prediction, y_test)))}
    return {'conditions': conditions, 'settings': {'C': 1, 'max_iter': 2000, 'seed': 20260927,
        'train_only_imputation_and_scaling': True, 'holdout_is_original_clean_data': True, 'tuning': False},
        'sklearn_version': sklearn_version}
