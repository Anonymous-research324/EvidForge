from learning.environment import RUNTIME
"""Original frozen scoring functions; no method or metric changes."""
from unseen_table.common import equal

def downstream_auc(reference,dirty,repaired,holdout,targets):
    """Same frozen LR/imputation/scaling settings; add probability-based AUC only."""
    from unseen_table.common import number
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    def matrix(rows):
        return [[float(number(r[c])) if number(r[c]) is not None else float('nan') for c in targets] for r in rows]
    y=[r['class'] for r in reference]; test_y=[r['class'] for r in holdout]
    result={}
    for name,rows in [('dirty',dirty),('repaired',repaired),('original_reference',reference)]:
        estimator=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            LogisticRegression(C=1.0,max_iter=2000,random_state=20260927))
        estimator.fit(matrix(rows),y); probability=estimator.predict_proba(matrix(holdout))
        classes=list(estimator.classes_)
        if set(test_y)!=set(classes):
            result[name]=None
        elif len(classes)==2:
            result[name]=float(roc_auc_score([int(v==classes[1]) for v in test_y],probability[:,1]))
        else:
            result[name]=float(roc_auc_score(test_y,probability,labels=classes,multi_class='ovr',average='macro'))
    return result

def score(dirty,repaired,truth,targets,identity):
    if len(dirty)!=len(repaired) or len(dirty)!=len(truth):raise ValueError('Scoring rows differ')
    errors=clean=changes=correct=damage=0
    for d,r,t in zip(dirty,repaired,truth):
        if any(d[c]!=r[c] or d[c]!=t[c] for c in identity):raise ValueError('Scoring identities differ')
        for c in targets:
            err=not equal(d[c],t[c]);changed=not equal(d[c],r[c]);errors+=err;clean+=not err;changes+=changed;correct+=err and changed and equal(r[c],t[c]);damage+=not err and changed
    p=correct/changes if changes else 0;r=correct/errors if errors else 0
    return {'P':p,'R':r,'F1':2*p*r/(p+r) if p+r else 0,'damage':damage/clean if clean else 0,'AUC':(1+(changes-damage)/errors-damage/clean)/2 if errors and clean else None,'errors':errors,'clean':clean,'changes':changes,'correct':correct,'wrong':changes-correct,'normal_damaged':damage}
