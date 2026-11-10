"""Unlabelled numerical subpopulations for exact candidate comparisons.

Clusters use authorized teaching values, excluding identities/classes. Each
component has shrinkage covariance. Query targets are excluded from membership
inference; component/conditional means can never become repair proposals.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from unseen_table.common import number,canonical,equal
from .conditional_context import fit as fit_global,transform
from .codec import possible


def fit(records,declaration,mode='signed_log',clusters=2):
    global_model=fit_global(records,declaration,mode)
    if not global_model or len(records)<clusters*8:return {}
    cols=global_model['columns']
    z=np.array([[transform(number(r['truth'][c]),i,global_model) for i,c in enumerate(cols)] for r in records])
    if not np.all(np.isfinite(z)):return {}
    labels=KMeans(clusters,random_state=20260927,n_init=5).fit_predict(z);parts=[]
    for k in range(clusters):
        ids=np.flatnonzero(labels==k)
        if len(ids)<8:return {}
        estimator=LedoitWolf().fit(z[ids]);parts.append({'weight':len(ids)/len(z),'center':estimator.location_.tolist(),
            'covariance':estimator.covariance_.tolist(),'source_groups':len({records[int(j)]['group'] for j in ids})})
    return {'global':global_model,'components':parts,'clusters':clusters,'mode':mode,
            'training_groups':sorted({r['group'] for r in records}),'class_labels_used':False}


def hypotheses(row,cell,trusted,model,codec):
    if not model or number(row[cell['column']]) is None:return {}
    global_model=model['global'];cols=global_model['columns'];col=cell['column'];old=row[col]
    if col not in cols:return {}
    i=cols.index(col);ids=[j for j,c in enumerate(cols) if j!=i and number(trusted.get(c,'')) is not None]
    if len(ids)<3:return {}
    x=np.array([transform(number(trusted[cols[j]]),j,global_model) for j in ids]);components=[]
    for part in model['components']:
        center=np.array(part['center']);cov=np.array(part['covariance']);source=cov[np.ix_(ids,ids)]+np.eye(len(ids))*1e-8
        sign,logdet=np.linalg.slogdet(source)
        if sign<=0:continue
        delta=x-center[ids];gain=np.linalg.solve(source,cov[i,ids]);mu=float(center[i]+gain@delta)
        variance=max(1e-6,float(cov[i,i]-gain@cov[i,ids]))
        membership=np.log(part['weight'])-.5*(logdet+float(delta@np.linalg.solve(source,delta)))
        components.append((membership,mu,variance))
    if not components:return {}
    weights=np.array([m[0] for m in components]);weights=np.exp(weights-weights.max());weights/=weights.sum()
    values={canonical(q['value']):q['value'] for q in cell['options'] if number(q['value']) is not None
            and (equal(old,q['value']) or possible(old,q['value'],col,codec))}
    if len(values)<2:return {}
    ranked=[]
    for value in values.values():
        z=transform(number(value),i,global_model)
        terms=np.array([np.log(max(w,1e-100))-.5*(np.log(v)+(z-mu)**2/v) for w,(_,mu,v) in zip(weights,components)])
        peak=terms.max();logscore=float(peak+np.log(np.exp(terms-peak).sum()))
        residual=min(abs(z-mu)/v**.5 for (_,mu,v),w in zip(components,weights) if w>=.05)
        ranked.append((logscore,value,residual))
    ranked.sort(key=lambda q:(-q[0],q[1]));winner,runner=ranked[:2]
    if equal(old,winner[1]) or winner[2]>1:return {}
    return {f"mixture_{model['mode']}_k{model['clusters']}_gap{gap}":{'value':winner[1],
        'log_score_gap':winner[0]-runner[0],'trusted_other_fields':len(ids),'target_self_excluded':True,
        'class_labels_used':False,'predicted_mean_not_written':True}
        for gap in [3,4,5] if winner[0]-runner[0]>=gap}
