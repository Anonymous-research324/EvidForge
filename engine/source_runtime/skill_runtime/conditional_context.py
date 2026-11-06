"""Source-trained conditional covariance evidence for exact candidate ranking.

No regression prediction is written into a table. All target values remain
members of the original observed-value candidate pool. Unknown/ambiguous cases
are kept. The model excludes the target itself, IDs and classification labels.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from unseen_table.common import number,canonical,equal
from .codec import possible


def fit(records,declaration,mode='raw'):
    cols=[c for c in declaration['targets'] if sum(number(r['truth'][c]) is not None for r in records)>=10]
    if len(cols)<3:return {}
    matrix=np.array([[float(number(r['truth'][c])) if number(r['truth'][c]) is not None else np.nan for c in cols] for r in records])
    center=np.nanmedian(matrix,axis=0);raw_scale=np.maximum(np.nanstd(matrix,axis=0),1e-8)
    matrix=np.where(np.isfinite(matrix),matrix,center)
    transformed=matrix if mode=='raw' else np.sign(matrix)*np.log1p(np.abs(matrix)/raw_scale)
    mean=transformed.mean(axis=0);scale=np.maximum(transformed.std(axis=0),1e-8)
    z=(transformed-mean)/scale
    estimator=LedoitWolf().fit(z)
    return {'columns':cols,'mode':mode,'raw_scale':raw_scale.tolist(),'mean':mean.tolist(),
        'scale':scale.tolist(),'covariance':estimator.covariance_.tolist(),
        'shrinkage':float(estimator.shrinkage_),'training_groups':sorted({r['group'] for r in records}),
        'fit_uses_only_authorized_training_sources':True,'target_self_excluded':True}


def transform(value,i,model):
    value=float(value)
    if model['mode']!='raw':value=np.sign(value)*np.log1p(abs(value)/model['raw_scale'][i])
    return (value-model['mean'][i])/model['scale'][i]


def conditional(target,trusted,model):
    cols=model.get('columns',[])
    if target not in cols:return None
    i=cols.index(target);indices=[];values=[]
    for j,col in enumerate(cols):
        if j==i or col not in trusted:continue
        n=number(trusted[col])
        if n is not None:indices.append(j);values.append(transform(n,j,model))
    if len(indices)<3:return None
    covariance=np.array(model['covariance']);cross=covariance[i,indices]
    source=covariance[np.ix_(indices,indices)]+np.eye(len(indices))*1e-8
    gain=np.linalg.solve(source,cross)
    mean=float(gain @ np.array(values));variance=max(1e-6,float(covariance[i,i]-gain @ cross))
    return {'mean_standardized':mean,'uncertainty_standardized':variance**.5,
            'trusted_other_fields':len(indices),'target_index':i,'target_excluded':True}


def hypotheses(row,cell,trusted,model,codec):
    col=cell['column'];old=row[col]
    if number(old) is None:return {}
    prediction=conditional(col,trusted,model)
    if prediction is None:return {}
    feasible={canonical(p['value']):p['value'] for p in cell['options']
              if number(p['value']) is not None and possible(old,p['value'],col,codec)}
    if len(feasible)<2:return {}
    ranked=sorted((abs(transform(number(v),prediction['target_index'],model)-prediction['mean_standardized'])
                  /prediction['uncertainty_standardized'],key,v) for key,v in feasible.items())
    if equal(old,ranked[0][2]):return {}
    best,runner=ranked[0][0],ranked[1][0];result={}
    for gap in [1,2,4,8]:
        if best<=1 and runner-best>=gap:
            result['conditional_'+model['mode']+'_gap_'+str(gap)]={'value':ranked[0][2],
                'best_standardized_residual':best,'runner_standardized_residual':runner,
                'trusted_other_fields':prediction['trusted_other_fields'],'target_self_excluded':True,
                'estimated_continuous_mean_not_written':True}
    return result
