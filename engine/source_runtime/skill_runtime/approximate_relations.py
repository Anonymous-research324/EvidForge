"""Learn soft numerical relations to compare exact candidates, never impute means.

Every relation is fitted on authorized, source-separated clean teaching values.
Its residual band is source-cross-checked. Runtime uses only trusted OTHER
fields; IDs/classes and the query's own target value cannot enter a predictor.
These hypotheses remain pending clause qualification on held source groups.
"""
import math
import numpy as np
from sklearn.model_selection import GroupKFold
from unseen_table.common import number, canonical, equal
from .codec import possible


def basis(x, kind, center, scale):
    if kind == 'log_affine':
        if np.any(x <= 0): return None
        x = np.log(x)
    z = (x - center) / scale
    return np.column_stack([np.ones(len(z)), z, z*z]) if kind == 'quadratic' else np.column_stack([np.ones(len(z)), z])


def estimate(x, spec):
    xx = np.array([x], dtype=float)
    if spec['kind'] == 'log_affine' and x <= 0: return None
    matrix = basis(xx,spec['kind'],spec['center'],spec['scale'])
    value = float((matrix @ np.array(spec['coefficients']))[0])
    if spec['kind'] == 'log_affine':
        if value > 700: return None
        value = math.exp(value)
    return value if math.isfinite(value) else None


def fit(records, declaration):
    targets = declaration['targets'];models=[]
    matrix = {c:[number(r['truth'][c]) for r in records] for c in targets}
    for target in targets:
        for source in targets:
            if source == target: continue
            ids = [i for i in range(len(records)) if matrix[source][i] is not None and matrix[target][i] is not None]
            if len(ids) < 12: continue
            x=np.array([float(matrix[source][i]) for i in ids]);y=np.array([float(matrix[target][i]) for i in ids])
            groups=[records[i]['group'] for i in ids]
            if len(set(groups)) < 10 or len(np.unique(x)) < 8: continue
            variance=float(np.var(y))
            if variance <= 1e-12: continue
            best=None
            for kind in ['affine','quadratic','log_affine']:
                if kind == 'log_affine' and (np.any(x <= 0) or np.any(y <= 0)): continue
                predictions=np.zeros(len(y));valid=True
                for train, check in GroupKFold(3).split(x,groups=groups):
                    tx=np.log(x[train]) if kind=='log_affine' else x[train]
                    center=float(tx.mean());scale=max(1e-8,float(tx.std()))
                    design=basis(x[train],kind,center,scale)
                    yy=np.log(y[train]) if kind=='log_affine' else y[train]
                    coefficients=np.linalg.lstsq(design,yy,rcond=None)[0]
                    pred=basis(x[check],kind,center,scale) @ coefficients
                    if kind=='log_affine':pred=np.exp(np.clip(pred,-700,700))
                    if not np.all(np.isfinite(pred)):valid=False;break
                    predictions[check]=pred
                if not valid: continue
                mse=float(np.mean((predictions-y)**2));r2=1-mse/variance
                if r2 < .95: continue
                residuals=np.abs(predictions-y)
                band=max(float(np.quantile(residuals,.95)),float(np.std(y))*.01,1e-8)
                if best is None or mse < best['source_cross_checked_mse']:
                    tx=np.log(x) if kind=='log_affine' else x
                    center=float(tx.mean());scale=max(1e-8,float(tx.std()))
                    yy=np.log(y) if kind=='log_affine' else y
                    coefficients=np.linalg.lstsq(basis(x,kind,center,scale),yy,rcond=None)[0]
                    best={'source':source,'target':target,'kind':kind,'center':center,'scale':scale,
                        'coefficients':coefficients.tolist(),'residual_band':band,
                        'source_cross_checked_r2':r2,'source_cross_checked_mse':mse,
                        'source_min':float(x.min()),'source_max':float(x.max()),'source_groups':len(set(groups)),
                        'target_self_excluded':True}
            if best is not None:models.append(best)
    return models


def hypotheses(row, cell, trusted, models, codec):
    col=cell['column'];old=row[col]
    if not old.strip(): return {}
    exact={canonical(p['value']):p['value'] for p in cell['options']
           if number(p['value']) is not None and possible(old,p['value'],col,codec)}
    if len(exact)<2:return {}
    votes={};proofs=[]
    for spec in models:
        source=spec['source']
        if spec['target'] != col or source == col or source not in trusted:continue
        x=number(trusted[source])
        if x is None or not spec['source_min'] <= float(x) <= spec['source_max']:continue
        prediction=estimate(float(x),spec)
        if prediction is None:continue
        distances=sorted([(abs(float(number(v))-prediction)/spec['residual_band'],k)
                          for k,v in exact.items()])
        if distances[0][0] <= 1 and distances[1][0] >= 2:
            choice=distances[0][1];votes[source]=choice
            proofs.append({'source':source,'choice':choice,'best_band_residual':distances[0][0],
                           'second_band_residual':distances[1][0],'relation':spec['kind'],
                           'source_cross_checked_r2':spec['source_cross_checked_r2']})
    counts={v:sum(choice==v for choice in votes.values()) for v in set(votes.values())}
    if not counts:return {}
    winner=max(counts,key=lambda v:(counts[v],v))
    if equal(old,exact[winner]):return {}
    result={}
    for minimum in [1,2,3]:
        if counts[winner]>=minimum and len(counts)==1:
            result['soft_relation_consensus_'+str(minimum)]={'value':exact[winner],'proofs':proofs,
                'distinct_other_fields':counts[winner],'estimate_not_written':True}
    return result
