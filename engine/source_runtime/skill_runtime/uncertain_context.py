"""Compare anchored exact candidates using uncertain OTHER fields.

An unapproved guess never becomes a trusted cell. Its mixture variance remains
in the conditional observation covariance, with a reliability variance floor.
Conditional means are numerical evidence only and are never repair values.
"""
import numpy as np
from unseen_table.common import equal, number, canonical
from .codec import possible
from .conditional_context import transform


def compare(row,cells,approved,model,codec,*,rounds=3):
    if not model:return {}
    columns=model['columns'];cov=np.asarray(model['covariance']);families={}
    index={c['column']:c for c in cells}
    for col in columns:
        cell=index[col];old=row[col];i=columns.index(col)
        # Missing continuous values are unresolved; do not manufacture context
        # from a guessed point or mean for them.
        if not old.strip():continue
        if col in approved:
            value=number(approved[col])
            if value is not None:
                families[col]={'fixed':True,'values':[approved[col]],'z':np.array([transform(value,i,model)]),'weights':np.ones(1)}
            continue
        options={canonical(q['value']):q for q in cell['options']
                 if number(q['value']) is not None and (equal(old,q['value']) or possible(old,q['value'],col,codec))}
        if len(options)<2:continue
        qs=list(options.values());scores=np.clip(np.array([q['score'] for q in qs]),1e-5,1-1e-5)
        # Source-trained candidate scores supply a prior; renormalizing them is
        # NOT a permission score. Candidate uncertainty is explicitly retained.
        logits=np.log(scores/(1-scores));weights=np.exp(logits-logits.max());weights/=weights.sum()
        families[col]={'fixed':False,'values':[q['value'] for q in qs],
                       'z':np.array([transform(number(q['value']),i,model) for q in qs]),
                       'weights':weights,'log_prior':logits,'reliability':float(scores.max())}
    trail={};evidence={}
    for step in range(rounds):
        next_weights={}
        for col,items in families.items():
            if items['fixed'] or number(row[col]) is None:continue
            i=columns.index(col);others=[c for c in families if c!=col]
            if len(others)<3:continue
            ids=[columns.index(c) for c in others];means=[];noise=[]
            for c in others:
                f=families[c];mean=float(f['weights'] @ f['z']);variance=float(f['weights'] @ (f['z']-mean)**2)
                if not f['fixed']:
                    variance=max(variance,(1-f['reliability'])*float(cov[columns.index(c),columns.index(c)]))
                means.append(mean);noise.append(variance)
            source=cov[np.ix_(ids,ids)]+np.diag(noise)+np.eye(len(ids))*1e-8
            gain=np.linalg.solve(source,cov[i,ids]);mu=float(gain@np.asarray(means))
            variance=max(1e-6,float(cov[i,i]-gain@cov[i,ids]));residual=np.abs(items['z']-mu)/variance**.5
            logposterior=items['log_prior']-.5*np.minimum(900,residual**2)
            order=np.argsort(-logposterior,kind='stable');a,b=int(order[0]),int(order[1])
            weights=np.exp(logposterior-logposterior.max());weights/=weights.sum();next_weights[col]=weights
            best=items['values'][a];trail.setdefault(col,[]).append(best)
            evidence[col]={'value':best,'log_score_gap':float(logposterior[a]-logposterior[b]),
                           'best_standardized_residual':float(residual[a]),'other_fields':len(others),
                           'fixed_other_fields':sum(families[c]['fixed'] for c in others),
                           'target_self_excluded':True,'rounds':step+1,
                           'stable_last_two':len(trail[col])>=2 and equal(trail[col][-1],trail[col][-2]),
                           'unapproved_values_remain_uncertain':True,'predicted_mean_not_written':True}
        for col,w in next_weights.items():families[col]['weights']=w
    return evidence
