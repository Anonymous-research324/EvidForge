"""Relearn exact candidate comparisons with source-separated contextual features.

All reference values stay in local fitting. Gaussian predictions are features,
never table values. Candidate models serialize to numeric JSON trees, no code
generation or pickle. A source group never crosses feature-fitting boundaries.
"""
from collections import defaultdict
import math
import numpy as np
from sklearn.model_selection import GroupKFold
from unseen_table.common import equal,number,fingerprint,dump,load,sha256
from unseen_table.learn_transforms import infer
from unseen_table.learn_value_modes import fit as fit_modes
from unseen_table.learn_relations import fit as fit_relations
from unseen_table.numeric_context import fit as fit_context,evidence
from unseen_table.row_state import discover
from unseen_table.candidate_pool import candidates
from unseen_table.features import candidate_features
from unseen_table.models import fit_classifier,predict
from .conditional_context import fit as fit_gaussian,conditional,transform


def cheap_fit(records,declaration):
    base=infer(records,declaration)
    base['training_cells']=sum(len(r['truth']) for r in records)
    base['value_modes']=fit_modes(records,base)
    base['relations']=fit_relations(records,declaration)
    base['row_state']={'points':discover(records,declaration)}
    base['numeric_context']=fit_context(records,base)
    models={mode:fit_gaussian(records,declaration,mode) for mode in ['raw','signed_log']}
    return {'base':base,'conditional_models':models,'training_groups':sorted({r['group'] for r in records})}


def pack(rows,fitted):
    base=fitted['base'];features=[];cells=[]
    for i,row in enumerate(rows):
        context=evidence(row,base);context['states']={}
        for col in base['declaration']['targets']:
            options=candidates(row,col,base,context['trusted'])
            contexts={mode:conditional(col,context['trusted']['values'],model) if model else None
                      for mode,model in fitted['conditional_models'].items()}
            start=len(features);old=row[col]
            for option in options:
                vector=candidate_features(row,col,option,base,context)
                extra=[]
                for mode,model in fitted['conditional_models'].items():
                    prediction=contexts[mode];new=number(option['value']);observed=number(old)
                    if prediction and new is not None:
                        j=prediction['target_index'];mu=prediction['mean_standardized'];sigma=prediction['uncertainty_standardized']
                        new_residual=min(30,abs(transform(new,j,model)-mu)/sigma)
                        old_residual=min(30,abs(transform(observed,j,model)-mu)/sigma) if observed is not None else 30
                        extra += [1,new_residual,old_residual,old_residual-new_residual,
                                  min(64,prediction['trusted_other_fields']),sigma]
                    else:extra += [0,0,0,0,0,0]
                features.append(vector+extra)
            cells.append({'row':i,'column':col,'original':old,'options':options,'feature_start':start,
                          'old_form':'blank' if not old.strip() else 'text' if number(old) is None else
                            'integer' if number(old)==number(old).to_integral() else 'decimal'})
    return features,cells


def training_rows(records):
    rows=[];truth=[];groups=[];weights=[]
    for r in records:
        for dirty,weight in [(r['dirty'],1.0),({**r['dirty'],**r['truth']},.2)]:
            rows.append(dirty);truth.append(r['truth']);groups.append(r['group']);weights.append(weight)
    return rows,truth,groups,weights


def train(records,declaration,*,iterations=45):
    x=[];y=[];weights=[];audit=[]
    groups=[r['group'] for r in records]
    for a,b in GroupKFold(3).split(records,groups=groups):
        teaching=[records[int(i)] for i in a];checking=[records[int(i)] for i in b]
        fitted=cheap_fit(teaching,declaration)
        rows,truth,query_groups,query_weights=training_rows(checking)
        fx,cells=pack(rows,fitted)
        for cell in cells:
            row,col=cell['row'],cell['column'];start=cell['feature_start'];options=cell['options']
            positive=[j for j,q in enumerate(options) if equal(q['value'],truth[row][col])]
            keep=[j for j,q in enumerate(options) if equal(q['value'],cell['original'])]
            negative=[j for j in range(len(options)) if j not in positive and j not in keep]
            # Include every exact positive/KEEP and bounded hard negatives. All
            # candidates remain available at prediction; no evaluation deletion.
            negative.sort(key=lambda j:(min(fx[start+j][-11],fx[start+j][-5]),
                               -options[j]['support'],options[j]['value']))
            selected=list(dict.fromkeys(positive+keep+negative[:10]))
            for j in selected:
                label=int(j in positive);x.append(fx[start+j]);y.append(label)
                weights.append(query_weights[row]*(1.0 if label else 1/max(1,len(selected)-len(positive))))
        audit.append({'training_groups':fitted['training_groups'],'query_groups':sorted({r['group'] for r in checking}),
                      'no_query_source_in_feature_fitting':not (set(fitted['training_groups'])&{r['group'] for r in checking})})
    model=fit_classifier(x,y,weights,seed=20260927,iterations=iterations)
    return {'ranker':model,'features':cheap_fit(records,declaration),'source_feature_folds':audit,
            'training_groups':sorted(set(groups)),'seed':20260927,'original_reference_cases_visible_to9B':False}


def predict_cells(rows,model):
    fx,cells=pack(rows,model['features']);scores=predict(model['ranker'],fx)
    for c in cells:
        start=c.pop('feature_start');options=c['options']
        for j,p in enumerate(options):p['score']=float(scores[start+j])
        ranked=sorted(options,key=lambda p:(-p['score'],p['kind']!='keep',p['value']))
        top=ranked[0]
        if not equal(top['value'],c['original']):
            top['kind']=next((k for k in ['source','relation','constant','affix','strip','casefold','scale','shift','mapping','domain','edit'] if k in top['kinds']),top['kind'])
        runner=ranked[1]['score'] if len(ranked)>1 else 0
        c.update(top=top,margin=top['score']-runner,tied=len(ranked)>1 and top['score']-runner<=1e-10,
                 candidate_count=len(options),error_probability=0.0)
    return cells
