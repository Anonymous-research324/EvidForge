"""Learn small-domain context evidence without identifiers or class labels.

Two predictors use only observed OTHER repair fields. Models are numerical JSON,
not pickle or model-generated arbitrary code. No predictions are new answers:
the executor still requires an existing candidate and a source-qualified rule.
"""
from collections import Counter
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import ExtraTreesClassifier
from learning.environment import RUNTIME
from unseen_table.common import canonical, number
from unseen_table.learn_transforms import apply_rule
from skill_runtime.codec import fit as fit_codec, encode, decode
from unseen_table.models import predict_family


def key(value):
    return canonical(value) if number(value) is not None else None


def contextual_value(row, column, domain, codec):
    choices = set()
    original = key(row[column])
    if original in domain:
        choices.add(original)
    for rule in codec['rules']:
        if column in rule['fields']:
            value = apply_rule(rule, row, column)
            value = key(value) if value is not None else None
            if value in domain:
                choices.add(value)
    return next(iter(choices)) if len(choices)==1 else None


def matrix(rows, fields, domains, codec):
    columns = []
    for column in fields:
        domain = domains[column]; index = {v:i for i,v in enumerate(domain)}
        columns.append([index.get(contextual_value(row,column,domain,codec),len(domain)) for row in rows])
    return np.asarray(columns,dtype=int).T


def fit(records, declaration):
    targets = declaration['targets']
    domains = {c:sorted({key(r['truth'][c]) for r in records if key(r['truth'][c]) is not None}) for c in targets}
    small = [c for c in targets if 1<len(domains[c])<=16]
    codec=fit_codec(records,declaration); models={}
    for target in small:
        fields=[c for c in small if c!=target]
        if not fields:
            continue
        x=matrix([r['dirty'] for r in records],fields,domains,codec)
        y=[key(r['truth'][target]) for r in records]
        nb=CategoricalNB(alpha=1.0,min_categories=[len(domains[c])+1 for c in fields]);nb.fit(x,y)
        forest=ExtraTreesClassifier(n_estimators=160,max_depth=8,min_samples_leaf=2,max_features=1.0,
                                   random_state=20260927,n_jobs=1);forest.fit(x,y)
        trees=[]
        for estimator in forest.estimators_:
            t=estimator.tree_;values=t.value[:,0,:];values=values/values.sum(axis=1,keepdims=True)
            trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),
                          'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'values':values.tolist()})
        tree_model={'classes':forest.classes_.tolist(),'trees':trees}
        assert np.max(np.abs(predict_family(tree_model,x)-forest.predict_proba(x)))<1e-10
        models[target]={'fields':fields,'classes':nb.classes_.tolist(),'class_log_prior':nb.class_log_prior_.tolist(),
                        'feature_log_prob':[a.tolist() for a in nb.feature_log_prob_],'forest':tree_model,
                        'source_class_counts':dict(Counter(y))}
    return {'domains':domains,'codec':encode(codec),'models':models,'protected_fields_used':False,
            'source_groups':sorted({r['group'] for r in records})}


def predict(rows, model):
    codec=decode(model['codec']); result={}
    for target, fitted in model['models'].items():
        x=matrix(rows,fitted['fields'],model['domains'],codec)
        scores=np.tile(np.asarray(fitted['class_log_prior']), (len(rows),1))
        for i, conditional in enumerate(fitted['feature_log_prob']):
            scores+=np.asarray(conditional)[:,x[:,i]].T
        scores=np.exp(scores-scores.max(axis=1,keepdims=True));scores/=scores.sum(axis=1,keepdims=True)
        trees=predict_family(fitted['forest'],x)
        for row,(nb,tree) in enumerate(zip(scores,trees)):
            nb_best=str(fitted['classes'][int(np.argmax(nb))])
            tree_best=str(fitted['forest']['classes'][int(np.argmax(tree))])
            result[(row,target)]={'nb':dict(zip(fitted['classes'],map(float,nb))),
                'tree':dict(zip(fitted['forest']['classes'],map(float,tree))),
                'agrees':nb_best==tree_best,'domain':model['domains'][target],
                'observed_fraction':float(np.mean([x[row,i]!=len(model['domains'][c]) for i,c in enumerate(fitted['fields'])])),
                'codec':codec}
    return result
