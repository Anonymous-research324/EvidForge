"""Choose a classifier-only missing-value view by grouped training-label CV.

Protected classification labels are legitimate downstream supervision here;
they never enter the cell repair model. Core repaired CSVs are unchanged.
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,roc_auc_score
from .environment import RUNTIME
from unseen_table.common import load,number
from unseen_table.rank_candidates import predict_rows
from skill_runtime.package import open_skill

THRESHOLDS=[None,.5,.7,.8,.9,.95,.98,.99]


def select(rows,path):
    config=load(Path(path)/'configuration.json');_,base,*_=open_skill(config['base_skill_path'])
    targets=base['declaration']['targets'];identities=base['declaration']['entity_fields']
    x=np.array([[float(number(r[c])) if number(r[c]) is not None else np.nan for c in targets] for r in rows])
    risk=np.zeros_like(x);lookup={c:i for i,c in enumerate(targets)}
    for c in predict_rows(rows,base):risk[c['row'],lookup[c['column']]]=c['error_probability']
    enc=LabelEncoder();y=enc.fit_transform([r['class'] for r in rows]);groups=np.array(['|'.join(r[c] for c in identities) for r in rows])
    # Split duplicates together. Predeclared folds remain identical for every
    # threshold; detector learning uses only the disjoint authorized20% source.
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260927).split(x,y,groups))
    trials=[];matrices=[]
    for threshold in THRESHOLDS:
        matrix=x.copy();flags=np.zeros_like(risk,dtype=bool) if threshold is None else (risk>=threshold)&np.isfinite(x)
        matrix[flags]=np.nan;scores=[]
        for u,v in cv:
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20260927))
            model.fit(matrix[u],y[u]);prediction=model.predict(matrix[v]);prob=model.predict_proba(matrix[v])
            auc=roc_auc_score(y[v],prob[:,1]) if len(enc.classes_)==2 else roc_auc_score(y[v],prob,multi_class='ovr',average='macro')
            scores.append([f1_score(y[v],prediction,average='macro'),auc])
        f1,auc=np.mean(scores,axis=0);trials.append({'threshold':threshold,'CV_macro_f1':float(f1),'CV_probability_auc':float(auc),'masked_training_cells':int(flags.sum())});matrices.append(matrix)
    best=max(range(len(trials)),key=lambda i:(trials[i]['CV_macro_f1']+trials[i]['CV_probability_auc'],-trials[i]['masked_training_cells']))
    if sum(trials[best][k] for k in ['CV_macro_f1','CV_probability_auc'])<=sum(trials[0][k] for k in ['CV_macro_f1','CV_probability_auc'])+1e-12:best=0
    chosen=trials[best];output=[dict(r) for r in rows]
    for i,r in enumerate(output):
        for j,c in enumerate(targets):
            if np.isfinite(x[i,j]) and not np.isfinite(matrices[best][i,j]):r[c]=''
    return output,{'selected':chosen,'all_candidates':trials,'seed':20260927,'grouped_training_CV':True,
        'holdout_or_formal_cell_reference_read':False,'classifier_settings_changed':False,'core_repaired_table_modified':False,
        'mask_is_classifier_training_representation':'Not an exact repair or a deployment quality claim',
        'labels_used':'observed training class only, never cell repair learning or test labels'}
