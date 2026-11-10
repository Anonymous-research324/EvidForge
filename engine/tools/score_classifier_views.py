"""Classify sealed training views after the complete suite is frozen."""
from pathlib import Path
import argparse,sys,json
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,read_table,number,sha256,fresh


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();name=a.dataset
    private=ROOT/'benchmarks'/name/'local_private';_,holdout=read_table(private/'downstream_holdout_reference.csv');results={}
    for arm in ['pure','system']:
        view=ROOT/'work'/(name+'_'+arm+'_classifier_view_20261013');frozen=load(view/'view_freeze.json')
        for n,h in frozen['files'].items():
            if sha256(view/n)!=h:raise ValueError('Classifier view changed')
        fields,rows=read_table(view/'classifier_training_view.csv');cols=[c for c in fields if c not in ['id','class']]
        if {r['id'] for r in rows}&{r['id'] for r in holdout}:raise ValueError('Training and holdout overlap')
        def matrix(rs):return [[float(number(r[c])) if number(r[c]) is not None else np.nan for c in cols] for r in rs]
        enc=LabelEncoder();y=enc.fit_transform([r['class'] for r in rows]);test=enc.transform([r['class'] for r in holdout])
        clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20260927))
        clf.fit(matrix(rows),y);pred=clf.predict(matrix(holdout));prob=clf.predict_proba(matrix(holdout))
        auc=roc_auc_score(test,prob[:,1]) if len(enc.classes_)==2 else roc_auc_score(test,prob,multi_class='ovr',average='macro')
        results[arm]={'macro_f1':float(f1_score(test,pred,average='macro')),'probability_auc':float(auc),
            'selected_training_cv':load(view/'training_cv_selection.json')['selected']}
    root=fresh(ROOT/'reports'/(name+'_classifier_views_20261013'))
    report={'results':results,'same_original_classifier':True,'no_feature_generation':True,'core_repair_outputs_unchanged':True,
        'only_train_class_labels_for_selection':True,'test_only_scorer':True,'single_seed':20260927}
    dump(root/'results.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
