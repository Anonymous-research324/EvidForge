"""One independent holdout assessment after CAAFE features are sealed."""
from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_caafe_adapted_v3 import official_functions,apply,estimator
from unseen_table.common import load,dump,read_table,number,sha256,fresh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--holdout',required=True)
    p.add_argument('--report',required=True);a=p.parse_args();run=Path(a.run)
    freeze=load(run/'output_freeze.json')
    for name,h in freeze['files'].items():
        if sha256(run/name)!=h:raise ValueError('Features changed after output freeze')
    contract=load(Path(__file__).parent/'comparison_code_freeze_v3.json')
    for name,h in contract['files'].items():
        if sha256(Path(__file__).parent/name)!=h:raise ValueError('Comparison method changed after freeze')
    registration=load(run/'preregistration.json');input_path=next(iter(registration['inputs_sha256']))
    fields,train=read_table(input_path);cols=[c for c in fields if c not in ['id','class','row_id']]
    _,holdout=read_table(a.holdout)
    if {r['id'] for r in train}&{r['id'] for r in holdout}:raise ValueError('Training and holdout overlap')
    labels=load(run/'training_labels.json');mapping={c:i for i,c in enumerate(labels['classes'])}
    y=np.array(labels['labels']);y_test=np.array([mapping[r['class']] for r in holdout])
    x_train=pd.read_csv(run/'engineered_training_features.csv')
    raw_test=pd.DataFrame([[float(number(r[c])) if number(r[c]) is not None else np.nan for c in cols] for r in holdout],columns=cols)
    _,runner=official_functions();x_test=apply((run/'selected_feature_code.py').read_text(),raw_test,runner)
    if list(x_train.columns)!=list(x_test.columns):raise ValueError('Holdout feature schema differs')
    model=estimator();model.fit(x_train,y);pred=model.predict(x_test);prob=model.predict_proba(x_test)
    auc=roc_auc_score(y_test,prob[:,1]) if len(mapping)==2 else roc_auc_score(y_test,prob,multi_class='ovr',average='macro')
    report={'method':'CAAFE adapted9B feature engineering','same_original_classifier':True,
        'macro_f1':float(f1_score(y_test,pred,average='macro')),'probability_auc':float(auc),
        'accuracy':float(accuracy_score(y_test,pred)),'test_rows':len(holdout),'seed':20260927,
        'actual9B_calls':freeze['actual9B_calls'],'learning_seconds':freeze['seconds'],
        'cell_repair_metrics_applicable':False,'holdout_opened_after_output_freeze':True,
        'formal_cell_reference_read':False,'tuning_on_holdout':False,'adaptations':registration['differences']}
    root=fresh(a.report);dump(root/'results.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
