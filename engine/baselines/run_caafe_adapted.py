"""CAAFE adapted to the same9B, current dirty rows and unchanged LR settings.

Uses the pinned official prompt/AST code executor and proposal→CV→feedback
loop. Transport/seed/CV folds differ explicitly; this is not paper replication.
Only training labels are used; holdout values/labels are never read here.
"""
from pathlib import Path
import ast,argparse,sys,time,json,hashlib,copy,signal
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,fresh,read_table,number,sha256
from skill_runtime import client


def official_functions():
    folder=Path(__file__).parent/'caafe_official_source'
    prompt_tree=ast.parse((folder/'caafe.py').read_text())
    executor_tree=ast.parse((folder/'run_llm_code.py').read_text())
    functions=[n for n in prompt_tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_prompt']
    functions += [n for n in executor_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['check_ast','run_llm_code']]
    namespace={'np':np,'pd':pd,'ast':ast,'copy':copy,'Optional':__import__('typing').Optional}
    # Executed functions are pinned OFFICIAL source; generated code undergoes
    # its official AST whitelist below and a bounded execution timeout.
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(folder), 'exec'),namespace)
    return namespace['get_prompt'],namespace['run_llm_code']


def apply(code,df,runner):
    tree=ast.parse(code)
    if len(list(ast.walk(tree)))>600:raise ValueError('Feature code exceeds bounded size')
    if any(isinstance(n,(ast.For,ast.While,ast.FunctionDef,ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp,ast.Import,ast.ImportFrom,ast.Lambda)) for n in ast.walk(tree)):
        raise ValueError('Use bounded vectorized feature code; iterative loops need a separate reviewed benchmark')
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and node.id.startswith('_'):raise ValueError('Private names are unavailable')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['arange','linspace','repeat','merge','join','pivot','plot']:
            raise ValueError('Unbounded allocation is unavailable in this bounded benchmark')
    def expired(signum,frame):raise TimeoutError('Feature execution exceeded five seconds')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,5)
    try:result=runner(code,df,convert_categorical_to_integer=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    if len(result)!=len(df) or len(result.columns)>len(df.columns)+20 or len(result.columns)==0:
        raise ValueError('Keep rows and bounded feature count')
    result=result.apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    return result


def estimator():return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
    LogisticRegression(C=1,max_iter=2000,random_state=20260927))


def cross_validate(df,y,code,runner):
    scores=[]
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=20260927)
    for u,v in cv.split(df,y):
        train=apply(code,df.iloc[u].reset_index(drop=True),runner)
        valid=apply(code,df.iloc[v].reset_index(drop=True),runner)
        if list(train.columns)!=list(valid.columns):raise ValueError('Train/validation feature schemas differ')
        model=estimator();model.fit(train,y[u]);prediction=model.predict(valid);prob=model.predict_proba(valid)
        auc=roc_auc_score(y[v],prob[:,1]) if len(set(y))==2 else roc_auc_score(y[v],prob,multi_class='ovr',average='macro')
        scores.append([float(accuracy_score(y[v],prediction)),float(auc)])
    return np.mean(scores,axis=0).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--shared-summary',required=True)
    p.add_argument('--out',required=True);p.add_argument('--remote-helper',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--iterations',type=int,default=5);a=p.parse_args();root=fresh(a.out)
    if not 1<=a.iterations<=10:raise ValueError('Bounded predeclared LLM budget')
    fields,rows=read_table(a.input);cols=[c for c in fields if c not in ['id','class','row_id']]
    df=pd.DataFrame([[float(number(r[c])) if number(r[c]) is not None else np.nan for c in cols] for r in rows],columns=cols)
    enc=LabelEncoder();y=enc.fit_transform([r['class'] for r in rows]);prompt_function,runner=official_functions()
    description='Classification on the current dirty numerical/ordinal table; all features are anonymous. Target class is supervised, ID excluded. Original clean reference rows are unavailable.'
    ds=['dataset',None,None,[],cols+['class'],{},description]
    samples='\n'.join(f'{c} (float64): missing frequency {df[c].isna().mean():.3f}; current DIRTY samples {df[c].head(10).tolist()}' for c in cols)
    prompt=prompt_function(df,ds,iterative=1,data_description_unparsed=description,samples=samples)
    summary=load(a.shared_summary);client.REMOTE=a.remote_helper
    dump(root/'preregistration.json',{'official_repository':'https://github.com/noahho/CAAFE',
      'official_commit':'bb65b4d517bd86056352ad2ee98be6655f2d5aec','method':'CAAFE adapted9B feature engineering',
      'differences':['same Qwen3.5-9B BF16 temp0 seed20260927','current DIRTY examples, no clean reference rows',
        'unchanged benchmark LR/imputer/scaler','5-fold repeated twice supervised training CV','bounded vectorized official-whitelisted code'],
      'inputs_sha256':{a.input:sha256(a.input),a.shared_summary:sha256(a.shared_summary)},'iterations':a.iterations,
      'classification_training_labels_only':True,'formal_cell_gold_read':False,'holdout_read':False,'cell_repair_metrics_applicable':False})
    code='';before=cross_validate(df,y,code,runner);history=[];start=time.monotonic()
    current={f'row_{i}':{c:rows[i][c] for c in cols} for i in range(min(10,len(rows)))}
    system='You are an expert data scientist generating vectorized pandas feature engineering. Follow the supplied CAAFE prompt. df contains feature columns only, never the target. Return JSON {"code":"..."}, no markdown, imports, loops, file/network access, or target-derived features. Use np/pd/df and vectorized operations. Do not change row counts. You may add up to10 features or drop redundant features.'
    schema={'type':'object','properties':{'code':{'type':'string','maxLength':4096}},'required':['code'],'additionalProperties':False}
    for i in range(a.iterations):
        payload={'rows':current,'learned_summary':{**summary,'classification_prompt':prompt,
            'target_name':'class','current_training_labels':y[:len(current)].tolist(),'accepted_feature_code':code},
            'batch':i,'previous_answer':{'supervised_training_CV_ACC_AUC':before,'history':history}}
        result=client.call(system,payload,schema,root/'model_trace'/f'iteration-{i}',a.tag+str(i),cols)
        proposed=result['code'];accepted=False;error=None;after=None
        try:
            after=cross_validate(df,y,code+'\n'+proposed,runner)
            accepted=sum(after)>sum(before)+1e-12
        except (ValueError,TypeError,KeyError,ZeroDivisionError,SyntaxError,TimeoutError,IndexError,AttributeError,OverflowError) as exc:error=str(exc)
        history.append({'iteration':i,'code':proposed,'accepted':accepted,'before_ACC_AUC':before,'after_ACC_AUC':after,'error':error})
        if accepted:code+='\n'+proposed;before=after
        dump(root/'history.json',history);print('CAAFE_ADAPTED',i+1,accepted,after,error,flush=True)
    transformed=apply(code,df,runner);transformed.to_csv(root/'engineered_training_features.csv',index=False)
    (root/'selected_feature_code.py').write_text(code)
    dump(root/'training_labels.json',{'labels':y.tolist(),'classes':enc.classes_.tolist()})
    dump(root/'output_freeze.json',{'files':{n:sha256(root/n) for n in ['selected_feature_code.py','engineered_training_features.csv','training_labels.json','history.json','preregistration.json']},
        'actual9B_calls':a.iterations,'seconds':time.monotonic()-start,'formal_cell_gold_read':False,'holdout_read':False,
        'original_training_rows_sent':False,'current_dirty_examples_sent':True,'same_original_classifier_settings':True})
    print('CAAFE_FEATURES_FROZEN_BEFORE_HOLDOUT',flush=True)


if __name__=='__main__':main()
