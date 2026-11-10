"""Pinned official CAAFE prompt and AST executor, explicitly adapted9B/XGBoost."""
from pathlib import Path
import sys,os,json,subprocess,importlib.util,time,ast
os.environ['EVIDFORGE_SUPERVISED_COMPONENT']='caafe'
from common import ROOT,load,save,read_csv,write_csv,freeze,digest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from prediction import frame,evaluate_frames
from catalog import CATALOG
import seed_runtime
seed_runtime.install()
module=ROOT/'frozen_workspace/DataForge_UnseenTable_v23/baselines/run_caafe_adapted_v4.py'
spec=importlib.util.spec_from_file_location('pinned_caafe_adapter',module);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
from skill_runtime import client
def main(name):
 if CATALOG[name]['task']!='classification':raise ValueError('CAAFE classification supplement only')
 pub=ROOT/'datasets'/name/'public';metadata=load(pub/'dataset_metadata.json');cols=metadata['targets']
 _,rows=read_csv(pub/'repair_dirty.csv');_,labels=read_csv(ROOT/'datasets'/name/'downstream_training/labels.csv')
 assert [r['entity_id'] for r in rows]==[r['entity_id'] for r in labels]
 y=np.array([r['class'] for r in labels]);groups=np.array([r['entity_id'] for r in rows]);df=frame(rows,cols)
 out=ROOT/'outputs'/name/'caafe';out.mkdir(parents=True,exist_ok=False)
 prompt_function,runner=official.official_functions()
 description='Supervised classification using dirty numeric/ordinal-encoded feature columns. No clean original query rows. Target is not a feature.'
 ds=['dataset',None,None,[],cols+['class'],{},description]
 samples='\n'.join(f'{c} (float64): missing frequency {df[c].isna().mean():.3f}; current DIRTY values {df[c].head(10).tolist()}' for c in cols)
 prompt=prompt_function(df,ds,iterative=1,data_description_unparsed=description,samples=samples)
 summary=load(ROOT/'outputs'/name/'primary_llm/shared_summary.json')
 def cv(code):
  scores=[]
  folds=StratifiedGroupKFold(3,shuffle=True,random_state=1014)
  for u,v in folds.split(df,y,groups):
   a=official.apply(code,df.iloc[u].reset_index(drop=True),runner);b=official.apply(code,df.iloc[v].reset_index(drop=True),runner)
   if list(a.columns)!=list(b.columns):raise ValueError('Feature schema changed between folds')
   m=evaluate_frames(a,y[u],b,y[v],'classification',metadata['categorical_columns'])
   if m['AUC'] is None:raise ValueError('Undefined source CV AUC')
   scores.append([m['macro_F1'],m['AUC']])
  return np.mean(scores,axis=0).tolist()
 save(out/'preregistration.json',{'method':'CAAFE adapted9B + fixed XGBoost','official_commit':'bb65b4d517bd86056352ad2ee98be6655f2d5aec','iterations':5,
  'adaptations':['same Qwen3.5-9B BF16/temp0/seed1014','3-fold grouped supervised training CV macroF1+AUC','same fixed XGBoost','bounded official AST expression executor','no holdout reads'],
  'training_label_access_only':True,'formal_cell_reference_read':False,'holdout_read':False,'seed':1014})
 code='';before=cv(code);history=[]
 current={f'row_{i}':{c:'' if not np.isfinite(df.iloc[i][c]) else str(float(df.iloc[i][c])) for c in cols} for i in range(min(10,len(rows)))}
 system='Use the provided official CAAFE prompt to propose bounded vectorized numeric feature expressions for df. df contains FLOAT feature columns only and never target labels. Return assignments objects name/expression; do not import, loop, read files, access network, use labels as features, or change rows. np,pd,df already exist. Use numeric vector expressions; dirty nonnumeric inputs are already NaN. Do not use string cleaning or to_numeric. The supplied training CV feedback is independent of all holdout samples.'
 schema={'type':'object','properties':{'assignments':{'type':'array','minItems':1,'maxItems':8,'items':{'type':'object','properties':{'name':{'type':'string','pattern':'^[A-Za-z][A-Za-z0-9_]{0,39}$'},'expression':{'type':'string','maxLength':512}},'required':['name','expression'],'additionalProperties':False}}},'required':['assignments'],'additionalProperties':False}
 for attempt in range(5):
  payload={'rows':current,'learned_summary':{**summary,'classification_prompt':prompt,'target_name':'class','accepted_feature_code':code},
   'previous_answer':{'training_group_CV_macroF1_AUC':before,'history':history},'batch':attempt}
  answer=client.call(system,payload,schema,out/'model_trace'/f'a{attempt}',name.replace('_','').title()+f'DownstreamCAAFE1014A{attempt}',cols)
  proposal='\n'.join("df["+repr(x['name'])+"] = "+x['expression'] for x in answer['assignments'])
  accepted=False;error=None;after=None
  try:
   after=cv(code+'\n'+proposal);accepted=sum(after)>sum(before)+1e-12
  except Exception as e:error=type(e).__name__+': '+str(e)[:300]
  history.append({'attempt':attempt,'proposal':proposal,'accepted':accepted,'before':before,'after':after,'error':error})
  if accepted:code+='\n'+proposal;before=after
  save(out/'history.json',history);print('CAAFE_SOURCE',name,attempt+1,accepted,flush=True)
 (out/'selected_feature_code.py').write_text(code)
 save(out/'receipt.json',{'method':'CAAFE adapted9B','actual9B_calls':5,'classification_training_labels_only':True,'holdout_read':False,'formal_cell_reference_read':False})
 save(out/'output_freeze.json',{'files':freeze(out),'test_reference_read':False})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
