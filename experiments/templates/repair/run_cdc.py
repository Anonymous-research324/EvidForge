"""Published CDC conformal cleaner, group calibration inside authorized first10 only."""
from pathlib import Path
import argparse,os,sys,time,json,random,copy
from common import ROOT,SEED,load,save,read_csv,write_csv,freeze
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0,str(ROOT/'sources/cdc'))
from conformal_data_cleaning.cleaner.conformal import ConformalAutoGluonCleaner
from conformal_data_cleaning.cleaner import _base as cleanbase
from conformal_data_cleaning.utils import is_categorical as native_is_categorical
from conformal_inference.models.automl import autogluon as native_ag
random.seed(SEED);np.random.seed(SEED)
# The published sample-with-replacement heuristic otherwise calls EVERY feature
# categorical when teaching has <=200 rows. Cap its sample to actual source size.
cleanbase.is_categorical=lambda s:native_is_categorical(s,n_samples=min(1000,len(s)))
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=str(Path(os.fsdecode(args[0])).resolve())
  if '/datasets/' in p and '/private/' in p or '/local_private/' in p:raise PermissionError('Formal reference unavailable')
  if p.endswith(('development_bundle.json','validation_only.json')):raise PermissionError('Only first10 source may train')
sys.addaudithook(guard)
def main(name):
 start=time.monotonic();pub=ROOT/'datasets'/name/'public';out=ROOT/'outputs'/name/'cdc';out.mkdir(parents=True,exist_ok=False)
 cols=load(pub/'declaration.json')['targets'];groups=load(pub/'learning_only.json')['learning']
 records=[{**r,'source_group':g['group']} for g in groups for r in g['records']]
 fields,val=read_csv(pub/'validation_dirty.csv');_,formal=read_csv(pub/'repair_dirty.csv')
 rawteach=pd.DataFrame([{c:r['truth'][c] for c in cols} for r in records],columns=cols)
 categories=[c for c in cols if cleanbase.is_categorical(rawteach[c])]
 teach=rawteach.copy()
 for c in cols:
  teach[c]=teach[c].astype(object) if c in categories else pd.to_numeric(teach[c],errors='raise').astype(float)
 raw=val+formal;query=pd.DataFrame([{c:r[c] for c in cols} for r in raw],columns=cols)
 for c in cols:
  query[c]=query[c].replace('',np.nan).astype(object) if c in categories else pd.to_numeric(query[c],errors='coerce').astype(float)
 group_ids=np.array([r['source_group'] for r in records])
 def grouped_split(X,*a,**kw):
  assert len(X)==len(teach) and list(X.index)==list(teach.index),'Calibration may use teaching only'
  tr,ca=next(GroupShuffleSplit(n_splits=1,test_size=float(kw.get('test_size',.4)),random_state=SEED).split(X,groups=group_ids))
  return X.iloc[tr],X.iloc[ca]
 native_ag.train_test_split=grouped_split
 policy={'seed':SEED,'confidence_level':.9,'internal_calibration_fraction':.4,'calibration_grouped':True,'AutoGluon_learner_random_state':1014,'estimator_seeds':1014,
 'training_source_fraction':.1,'external_validation_used_for_fitting':False,'per_field_time_limit_seconds':60,'categorical_columns':categories,
 'hyperparameter_trials':10,'native_cleaner_code_unchanged':True,'type_sample_cap':'min(1000, authorized_source_rows), source-only small-table adaptation',
 'optional_model_families_available':['LightGBM','CatBoost','RandomForest','ExtraTrees','KNN'],'reuse_intermediate':False,'missing_prediction_reuse_shape_bug_avoided_using_native_option':True,'GPU_used':False,'test_gold_read':False}
 save(out/'method_freeze.json',policy)
 result=[dict(r) for r in raw];column_status=[];scores=np.zeros((len(raw),len(cols)))
 for j,c in enumerate(cols):
  if teach[c].nunique()<2:
   column_status.append({'column':c,'status':'abstained_constant_source'})
   continue
  # A cleaner for one target avoids mutation of other predictors' keyword dicts.
  cleaner=ConformalAutoGluonCleaner(confidence_level=.9,seed=SEED)
  try:
   cleaner.fit(teach,target_columns=[c],ci_ag_predictor_params={'path_prefix':out/'models'/c,'verbosity':1,'learner_kwargs':{'random_state':1014}},
     ci_calibration_size=.4,ci_ag_fit_params={'time_limit':60,'holdout_frac':.5,'num_cpus':4,'num_gpus':0,
       'hyperparameters':{'GBM':{'seed':1014},'CAT':{'random_seed':1014},'RF':{'random_state':1014},'XT':{'random_state':1014},'KNN':{}},
       'hyperparameter_tune_kwargs':{'num_trials':10},'ag_args_fit':{'num_cpus':4,'num_gpus':0}})
   cleaned,mask,sets=cleaner.transform(query,reuse_intermediate=False)
   for i in range(len(raw)):
    if bool(mask.iloc[i][c]):
     value=cleaned.iloc[i][c]
     if not pd.isna(value):result[i][c]=str(value);scores[i,j]=1.
   column_status.append({'column':c,'status':'fit_and_transformed','changed_or_imputed':int(mask[c].sum())})
  except Exception as e:
   # A computational failure is not a performance score. No output closes if any field crashes.
   save(out/'column_status.json',column_status+[{'column':c,'status':'failed','exception':type(e).__name__,'message':str(e)[:500]}])
   raise
  save(out/'column_status.json',column_status)
  print('CDC_FIELD_DONE',name,j+1,'/',len(cols),flush=True)
 write_csv(out/'validation_output.csv',fields,result[:len(val)])
 write_csv(out/'repair_output.csv',fields,result[len(val):])
 save(out/'receipt.json',{**policy,'elapsed_seconds':time.monotonic()-start,'source_groups':len(groups),'source_rows':len(records),'classification_labels_used':False})
 save(out/'output_freeze.json',{'files':freeze(out),'formal_reference_read':False})
 print('CDC_OUTPUT_CLOSED',name,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
