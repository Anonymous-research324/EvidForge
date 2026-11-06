"""Only this independent once-only scorer reads formal feature references and holdout."""
from pathlib import Path
import sys,os,json,time,math,importlib.util
from common import ROOT,load,save,read_csv,digest,freeze
from catalog import CATALOG,NAMES,METHODS,XGB_PARAMETERS
from prediction import frame,evaluate_frames
def closed(name,method):
 folder=ROOT/'outputs'/name/method
 return (folder/'output_freeze.json').is_file() or (ROOT/'status/failures'/(name+'__'+method+'.json')).is_file()
def method_csv(name,method):
 if method=='dirty':return ROOT/'datasets'/name/'public/repair_dirty.csv'
 if method=='pure_llm':return ROOT/'outputs'/name/'primary_llm/pure_output.csv'
 if method=='v23':return ROOT/'outputs'/name/'primary_llm/v23_output.csv'
 return ROOT/'outputs'/name/method/'repair_output.csv'
def main():
 for n in NAMES:
  assert (ROOT/'datasets'/n/'input_freeze.json').is_file(),'Dataset not ready'
  assert (ROOT/'status'/('model_done_'+n+'.json')).is_file(),'Model stages not closed'
  for m in ['raha_holoclean','garf','cdc']:
   assert closed(n,m),n+'/'+m+' not closed'
 receipt=ROOT/'audit/scoring_started.json'
 if receipt.exists():raise ValueError('Formal scoring already started; preserve receipts')
 # Verify every frozen method output before consuming any reference.
 for n in NAMES:
  for folder in [ROOT/'outputs'/n/'primary_llm',ROOT/'outputs'/n/'tool_llm',ROOT/'outputs'/n/'raha_holoclean',ROOT/'outputs'/n/'garf',ROOT/'outputs'/n/'cdc',ROOT/'outputs'/n/'caafe']:
   f=folder/'output_freeze.json'
   if f.is_file():
    for rel,h in load(f)['files'].items():
     assert digest(folder/rel)==h,'Output changed '+str(folder/rel)
 save(receipt,{'started':time.time(),'all_methods_and_feature_code_closed':True,'only_reader':'score_downstream.py'})
 results=[]
 for name,spec in CATALOG.items():
  pub=ROOT/'datasets'/name/'public';private=ROOT/'datasets'/name/'private'
  metadata=load(pub/'dataset_metadata.json');cols=metadata['targets'];_,dirty=read_csv(pub/'repair_dirty.csv')
  _,reference=read_csv(private/'repair_reference.csv');_,holdout=read_csv(private/'holdout_reference.csv')
  assert [x['entity_id'] for x in dirty]==[x['entity_id'] for x in reference]
  y=[x['class'] for x in reference];yt=[x['class'] for x in holdout];test=frame(holdout,cols)
  for method in [*METHODS,'clean_reference']+(['caafe'] if spec['task']=='classification' else []):
   failure=ROOT/'status/failures'/(name+'__'+method+'.json')
   if failure.is_file():results.append({'dataset':name,'method':method,'status':'N/A','failure':load(failure)});continue
   row={'dataset':name,'method':method,'task':spec['task'],'status':'complete','classifier':'XGBoost','parameters':XGB_PARAMETERS}
   try:
    if method=='clean_reference':train_rows=reference
    elif method=='caafe':train_rows=dirty
    else:
     path=method_csv(name,method)
     if not path.is_file():raise FileNotFoundError('Missing frozen method output')
     _,train_rows=read_csv(path)
    assert [x['entity_id'] for x in train_rows]==[x['entity_id'] for x in reference]
    training=frame(train_rows,cols);valid=test.copy()
    if method=='caafe':
     engine=ROOT/'frozen_workspace/DataForge_UnseenTable_v23'
     sys.path[:0]=[str(engine),str(engine/'source_runtime'),str(engine/'application')]
     module=engine/'baselines/run_caafe_adapted_v4.py';spec2=importlib.util.spec_from_file_location('score_pinned_caafe',module);caafe=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(caafe)
     _,runner=caafe.official_functions();code=(ROOT/'outputs'/name/'caafe/selected_feature_code.py').read_text()
     training=caafe.apply(code,training,runner);valid=caafe.apply(code,valid,runner)
    row['metrics']=evaluate_frames(training,y,valid,yt,spec['task'],metadata['categorical_columns'])
    if method=='tool_llm':row['method_receipt']=load(ROOT/'outputs'/name/'tool_llm/receipt.json')
   except Exception as e:row.update(status='N/A',error=type(e).__name__+': '+str(e)[:500])
   results.append(row);print('DOWNSTREAM_SCORED',name,method,row['status'],flush=True)
  save(ROOT/'reports/final_seed1014/scores.json',results)
 from report_downstream import build
 build(results)
 save(ROOT/'status/scoring_complete.json',{'rows':len(results),'time':time.time(),'seed':1014,'single_seed_no_significance_claim':True})
if __name__=='__main__':main()
