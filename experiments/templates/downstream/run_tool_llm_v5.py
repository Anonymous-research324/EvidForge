"""Same9B code generation+source feedback, without the V23 candidate engine."""
from pathlib import Path
import argparse,sys,subprocess,json,time,copy,os,re
os.environ['DF1014_TOOL_MAX_TOKENS']='8192'
from common import ROOT,SEED,load,save,read_csv,write_csv,digest,freeze,status
import seed_runtime
seed_runtime.install()
from skill_runtime import client
from sklearn.model_selection import GroupKFold
from unseen_table.common import equal,number
PROMPT="Write a complete executable Python function repair(rows, summary, teaching) that cleans erroneous CELLS in a table. This is NOT a classification task. There is NO single target column or hidden label to predict. Every key in every row is a repairable FEATURE COLUMN. Each of those columns may have errors. rows contains all feature columns, with string values, including blank strings and malformed strings. Preserve row order and all dictionary keys.\nteaching is provided to your function by the execution backend as paired training records: [{'dirty': {feature_name: string_value, ...}, 'truth': {same_feature_name: confirmed_string_value, ...}}, ...]. The truth dictionary confirms ALL feature columns of that teaching record. rows is a different collection of query records whose answers are hidden. Learn from teaching and the authorized summary. Do not assume query rows correspond to teaching rows and do not copy training answers by position.\nsummary contains feature-column information under summary['columns']; summary['learned_summary']['columns'] is an alias of the same information. Names in that mapping are the columns to repair, not classifier inputs plus a target label.\nReturn a list of dictionaries in the same order and shape as rows, with string values, or sparse {'repairs': {'row_0': {'feature_name': 'replacement'}, ...}}. Empty sparse repair dict means abstention. Change only cells for which your learned evidence supports a correction; preserve others.\nAllowed imports: re,math,decimal,statistics,collections,numpy,pandas,random,sklearn,sklearn.ensemble,sklearn.linear_model,sklearn.neighbors,sklearn.impute,sklearn.preprocessing,sklearn.pipeline,sklearn.metrics,sklearn.model_selection. Estimators use random_state=1014 and at most 4 CPU workers. No file/network lookup, IDs, labels, V23 engine, dataset-specific hard-coded rules, or global mutation.\nYou receive your previous actual program and source-only execution feedback. Repair its errors or improve it using that feedback. External validation and query answers remain hidden. Output ONLY complete multiline Python source, optionally in a python code fence, at most 100 lines, with minimal comments and no discussion. End with a complete return statement."
def worker(program,rows,summary,teaching,where):
 where.mkdir(parents=True,exist_ok=False)
 inp=where/'input.json';out=where/'output.json'
 save(inp,{'program':program,'rows':rows,'summary':summary,'teaching':teaching})
 with (where/'execution.log').open('x') as f:
  p=subprocess.run(['@RUN_PYTHON@','-B',str(ROOT/'code/tool_worker.py'),str(inp),str(out)],
     stdout=f,stderr=subprocess.STDOUT,timeout=120)
 if p.returncode:
  tail=(where/'execution.log').read_text(errors='replace').strip().splitlines()[-1][:350]
  raise RuntimeError('Program execution failed: '+tail)
 result=load(out)
 for a,b in zip(rows,result):
  for c in a:
   if not equal(a[c],b[c]) and b[c] and number(b[c]) is None:raise ValueError('Changed value is not finite numerical data')
 return result
def metric(records,outputs,cols):
 e=n=changes=exact=damaged=0
 for r,b in zip(records,outputs):
  for c in cols:
   wrong=not equal(r['dirty'][c],r['truth'][c]);change=not equal(r['dirty'][c],b[c])
   e+=wrong;n+=not wrong;changes+=change;exact+=wrong and change and equal(b[c],r['truth'][c]);damaged+=not wrong and change
 p=exact/changes if changes else 0;r=exact/e if e else 0
 return {'P':p,'R':r,'F1':2*p*r/(p+r) if p+r else 0,'damage':damaged/n if n else 0,'changes':changes}
def main(name):
 selected=load(ROOT/'outputs/v23'/name/'selection.json')
 summary=load(ROOT/'outputs'/name/'primary_llm/shared_summary.json')
 pub=ROOT/'datasets'/name/'public';cols=load(pub/'declaration.json')['targets']
 groups=load(pub/'learning_only.json')['learning']
 records=[{**r,'group':g['group']} for g in groups for r in g['records']]
 out=ROOT/'outputs'/name/'tool_llm';out.mkdir(parents=True,exist_ok=False);candidates=[];previous={}
 folds=list(GroupKFold(3).split(records,groups=[r['group'] for r in records]))
 for attempt in range(10):
  payload={'learned_summary':summary,'previous_answer':previous}
  schema={'type':'object','properties':{'program':{'type':'string','maxLength':24000}},'required':['program'],'additionalProperties':False}
  tag=name.replace('_','').title()+f'Tool1014ClarifiedV5A{attempt}'
  body={'_trace_tag':tag,'messages':[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(payload,separators=(',',':'))}],
        'max_tokens':4096,'seed':1014,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
  trace=out/'trace';trace.mkdir(exist_ok=True);save(trace/f'program_{attempt}_request.json',body)
  model_start=time.monotonic()
  proc=subprocess.run(['@RUN_PYTHON@','-B',str(ROOT/'code/model_service.py'),'call'],input=json.dumps(body),capture_output=True,text=True,timeout=330)
  if proc.returncode:raise RuntimeError('Model transport failed; retained helper receipt')
  response=json.loads(proc.stdout);save(trace/f'program_{attempt}_response.json',response)
  save(trace/f'program_{attempt}_timing.json',{'seconds':time.monotonic()-model_start,'usage':response.get('usage')})
  text=response['choices'][0]['message']['content'].strip()
  ticks=chr(96)*3
  if text.startswith(ticks):
   text=text.split('\n',1)[1] if '\n' in text else text[3:]
   if text.rstrip().endswith(ticks):text=text.rstrip()[:-3].rstrip()
  program=text
  if response['choices'][0]['finish_reason']!='stop':previous={'source_feedback_error':'TruncatedCode','instruction':'Return a shorter complete Python function.'}
  (out/f'program_{attempt}.py').write_text(program)
  predicted=[None]*len(records)
  try:
   import ast
   ast.parse(program)
   for f,(u,v) in enumerate(folds):
    teaching=[{'dirty':records[int(i)]['dirty']|{},'truth':records[int(i)]['truth']|{}} for i in u]
    for r in teaching:
     r['dirty']={c:r['dirty'][c] for c in cols};r['truth']={c:r['truth'][c] for c in cols}
    rows=[{c:records[int(i)]['dirty'][c] for c in cols} for i in v]
    result=worker(program,rows,summary,teaching,out/'source_checks'/f'a{attempt}_f{f}')
    for i,r in zip(v,result):predicted[int(i)]=r
   m=metric(records,predicted,cols);candidate={'attempt':attempt,'metrics':m,'program_path':str(out/f'program_{attempt}.py'),'program_sha256':digest(out/f'program_{attempt}.py'),'accepted_execution':True}
   previous={'source_feedback':m,'source_eval_note':'group-held query rows; common summary is from authorized learning only; external validation hidden'}
  except Exception as e:
   candidate={'attempt':attempt,'accepted_execution':False,'error':type(e).__name__}
   previous={'source_feedback_error':type(e).__name__,'exception_message':str(e)[:300],'instruction':'Use only permitted APIs and preserve output shape. Source execution log stays local.'}
  previous={'previous_program':program,'feedback':previous,'task_reminder':'Repair every supported erroneous feature cell. There is no separate target label. Query rows and teaching rows are different records.'}
  candidates.append(candidate);save(out/'source_candidates.json',candidates)
  print('TOOL_SOURCE',name,attempt+1,candidate.get('metrics',candidate.get('error')),flush=True)
 eligible=[x for x in candidates if x['accepted_execution'] and x['metrics']['P']>=.98 and x['metrics']['damage']<=.005]
 valid=[x for x in candidates if x['accepted_execution']]
 if not valid:
  import shutil
  pure=ROOT/'outputs'/name/'primary_llm/pure_output.csv'
  shutil.copyfile(pure,out/'repair_output.csv')
  save(out/'receipt.json',{'method':'9B code-tool agent with explicit pure9B fallback when no executable program',
    'seed':1014,'actual_learning_calls':10,'new_formal_model_calls':0,'fallback':True,'fallback_reason':'all generated programs failed source execution',
    'fallback_source':str(pure),'fallback_output_sha256':digest(pure),'native_code_repair_succeeded':False,'formal_reference_read':False,
    'not_a_zero_score_for_runtime_failure':True,'protocol_revision':'downstream-feature-cell-task-v1','previous_actual_program_in_feedback':True,'prior_v4_generation_calls_retained':0,'qualification':'research comparison only; no deploy registration'})
  save(out/'program_selection_freeze.json',{'selected':'no-executable-program→pure9B safety fallback','decision_uses_source_execution_only':True,'formal_reference_read':False})
  save(out/'output_freeze.json',{'files':freeze(out),'formal_reference_read':False})
  print('TOOL_AGENT_OUTPUT_CLOSED_WITH_DISCLOSED_PURE_FALLBACK',name,flush=True)
  return
 choice=max(eligible or valid,key=lambda x:x['metrics']['F1'])
 save(out/'program_selection_freeze.json',{'chosen':choice,'source_qualified':bool(eligible),'candidate_policy':'highest source F1 within P98 and damage.5; if none research-only best executable','external_validation_not_seen':True,'formal_reference_read':False})
 program=Path(choice['program_path']).read_text();teaching=[{'dirty':{c:r['dirty'][c] for c in cols},'truth':{c:r['truth'][c] for c in cols}} for r in records]
 for zone in ['validation','repair']:
  fields,rows=read_csv(pub/(zone+'_dirty.csv'))
  result=worker(program,[{c:r[c] for c in cols} for r in rows],summary,teaching,out/(zone+'_execution'))
  final=[{**r,**x} for r,x in zip(rows,result)]
  write_csv(out/(zone+'_output.csv'),fields,final)
 save(out/'receipt.json',{'method':'real9B code generation + executable source feedback','seed':SEED,'actual_learning_calls':10,'formal_model_calls':0,
  'frozen_generated_code_applied_to_formal_rows':True,'V23_candidate_engine_access':False,'test_gold_read':False,
  'source_cv_summary_scope':'shared authorized learning summary; fold training pairs restricted; external validation untouched',
  'protocol_revision':'downstream-feature-cell-task-v1','previous_actual_program_in_feedback':True,'prior_v4_generation_calls_retained':0})
 save(out/'output_freeze.json',{'files':freeze(out),'formal_reference_read':False})
 save(ROOT/'status/tool_adapter_ready.json',{'dataset':name,'code':'run_tool_llm_v5.py','executable_outputs_closed':True,'formal_reference_read':False})
 status('tool_llm_complete',dataset=name)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();main(a.dataset)
