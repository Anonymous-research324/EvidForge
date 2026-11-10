"""Fresh pure9B versus V23 outputs. All calls seed1014; no test reference read."""
from pathlib import Path
import sys,argparse,subprocess,json,time,os,math
from common import ROOT,SEED,save,load,read_csv,write_csv,freeze,digest,status
import seed_runtime
seed_runtime.install()
from unseen_table.common import equal,number
from skill_runtime.summary import expanded_summary
from workflow.packages import execute
from skill_runtime import client
PURE="""You repair a dirty table using the provided authorized teaching summary. No formal original values, true error positions, or classification labels are supplied. Process every input row. Identify supported errors and return justified sparse replacements. Use learned affix/scale/digit evidence and current other fields. Preserve normal values; do not invent an erased continuous value. You have no execution or V23 candidate tool. Return only the requested JSON, every row key exactly once, an empty object for rows without justified edits."""
SYSTEM="""Use the registered apply_verified_skill action to execute the frozen V23 checkpoint on the supplied dirty table. Its source identity, fields and role permissions have been checked. Independent validation qualification is reported separately; this is an authorized research run, never permission to deploy a failed skill. Do not change thresholds, invent additional answers, or access references. Return the requested JSON action."""
def main(name):
 selected=load(ROOT/'outputs/v23'/name/'selection.json');skill=Path(selected['skill'])
 config=load(skill/'configuration.json');base=Path(config['base_skill_path'])
 summary=expanded_summary(base)
 prior=load(Path(config['prior_skill_path'])/'configuration.json') if config['format_version']==19 else config
 summary.update(source_reviewed_digit_gate=prior['base_review_policy'],source_exact_affix_enabled=prior['exact_affix_enabled'])
 if (skill/'compiled_candidate.json').exists():
  c=load(skill/'compiled_candidate.json');summary['source_generated_scoring_configuration']={'program':c['program'],'source_checks':c['metrics']}
 pub=ROOT/'datasets'/name/'public';fields,rows=read_csv(pub/'repair_dirty.csv')
 cols=load(pub/'declaration.json')['targets']
 out=ROOT/'outputs'/name/'primary_llm'
 if out.exists():raise FileExistsError('Preserve previous run')
 out.mkdir(parents=True);save(out/'shared_summary.json',summary)
 t=time.monotonic();programme,ds=execute(rows,skill)
 write_csv(out/'program_only.csv',fields,programme);save(out/'decisions.json',ds)
 programme_seconds=time.monotonic()-t
 # Actual dispatch is a single model decision; program is freshly computed.
 body={'batch':0,'learned_summary':summary,
       'previous_answer':{'registered_permission':{'source_and_schema_checked':True,'formal_research_authorized':True,'qualification_passed':selected['passed'],'row_count':len(rows),'scope':'whole supplied table'}}}
 schema={'type':'object','properties':{'action':{'type':'string','enum':['apply_verified_skill']}},'required':['action'],'additionalProperties':False}
 answer=client.call(SYSTEM,body,schema,out/'system_dispatch',name.replace('_','').title()+'V23Dispatch1014R3',cols)
 if answer!={'action':'apply_verified_skill'}:raise ValueError('No validated system dispatch')
 write_csv(out/'v23_output.csv',fields,programme)
 output=[dict(r) for r in rows];history=[]
 maximum=max(1,120//len(cols))
 # Duplicate groups remain in ONE data partition; request chunks may split a
 # large group to respect the same 120-cell context/output budget for all LLMs.
 layout=[list(range(i,min(i+maximum,len(rows)))) for i in range(0,len(rows),maximum)]
 save(out/'batch_layout.json',layout)
 for batch,ids in enumerate(layout):
  current={f'row_{i}':{c:rows[j][c] for c in cols} for i,j in enumerate(ids)}
  val={'type':'string','maxLength':96}
  row_schema={'type':'object','properties':{c:val for c in cols},'additionalProperties':False}
  js={'type':'object','properties':{'repairs':{'type':'object','properties':{k:row_schema for k in current},'required':list(current),'additionalProperties':False}},'required':['repairs'],'additionalProperties':False}
  payload={'rows':current,'learned_summary':summary};success=False;t=time.monotonic()
  for attempt in range(3):
   try:
    a=client.call(PURE,payload,js,out/'trace'/f'pure_{batch:04d}_{attempt}',name.replace('_','').title()+f'Pure1013R3B{batch}R{attempt}',cols)
    patches=a['repairs']
    if set(a)!={'repairs'} or set(patches)!=set(current):raise ValueError('Incomplete row keys')
    changes=[]
    for key,p in patches.items():
     if not set(p)<=set(cols):raise ValueError('Protected field change')
     for c,v in p.items():
      if isinstance(v,str) and equal(v,rows[ids[int(key.split('_')[1])]][c]):continue
      if not isinstance(v,str) or len(v)>96 or v and number(v) is None:raise ValueError('Invalid finite numerical repair')
      changes.append((ids[int(key.split('_')[1])],c,v))
    for i,c,v in changes:output[i][c]=v
    success=True;break
   except (ValueError,KeyError,RuntimeError,OSError,TypeError) as e:
    save(out/'trace'/f'rejected_{batch}_{attempt}.json',{'error':repr(e),'no_partial_batch_write':True})
    payload['format_error']=str(e)
  history.append({'batch':batch,'accepted':success,'seconds':time.monotonic()-t,'calls':attempt+1})
  save(out/'progress.json',history);write_csv(out/'pure_partial.csv',fields,output)
  status('pure_llm_repair',dataset=name,batch=batch+1,batches=len(layout),accepted=success)
  print('PURE1014',name,batch+1,'/',len(layout),success,flush=True)
  if not success:raise RuntimeError('Incomplete real LLM output; do not score as zero')
 write_csv(out/'pure_output.csv',fields,output)
 save(out/'receipt.json',{'seed':SEED,'dataset':name,'calls':sum(h['calls'] for h in history)+1,'program_seconds':programme_seconds,
  'real_v23_dispatch_calls':1,'formal_reference_read':False,'shared_teaching_summary_identical':True,
  'system_output_equals_fresh_program':True,'automatic_deployment_qualified':selected['passed'],
  'system_native_learning_separately_recorded':True})
 save(out/'output_freeze.json',{'files':freeze(out),'all_primary_outputs_closed':True,'test_gold_read':False})
 status('primary_llm_complete',dataset=name)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();main(a.dataset)
