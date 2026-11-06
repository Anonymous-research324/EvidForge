#!/usr/bin/env python3
"""Fresh entry points for the paper's TabClean comparison and real-error baselines.

Only orchestration, directory bindings and once-only scoring are added. The
archived final workers, candidate/repair algorithms and numerical metrics remain
unchanged. Historical controllers, PIDs and imported old outputs are not run.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,socket,subprocess,sys,time,importlib,importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parent))
import supplement_runtime as rt
R=Path(__file__).resolve().parents[1]
METHODS=['full','raha_holoclean','cdc','tabclean']
read,save,sha=rt.read,rt.save,rt.sha

def checked_sources(machine):
 sources=Path(machine.get('baseline_sources',R/'.sources')).expanduser().absolute()
 deps=Path(machine.get('tabclean_dependencies',R/'.dependencies/tabclean')).expanduser().absolute()
 pins=read(R/'provenance/baseline_sources.json')
 needed=['tabclean'] if machine.get('_study')=='tabclean_same9b' else ['tabclean','raha','holoclean','cdc']
 for name in needed:
  p=sources/name
  if not p.is_dir():raise FileNotFoundError('Install pinned baseline source first: '+str(p))
  got=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
  if got!=pins[name]['commit']:raise ValueError('Wrong pinned source revision: '+name)
  if subprocess.check_output(['git','-C',str(p),'diff','HEAD','--','*.py'],text=True):raise ValueError('Baseline source code modified: '+name)
 if not deps.is_dir():raise FileNotFoundError('Install isolated TabClean dependencies: '+str(deps))
 versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions(path=[str(deps)])}
 for pin in read(R/'environments/tabclean_target.lock.json')['pins']:
  name,version=pin.split('==');assert versions.get(name.lower().replace('_','-'))==version,'Wrong TabClean dependency '+pin
 return sources,deps

def prepare(study,work,machine_file,inputs):
 cfg=read(machine_file);sources,deps=checked_sources({**cfg,'_study':study})
 if study=='real_dirty':
  envs=Path(cfg['baseline_environments']).expanduser().absolute()
  for n in ['raha','holoclean','cdc']:
   if not (envs/n/'bin/python').is_file():raise FileNotFoundError('Baseline interpreter absent '+str(envs/n))
  if not (Path(cfg['postgres_bin'])/'pg_ctl').is_file():raise FileNotFoundError('PostgreSQL binary directory required')
 result=rt.materialize(study,work,machine_file,inputs);root=Path(work).resolve()
 refs=root/'references';refs.mkdir(exist_ok=True)
 author=refs/'TabClean_author';author.mkdir()
 for f in (sources/'tabclean').iterdir():
  if f.is_file() and (f.suffix in ['.py','.md','.txt'] or f.name.startswith('LICENSE')):shutil.copyfile(f,author/f.name)
 shutil.copytree(deps,root/'dependencies/tabclean',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 save(root/'audit/author_source.json',{'url':read(R/'provenance/baseline_sources.json')['tabclean']['url'],'commit':read(R/'provenance/baseline_sources.json')['tabclean']['commit'],'files':{str(p.relative_to(author)):sha(p) for p in author.iterdir() if p.is_file()}})
 if study=='real_dirty':
  for n in ['raha','holoclean','cdc']:(refs/n).symlink_to(sources/n,target_is_directory=True)
  (root/'services_v3').symlink_to(root/'services',target_is_directory=True)
  (root/'datasets').mkdir();(root/'outputs').mkdir()
  for name in read(root/'configs/cases.json'):
   c=root/'cases'/name;(c/'audit').mkdir(exist_ok=True);v=root/'datasets'/name;v.mkdir();(v/'public').symlink_to(c/'public',target_is_directory=True);(root/'outputs'/name).symlink_to(c/'outputs',target_is_directory=True)
 files=read(root/'audit/execution_freeze.json')['files'];files['audit/author_source.json']=sha(root/'audit/author_source.json')
 save(root/'audit/paper_baseline_binding.json',{'study':study,'source_root':str(sources),'dependency_target':str(deps),'original_workers_and_metrics':True,'historical_controller_started':False,'at':time.time()})
 if study=='tabclean_same9b':save(root/'audit/execution_freeze_simple_v1.json',{'files':files,'fresh_binding':True})
 return {'study':study,'work':str(root),'cases':len(read(root/'configs/cases.json')),'source_and_dependency_pins_checked':True,'model_calls':0,'next':'start owned service, run-case, stop service, score'}

def environment(root,name,gpu,method):
 return {**os.environ,'EF_GPU':str(gpu),'EF_SEED':str(read(root/'cases'/name/'case.json')['seed']),'EF_CASE_ID':name,'EF_CASE':name,'EF_METHOD':method,'EF_PHASE':'paper_fresh_reproduction','PYTHONPATH':str(root/'code'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLCONFIGDIR':str(root/'tmp/matplotlib')}

def run_case(root,name,method,gpu):
 root=Path(root).resolve();meta=read(root/'audit/portable_materialization.json');study=meta['study'];names=read(root/'configs/cases.json')
 if name not in names:raise ValueError('Case is outside final paper scope')
 if study=='tabclean_same9b' and method!='paired':raise ValueError('Final TabClean experiment runs the paired worker')
 if study=='real_dirty' and method not in METHODS:raise ValueError('Only the four final paper methods are allowed')
 c=root/'cases'/name;marker=c/'status'/('paper_dispatch_'+method+'.json')
 if marker.exists():raise RuntimeError('Previous attempt exists; retain partial state, no automatic replay')
 if method in ['paired','full','tabclean']:
  service=root/'services'/f'gpu{gpu}';owner=read(service/'owner.json')
  if owner['host']!=socket.gethostname() or not (service/'ready.json').exists() or (service/'stopped.json').exists():raise RuntimeError('Fresh owned model service not ready on this host')
  proc=Path('/proc')/str(owner['pid'])/'cmdline'
  if not proc.exists() or owner['name'].encode() not in proc.read_bytes():raise RuntimeError('Model owner process differs')
 save(marker,{'at':time.time(),'method':method,'case':name,'gpu':gpu,'fresh_scope_only':True})
 python=meta['machine']['run_python'];code=root/'code'
 if method=='paired':cmd=[python,'-B',str(code/'paired_simple.py'),'--case',str(c),'--gpu',str(gpu)]
 elif method=='full':cmd=[python,'-B',str(code/'worker_recovery_v3.py'),'case','--case',str(c),'--gpu',str(gpu)]
 elif method=='tabclean':cmd=[python,'-B',str(code/'tabclean_recovery_v3.py'),'--case',str(c),'--gpu',str(gpu)]
 else:cmd=[python,'-B','-c',"import os;from pathlib import Path;from cpu_recovery_v3 import one;one(os.environ['EF_CASE_ID'],os.environ['EF_METHOD'])"]
 with (c/'logs'/('paper_'+method+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=root,env=environment(root,name,gpu,method),stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Preserve failure log and real budgets; do not re-run this case: '+name+'/'+method)
 return {'case':name,'method':method,'outputs_fixed':True,'formal_reference_read':False}

def assert_models_off(root):
 for service in (root/'services').glob('gpu*'):
  if not (service/'owner.json').exists():continue
  owner=read(service/'owner.json')
  if owner['host']!=socket.gethostname() or not (service/'stopped.json').exists():raise RuntimeError('Stop every owned service on its actual host before scoring')
  p=Path('/proc')/str(owner['pid'])/'cmdline'
  if p.exists() and owner['name'].encode() in p.read_bytes():raise RuntimeError('Owned model process is still alive')
  with socket.socket() as s:
   if s.connect_ex(('127.0.0.1',owner['port']))==0:raise RuntimeError('Owned model port still open')

def reject_failures(root):
 for folder in [root/"status",*(root/"cases").glob("*/status")]:
  for p in folder.glob("*.json"):
   if "error" in p.stem or "failure" in p.stem:raise RuntimeError("Unreviewed failure blocks scoring: "+str(p))

def verify(root):
 reject_failures(root)
 meta=read(root/'audit/portable_materialization.json');names=read(root/'configs/cases.json');study=meta['study'];assert_models_off(root);rt.core_check(root/'engine')
 for f in meta['adapted_files']:
  if sha(root/f['path'])!=f['materialized_sha256']:raise ValueError('Changed worker source '+f['path'])
 author=read(root/'audit/author_source.json')
 for rel,h in author['files'].items():assert sha(root/'references/TabClean_author'/rel)==h,'Author source changed'
 methods=['full','tabclean'] if study=='tabclean_same9b' else METHODS
 for name in names:
  c=root/'cases'/name;fr=read(c/'input_freeze.json')
  for zone in ['public','private']:
   for rel,h in fr[zone].items():assert sha(c/zone/rel)==h,'Input changed '+name+'/'+zone+'/'+rel
  assert sha(c/'case.json')==fr.get('case_sha',fr.get('case_sha256'))
  for method in methods:
   out=c/'outputs'/('cdc_v3' if method=='cdc' else method)
   for rel,h in read(out/'output_freeze.json')['files'].items():assert sha(out/rel)==h,'Frozen output changed'
  for stage in ['prior','context']:
   n=c/'work/full'/(stage+'_stage/native_v3');receipt=read(n/'result.json');budget=read(n/'budget.json')
   assert receipt['closed'] and receipt['native_exit_code']==0 and receipt['synthetic_control_count']==0
   assert receipt['real9B_calls']==budget['calls']==5 and budget['pending_call'] is None
   events=[json.loads(x) for x in (n/'native_hooks.jsonl').read_text().splitlines() if x.strip()]
   assert sum(x['event']=='PreToolUse' for x in events)==sum(x['event']=='PostToolUse' for x in events)==5 and any(x['event']=='Stop' for x in events)
   tag=read(n/'session.json')['tag']
   for i in range(1,6):
    requests=list((root/'services').glob('gpu*/requests/'+tag+'call'+str(i)));assert len(requests)==1
    request=requests[0];body=read(request/'request.json');raw=read(request/'response.json');owner=read(request.parents[1]/'owner.json')
    assert raw==read(n/f'actual9b-response-{i}.json') and raw['usage']==budget['actual_usage'][i-1]
    assert body['seed']==read(c/'case.json')['seed'] and body['model']==owner['name'] and body['temperature']==(.7 if study=='tabclean_same9b' else 0)
 for request in (root/'services').glob('gpu*/requests/*'):
  if not (request/'request.json').exists():continue
  body=read(request/'request.json');response=read(request/'response.json');binding=read(request/'binding.json');case=root/'cases'/binding['case'];owner=read(request.parents[1]/'owner.json')
  assert binding['case'] in names and body['seed']==read(case/'case.json')['seed'] and body['model']==owner['name']
  assert body['temperature']==(.7 if study=='tabclean_same9b' else 0) and body['max_tokens']<=8192 and response.get('usage')
  if study=='tabclean_same9b':assert body['top_p']==.8 and body['top_k']==20 and body['presence_penalty']==1.5
 return {'passed':True,'study':study,'cases':len(names),'methods':methods,'formal_reference_rows_parsed':0,'all_owned_models_stopped':True,'at':time.time()}

def score(root):
 root=Path(root).resolve();receipt=verify(root);names=read(root/'configs/cases.json');meta=read(root/'audit/portable_materialization.json');marker=root/'audit/paper_fresh_scoring_started.json'
 if marker.exists() or any((root/'cases'/n/'audit_scoring_started.json').exists() for n in names):raise RuntimeError('Once-only scoring already entered; do not regrade')
 if meta['study']=='real_dirty' and (len(names)!=24 or {read(root/'cases'/n/'case.json')['seed'] for n in names}!={1013,1014,1015}):raise ValueError('Final real-error comparison requires all24 cases and all3 seeds')
 save(root/'audit/paper_fresh_outputs_ready.json',receipt);save(marker,{'at':time.time(),'once_only':True,'no_old_results_used':True})
 os.chdir(root);sys.path.insert(0,str(root/'code'));rows=[]
 if meta['study']=='tabclean_same9b':
  original=importlib.import_module('study_simple');outcomes={n:{'status':'closed','gpu':read(root/'cases'/n/'status/paper_dispatch_paired.json')['gpu']} for n in names};original.verify_all(names,outcomes)
  rows=[x for n in names for x in original.score_case(n,{'status':'closed'})];original.aggregate(rows)
 else:
  original=importlib.import_module('score_recovery_v3');original.METHODS=METHODS
  for n in names:
   c=root/'cases'/n;cfg=read(c/'case.json');save(c/'audit_scoring_started.json',{'at':time.time(),'methods':METHODS,'new_frozen_outputs_only':True})
   _,dirty=original.read_csv(c/'public/repair_dirty.csv');gold=original.truth_align(dirty,read(c/'private/repair_truth.json'))
   _,validation=original.read_csv(c/'public/validation_dirty.csv');vg=original.truth_align(validation,read(c/'private/validation_truth.json'));case_rows=[]
   for method in METHODS:
    out=c/'outputs'/('cdc_v3' if method=='cdc' else method);_,rr=original.read_csv(out/'repair_output.csv');_,vr=original.read_csv(out/'validation_output.csv');rr=original.align(dirty,rr);vr=original.align(validation,vr)
    assert all(a[k]==b[k] for a,b in zip(dirty,rr) for k in cfg['protected']);assert all(a[k]==b[k] for a,b in zip(validation,vr) for k in cfg['protected'])
    mr=original.metric(dirty,rr,gold,cfg['targets']);mv=original.metric(validation,vr,vg,cfg['targets']);qualified=mv['changes']>=5 and mv['clean']>=10 and mv['P']>=.98 and mv['normal_damage_rate'] is not None and mv['normal_damage_rate']<=.005
    row={'dataset':cfg['dataset'],'seed':cfg['seed'],'case':n,'method':method,'status':'scored','repair':mr,'validation':mv,'qualified':qualified,'result_origin':'fresh_frozen_output','experiment_partition':'formal','protocol_log_complete':True}
    if method=='full':row['original_evidforge_qualification']=read(c/'work/full/selection.json')['passed']
    case_rows.append(row);rows.append(row)
   save(c/'scores.json',case_rows);save(c/'status/scored.json',{'at':time.time(),'methods':4,'reference_read_once':True})
  aggregate,paired=original.aggregate_records(rows,sorted({x['dataset'] for x in rows}));save(root/'reports/all_scores.json',rows);save(root/'reports/aggregate.json',aggregate);save(root/'reports/paired_differences.json',paired)
 costs=[]
 for req in (root/'services').glob('gpu*/requests/*'):
  if (req/'response.json').exists():costs.append({'path':str(req.relative_to(root)),'binding':read(req/'binding.json'),'usage':read(req/'response.json')['usage'],'timing':read(req/'timing.json')})
 save(root/'reports/actual_model_costs.json',{'requests':costs,'actual_calls':len(costs),'old_calls_imported':False})
 save(root/'status/scoring_complete.json',{'at':time.time(),'records':len(rows),'once_only':True,'original_metrics_unchanged':True})
 return {'records':len(rows),'scores':str(root/'reports/all_scores.json'),'metric_source':'original archived final scorer; no regrading of old results'}

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 q=sub.add_parser('prepare');q.add_argument('--study',choices=['tabclean_same9b','real_dirty'],required=True);q.add_argument('--work',required=True);q.add_argument('--machine',required=True);q.add_argument('--inputs',required=True)
 q=sub.add_parser('run-case');q.add_argument('--work',required=True);q.add_argument('--case',required=True);q.add_argument('--method',choices=['paired',*METHODS],required=True);q.add_argument('--gpu',type=int,default=0)
 for name in ['verify','score']:q=sub.add_parser(name);q.add_argument('--work',required=True)
 a=p.parse_args()
 if a.action=='prepare':result=prepare(a.study,a.work,a.machine,a.inputs)
 elif a.action=='run-case':result=run_case(a.work,a.case,a.method,a.gpu)
 elif a.action=='verify':result=verify(Path(a.work).resolve())
 else:result=score(a.work)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
