#!/usr/bin/env python3
"""Materialize original supplementary workers into an isolated, configured run.

No historical controller is executed. Core bytes and scientific scalar constants
are unchanged. Only declared string path bindings and the service base port are
adapted. Fresh input exports are verified; old outputs, PID owners, budgets and
score markers are forbidden in a new run. Preparation is not a completed rerun.
"""
from __future__ import annotations
from pathlib import Path
import argparse,ast,hashlib,json,os,re,shlex,shutil,subprocess,sys,tarfile,time
from paper_scope import scopes,validate_case,runtime_source
from reproduction_data import verify_original_freeze
REPO=Path(__file__).resolve().parents[1]
WORKERS={'llm_digit70':'worker.py','new_errors':'worker.py','example_budget':'worker_budget.py',
 'robustness_retinopathy':'worker_v4.py','semantic_historical':'worker_semantic.py',
 'real_dirty':'worker_recovery_v3.py','tabclean_same9b':'paired_simple.py','advisor_supplement':'worker_v4.py','sealed_reuse':'learn_reuse.py'}
WORKERS={k:v for k,v in WORKERS.items() if k in scopes()}
FORBIDDEN_PARTS={'work','outputs','services','status','logs','reports','__pycache__'}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def core_check(root):
 files=read(REPO/'provenance/core_sha256.json')['files']
 for rel,h in files.items():
  if sha(Path(root)/rel)!=h:raise ValueError('Fixed core differs: '+rel)
 return files
def strings_rebind(text,path,mapping,port=None,startup_checks=None):
 """Transform strings only; PORT=<base>+GPU is the sole scalar exception."""
 tree=ast.parse(text,filename=str(path));changes=[]
 class Bind(ast.NodeTransformer):
  def visit_Constant(self,node):
   if not isinstance(node.value,str):return node
   old=node.value;new=old
   for a,b in sorted(mapping.items(),key=lambda x:-len(x[0])):
    if old==a or old.startswith(a+'/'):new=b+old[len(a):];break
   if new!=old:changes.append({'line':node.lineno,'before':old,'after':new});return ast.copy_location(ast.Constant(new),node)
   return node
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if startup_checks is not None and isinstance(node.func,ast.Name) and node.func.id=="range" and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==300:
    changes.append({"line":node.lineno,"field":"service startup readiness checks only","before":300,"after":startup_checks});node.args[0].value=startup_checks
   return node
  def visit_Assign(self,node):
   node=self.generic_visit(node)
   if port is not None and any(isinstance(x,ast.Name) and x.id=='PORT' for x in node.targets):
    value=node.value
    if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add) and isinstance(value.left,ast.Constant) and isinstance(value.left.value,int) and isinstance(value.right,ast.Name) and value.right.id=='GPU':
     changes.append({'line':node.lineno,'field':'PORT base','before':value.left.value,'after':port});value.left.value=port
   return node
 tree=Bind().visit(tree);ast.fix_missing_locations(tree)
 compile(tree,str(path),'exec')
 return ast.unparse(tree)+'\n',changes
def safe_members(tf):
 for m in tf:
  p=Path(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source archive member: '+m.name)
  yield m
def input_check(source):
 source=Path(source).resolve();manifest=read(source/'input_export.json')
 files=manifest.get('files')
 if not isinstance(files,dict) or not files:raise ValueError('Input export must carry nonempty original SHA-256 manifest')
 for rel,h in files.items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe input export path '+rel)
  if any(x in FORBIDDEN_PARTS for x in p.parts):raise ValueError('Historical execution artifacts in input export '+rel)
  if sha(source/rel)!=h:raise ValueError('Input export mismatch '+rel)
 if not (source/'configs/cases.json').exists():raise ValueError('Original case list required')
 verify_original_freeze(source,{'case_ids':read(source/'configs/cases.json')})
 return manifest
def materialize(study,work,machine_path,inputs):
 if study not in WORKERS:raise NotImplementedError('Study-specific fresh lifecycle is not implemented: '+study)
 root=Path(work).resolve();machine=read(machine_path);source=Path(inputs).resolve();input_manifest=input_check(source)
 if input_manifest.get('study')!=study:raise ValueError('Input export study differs')
 if root.exists():raise FileExistsError('Fresh work directory required; do not overwrite or replay '+str(root))
 source_spec=next(s for s in read(REPO/'provenance/paper_experiments.json')['studies'] if s['id']==study)
 python=str(Path(machine.get('run_python',sys.executable)).expanduser().absolute())
 infer=str(Path(machine['infer_python']).expanduser().absolute());codex=str(Path(machine['codex']).expanduser().absolute());weights=str(Path(machine['model_dir']).expanduser().absolute())
 for name,p in [('run_python',python),('infer_python',infer),('codex',codex)]:
  if not Path(p).is_file():raise FileNotFoundError(name+': '+p)
 if not Path(weights).is_dir():raise FileNotFoundError('model_dir: '+weights)
 root.parent.mkdir(parents=True,exist_ok=True)
 root.mkdir();(root/'code').mkdir()
 for name in ['audit','logs','status','tmp','services','dependencies','reports']:(root/name).mkdir()
 shutil.copytree(REPO/'engine',root/'engine');core=core_check(root/'engine')
 for rel in input_manifest['files']:
  dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,dest)
 # Case export is input-only; regenerate only empty execution directories.
 case_names=read(root/'configs/cases.json')
 if not isinstance(case_names,list) or not case_names or not all(isinstance(n,str) and '/' not in n and n not in {'.','..'} for n in case_names):raise ValueError('Cases must be safe names')
 for name in case_names:
  case=root/'cases'/name
  validate_case(study,read(case/'case.json'))
  for folder in (['work','outputs','status','logs','sealed','audit'] if study=='sealed_reuse' else ['work','outputs','status','logs']):(case/folder).mkdir(parents=True,exist_ok=True)
 mapping={source_spec['server_root']:str(root),'/root/fsas/home/anonymous/envs/dataforge-run/bin/python':python,
 '/root/fsas/home/anonymous/envs/dataforge-infer/bin/python':infer,'/root/fsas/home/anonymous/bin/codex':codex,
 '/root/fshare/models/Qwen/Qwen3.5-9B':weights,
 '/root/.cache/matplotlib':str(root/'tmp/matplotlib'),
 '/root/fsas/home/anonymous/projects/EvidForge_Advisor_Study_20261021/reports/paper_ablation_tables_20261022/fonts':str(root/'resources/fonts')}
 if study=='real_dirty':
  if 'baseline_environments' in machine:mapping['/root/fsas/home/anonymous/projects/DataForge_V23_Benchmark_seed1013/environments']=str(Path(machine['baseline_environments']).expanduser().absolute())
  if 'postgres_bin' in machine:mapping['/root/fsas/home/anonymous/projects/DataForge_V23_Benchmark_seed1013/environments/postgres/bin']=str(Path(machine['postgres_bin']).expanduser().absolute())
 if study=='semantic_historical':mapping['/root/fsas/home/anonymous/projects/EvidForge_Advisor_Study_20261021/engine']=str(root/'engine')
 for old,new in machine.get('path_bindings',{}).items():mapping[old]=str(Path(new).expanduser().absolute())
 font_names={'regular':'SongtiSC-Regular.ttf','bold':'SongtiSC-Bold.ttf','latin':'TimesNewRoman.ttf'};font_receipt=[]
 for role,source_font in machine.get('display_fonts',{}).items():
  if role not in font_names:raise ValueError('Unknown display font role '+role)
  source_font=Path(source_font).expanduser().absolute();dest=root/'resources/fonts'/font_names[role];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_font,dest);font_receipt.append({'role':role,'source':str(source_font),'sha256':sha(dest),'private_local_display_only':True})
 port=int(machine.get('service_base_port',8511))
 if not 1024<=port<=65533:raise ValueError('Invalid service base port')
 adapted=[]
 archive=REPO/'experiments/paper_runs'/study/'execution_code.tar.gz';expected={x['path']:x['sha256'] for x in source_spec['execution_code_files']}
 with tarfile.open(archive) as tf:
  for member in safe_members(tf):
   data=tf.extractfile(member).read();h=hashlib.sha256(data).hexdigest()
   if expected.get(member.name)!=h:raise ValueError('Original script SHA differs '+member.name)
   dst=root/member.name;dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.suffix=='.py':
    scoped,scope_changes=runtime_source(study,dst.name,data.decode())
    text,changes=strings_rebind(scoped,dst,mapping,port if dst.name=='model_service.py' else None,int(machine.get('startup_checks',300)) if dst.name=='model_service.py' else None)
    changes += [{'publication_scope_adapter':x} for x in scope_changes];dst.write_text(text)
   else:dst.write_bytes(data);changes=[]
   adapted.append({'path':member.name,'original_sha256':h,'materialized_sha256':sha(dst),'bindings':changes})
 # Host pin is an ownership guard, bound explicitly to the current fresh host.
 if study=='new_errors':
  import socket
  save(root/'configs/execution_host.json',{'hostname':socket.gethostname(),'purpose':'fresh service ownership only; scientific case configuration unchanged'})
 # Operational wrappers do not change runtime source, scientific settings, or gate functions.
 for name,entry in [('ef-python','runtime.py'),('ef-python-v4','runtime_v4.py'),('ef-python-serialization-v3','runtime_serialization_v3.py')]:
  if not (root/'code'/entry).exists():continue
  dst=root/'code'/name;dst.write_text('#!/bin/sh\nexec '+shlex.quote(python)+' -B '+shlex.quote(str(root/'code'/entry))+' "$@"\n');dst.chmod(0o755)
  rel='code/'+name;prior=next((x for x in adapted if x['path']==rel),None)
  adapted=[x for x in adapted if x['path']!=rel]
  adapted.append({'path':rel,'generated_operational_wrapper':True,'original_sha256':prior.get('original_sha256') if prior else None,'materialized_sha256':sha(dst),'runtime':entry})
 save(root/'audit/core124.json',{'files':core,'original_engine':str(REPO/'engine'),'source':'published immutable original124'})
 save(root/'audit/old_scores_frozen.json',{})
 # Historical-score mutation guard has no external old-study dependency in a fresh run.
 # Empty historical set does not remove input/output/core/native/once-only scoring checks.
 freeze={x['path']:x['materialized_sha256'] for x in adapted}
 for p in (root/'configs').glob('*'):
  if p.is_file():freeze[str(p.relative_to(root))]=sha(p)
 freeze['audit/core124.json']=sha(root/'audit/core124.json');freeze['audit/old_scores_frozen.json']=sha(root/'audit/old_scores_frozen.json')
 save(root/'audit/execution_freeze.json',{'files':freeze,'scope':'fresh operational path/wrapper binding; source and adapted SHA both retained'})
 save(root/'audit/portable_materialization.json',{'study':study,'machine':machine,'path_bindings':mapping,'source_root':str(source),'source_input_export_sha256':sha(source/'input_export.json'),'case_names':case_names,'adapted_files':adapted,'core_sha256':core,'historical_controller_not_executed':True,'historical_score_guards':'empty old-study set in a fresh run; scoring and reference guards remain unchanged','port_base':port,'private_display_fonts':font_receipt,'created':time.time(),'status':'MATERIALIZED_NOT_EXECUTED'})
 return doctor(root)
def doctor(root):
 root=Path(root).resolve();meta=read(root/'audit/portable_materialization.json');missing=[];unresolved=[];curator_references=[]
 core_check(root/'engine')
 for item in meta['adapted_files']:
  p=root/item['path']
  if sha(p)!=item['materialized_sha256']:raise ValueError('Materialized source changed '+item['path'])
  if p.suffix=='.py':
   text=p.read_text();tree=ast.parse(text)
   declared=[str(root),str(REPO/'engine'),*meta.get('path_bindings',{}).values()]
   declared += [str(meta['machine'][k]) for k in ['run_python','infer_python','codex','model_dir'] if k in meta['machine']]
   for node in ast.walk(tree):
    if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
    value=node.value
    if not value.startswith(('/root/','/home/')):continue
    # A rebound path may still start /root on the Linux validation host. Match
    # declared target prefixes; detect old roots, not a particular home spelling.
    if any(value==prefix or value.startswith(prefix.rstrip('/')+'/') for prefix in declared):continue
    entry={'file':item['path'],'line':node.lineno,'literal':value}
    if 'curator' in p.stem or p.name.startswith(('curate_','final_review','finalize','monitor','controller','delivery_','queue_')) or p.name in {'source_generator.py','original_curator_functions.py','corruption_functions.py'}:curator_references.append(entry)
    else:unresolved.append(entry)
 for name in meta['case_names']:
  case=root/'cases'/name
  for rel in (['case.json','input_freeze.json','source/development_bundle.json','source/declaration.json','source/context.json'] if meta['study']=='sealed_reuse' else ['case.json','input_freeze.json','public/development_bundle.json','public/declaration.json','public/context.json','public/repair_dirty.csv']):
   if not (case/rel).exists():missing.append(str(Path('cases')/name/rel))
 display_required=any(isinstance(x.get('after'),str) and 'resources/fonts' in x['after'] for item in meta['adapted_files'] for x in item.get('bindings',[]));missing_fonts=[name for name in ['SongtiSC-Regular.ttf','SongtiSC-Bold.ttf','TimesNewRoman.ttf'] if display_required and not (root/'resources/fonts'/name).is_file()]
 return {'study':meta['study'],'missing_display_fonts':missing_fonts,'status':'NEEDS_BINDINGS' if unresolved or missing else 'PREPARED_NEEDS_EXECUTION_VALIDATION','cases':len(meta['case_names']),'missing':missing,'unresolved_absolute_paths':unresolved,'archived_curator_only_references_not_executed':curator_references,'core124_verified':True,'native_calls':0,'executed':False}
def case_command(root,name,gpu):
 root=Path(root).resolve();meta=read(root/'audit/portable_materialization.json')
 if name not in meta['case_names']:raise ValueError('Unregistered case')
 worker=WORKERS[meta['study']];python=meta['machine'].get('run_python',sys.executable)
 if meta['study']=='sealed_reuse':return [python,'-B',str(root/'code'/worker),'--case',str(root/'cases'/name)]
 if meta['study']=='tabclean_same9b':return [python,'-B',str(root/'code'/worker),'--case',str(root/'cases'/name),'--gpu',str(gpu)]
 return [python,'-B',str(root/'code'/worker),'case','--case',str(root/'cases'/name),'--gpu',str(gpu)]
def run_case(root,name,gpu):
 root=Path(root).resolve();check=doctor(root)
 if check['missing'] or check['unresolved_absolute_paths']:raise RuntimeError('Unresolved dependencies; inspect doctor report')
 case=root/'cases'/name
 if any((case/'status').iterdir()) or any((case/'work').iterdir()) or any((case/'outputs').iterdir()):raise RuntimeError('Case has execution state; no automatic replay')
 # Existing model helper must be started in this same fresh root; never adopt old PIDs.
 owner=read(root/'services'/('gpu'+str(gpu))/'owner.json')
 if owner.get('host')!=__import__('socket').gethostname():raise ValueError('Wrong model owner host')
 if not (root/'services'/('gpu'+str(gpu))/'ready.json').exists():raise ValueError('Fresh owned service is not ready')
 metadata=read(case/'case.json');env={**os.environ,'EF_GPU':str(gpu),'EF_SEED':str(metadata['seed']),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root/'code'),'MPLCONFIGDIR':str(root/'tmp/matplotlib')}
 cmd=case_command(root,name,gpu);log=root/'logs'/(name+'_portable.log')
 with log.open('x') as handle:p=subprocess.Popen(cmd,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
 save(root/'status'/(name+'_portable_owner.json'),{'pid':p.pid,'command':cmd,'case':name,'started':time.time()})
 rc=p.wait()
 save(root/'status'/(name+'_portable_result.json'),{'returncode':rc,'log':str(log),'finished':time.time(),'case_replay_permitted':False})
 if rc:raise RuntimeError('Case failed; preserve logs and budget: '+str(log))
 return {'case':name,'returncode':rc,'output_closed':(case/'status/closed.json').exists(),'score_not_automatically_read':True}
def audit_case(root,name):
 """Fresh case output/native/budget proof, never parse formal truth."""
 root=Path(root).resolve();check=doctor(root);meta=read(root/'audit/portable_materialization.json')
 if check['missing'] or check['unresolved_absolute_paths']:raise ValueError('Unresolved run bindings')
 if meta['study'] not in {'new_errors','llm_digit70'}:return audit_original_worker_case(root,name)
 if name not in meta['case_names']:raise ValueError('Unregistered case')
 case=root/'cases'/name
 if any((case/'status').glob('*failure*.json')):raise ValueError('Unknown case failure blocks score')
 assert read(case/'status/closed.json')['arms']==['full','no_llm']
 inp=read(case/'input_freeze.json')
 for zone in ['public','private']:
  for rel,h in inp[zone].items():
   if sha(case/zone/rel)!=h:raise ValueError('Changed input '+zone+'/'+rel)
 assert sha(case/'case.json')==inp['case_sha256']
 for arm in ['full','no_llm']:
  for rel,h in read(case/'outputs'/arm/'output_freeze.json')['files'].items():assert sha(case/'outputs'/arm/rel)==h
  for rel,h in read(case/'work'/arm/'learning_freeze.json')['files'].items():assert sha(case/'work'/arm/rel)==h
 assert read(case/'outputs/full/candidate_evidence.json')==read(case/'outputs/no_llm/candidate_evidence.json')
 shared=read(case/'work/pre_native_shared_fit.json');assert shared['base_files']==shared['no_llm_base_files']
 for rel,h in shared['base_files'].items():assert sha(case/'work/full/onboarding/prior_prepared/base_skill'/rel)==h
 no=case/'work/no_llm';assert not list(no.rglob('compiled_candidate.json'))
 assert read(case/'outputs/no_llm/receipt.json')['new9B_calls']==0
 calls=0;services=set();case_seed=read(case/'case.json')['seed']
 for stage in ['prior','context']:
  native=case/'work/full'/(stage+'_stage')/'native_v3';result=read(native/'result.json');budget=read(native/'budget.json')
  assert result['closed'] and result['native_exit_code']==0 and result['synthetic_control_count']==0 and not result['gateway_failure']
  assert result['real9B_calls']==budget['calls']==5 and budget['pending_call'] is None
  hooks=[json.loads(line) for line in (native/'native_hooks.jsonl').read_text().splitlines() if line.strip()]
  assert sum(x['event']=='PreToolUse' for x in hooks)==sum(x['event']=='PostToolUse' for x in hooks)==5
  assert any(x['event']=='Stop' for x in hooks)
  tag=read(native/'session.json')['tag']
  for i in range(1,6):
   matches=list((root/'services').glob('gpu*/requests/'+tag+'call'+str(i)))
   assert len(matches)==1;request=matches[0];service=request.parents[1];services.add(service)
   wire=read(request/'request.json');raw=read(request/'response.json')
   assert raw==read(native/f'actual9b-response-{i}.json')
   assert raw['usage']==budget['actual_usage'][i-1]
   assert wire['seed']==case_seed and wire['temperature']==0
   assert wire['model']==read(service/'owner.json')['name'];calls+=1
 # No new request can occur after formal scoring; owned service shutdown is mandatory.
 import socket
 for service in services:
  owner=read(service/'owner.json');assert owner['host']==socket.gethostname()
  assert (service/'stopped.json').exists()
  process=Path('/proc')/str(owner['pid'])/'cmdline'
  if process.exists():assert owner['name'].encode() not in process.read_bytes()
  with socket.socket() as so:assert so.connect_ex(('127.0.0.1',int(owner['port'])))!=0
 audit={'passed':True,'case':name,'case_seed':case_seed,'real_calls':calls,'formal_reference_rows_parsed':0,'actual_original_score_module_sha256':sha(root/'code/score.py'),'after_owned_service_shutdown':True,'at':time.time()}
 target=case/'status/portable_ready_for_scoring.json'
 if target.exists():
  prior=read(target)
  assert prior['actual_original_score_module_sha256']==audit['actual_original_score_module_sha256'];return prior
 save(target,audit);return audit
def audit_original_worker_case(root,name):
 """Verify original supplemental worker contracts without parsing reference rows."""
 root=Path(root).resolve();meta=read(root/'audit/portable_materialization.json');case=root/'cases'/name
 if name not in meta['case_names']:raise ValueError('Unregistered case')
 if meta['study'] not in {'advisor_supplement','robustness_retinopathy','semantic_historical','example_budget','sealed_reuse'}:
  raise NotImplementedError('This worker needs a distinct offline/baseline audit lifecycle: '+meta['study'])
 for p in (case/'status').glob('*.json'):
  if any(x in p.name for x in ['failure','error']):raise ValueError('Unknown worker failure: '+str(p))
 sealed=meta['study']=='sealed_reuse'
 closed=read(case/('status/outputs_closed.json' if sealed else 'status/closed.json'));arms=['B1','B2','B3','combined_control'] if sealed else closed['arms'];native_arms=['full'] if sealed else [x for x in ['full','no_feedback','no_predictive_context'] if x in arms]
 inp=read(case/'input_freeze.json')
 for zone in (['source','batches','private'] if sealed else ['public','private']):
  for rel,h in inp[zone].items():
   if sha(case/zone/rel)!=h:raise ValueError('Frozen input differs '+zone+'/'+rel)
 if sha(case/'case.json')!=inp['case_sha' if sealed else 'case_sha256']:raise ValueError('Case scientific configuration differs')
 for arm in arms:
  for rel,h in read(case/'outputs'/arm/'output_freeze.json')['files'].items():
   if sha(case/'outputs'/arm/rel)!=h:raise ValueError('Output changed '+arm+'/'+rel)
 for arm in native_arms:
  for rel,h in read(case/'work'/arm/'learning_freeze.json')['files'].items():
   if sha(case/'work'/arm/rel)!=h:raise ValueError('Learning state changed '+arm+'/'+rel)
 if sealed:
  capsule=read(case/'sealed/capsule.json')
  for path,h in capsule['files'].items():
   if sha(path)!=h:raise ValueError('Sealed capsule changed '+path)
  for arm in arms:
   r=read(case/'outputs'/arm/'receipt.json')
   assert r['new_model_calls']==r['new_tokens']==r['fit_attempts']==r['network_attempts']==r['subprocess_attempts']==0
  assert read(case/'outputs/combined_control/receipt.json')['combined_parity']
  assert read(case/'sealed/teaching_parity.json')['both_output_and_decisions_exact']
 if 'reuse' in arms and (case/'outputs/full/repair_output.csv').read_bytes()!=(case/'outputs/reuse/repair_output.csv').read_bytes():raise ValueError('Reuse byte parity differs')
 if 'no_llm' in arms:
  no=case/'work/no_llm';receipt=read(no/'onboarding/prior_prepared/no_llm_receipt.json')
  assert receipt['actual_calls']==0 and not receipt['generated_program'] and not list(no.rglob('compiled_candidate.json'))
  a=case/'work/full/onboarding/prior_prepared/base_skill';b=no/'onboarding/prior_prepared/base_skill'
  files={str(p.relative_to(a)):sha(p) for p in a.rglob('*') if p.is_file()}
  assert files=={str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file()},'Unequal pre-native source checkpoints'
 calls=0;services=set();seed=read(case/'case.json')['seed']
 for arm in native_arms:
  for stage in ['prior','context']:
   n=case/'work'/arm/(stage+'_stage/native_v3');result=read(n/'result.json');budget=read(n/'budget.json')
   assert result['closed'] and result['native_exit_code']==0 and result['synthetic_control_count']==0
   assert result['real9B_calls']==budget['calls']==5 and budget['pending_call'] is None
   hooks=[json.loads(x) for x in (n/'native_hooks.jsonl').read_text().splitlines() if x.strip()]
   assert sum(x['event']=='PreToolUse' for x in hooks)==sum(x['event']=='PostToolUse' for x in hooks)==5 and any(x['event']=='Stop' for x in hooks)
   tag=read(n/'session.json')['tag']
   for i in range(1,6):
    matches=list((root/'services').glob('gpu*/requests/'+tag+'call'+str(i)));assert len(matches)==1
    request=matches[0];service=request.parents[1];services.add(service);wire=read(request/'request.json');raw=read(request/'response.json')
    assert raw==read(n/f'actual9b-response-{i}.json') and raw['usage']==budget['actual_usage'][i-1]
    assert wire['seed']==seed and wire['temperature']==0 and wire['model']==read(service/'owner.json')['name'];calls+=1
 import socket
 for service in services:
  owner=read(service/'owner.json');assert owner['host']==socket.gethostname() and (service/'stopped.json').exists()
  p=Path('/proc')/str(owner['pid'])/'cmdline'
  assert not p.exists() or owner['name'].encode() not in p.read_bytes()
  with socket.socket() as so:assert so.connect_ex(('127.0.0.1',int(owner['port'])))!=0
 audit={'passed':True,'case':name,'arms':arms,'real_calls':calls,'formal_reference_rows_parsed':0,'after_owned_service_shutdown':True,'at':time.time()}
 target=case/'status/portable_ready_for_scoring.json'
 if not target.exists():save(target,audit)
 return audit

def score_case(root,name):
 root=Path(root).resolve();audit_case(root,name)
 # Execute the original metric and once-only case_score function unchanged, after
 # independent frozen-output/native verification. This subprocess is the scorer,
 # not a learner; no original formal reference is read in this parent process.
 meta=read(root/'audit/portable_materialization.json');python=meta['machine'].get('run_python',sys.executable)
 marker=root/'cases'/name/'status/audit_scoring_started.json'
 if marker.exists() or (root/'cases'/name/'audit_scoring_started.json').exists() or (root/'cases'/name/'audit/scoring_started.json').exists() or (root/'cases'/name/'status/scored.json').exists():raise RuntimeError('Case already entered scoring; no regrading')
 entries={'new_errors':('score','case_score'),'llm_digit70':('score','case_score'),'advisor_supplement':('score','score_case'),'robustness_retinopathy':('score','score_case'),'semantic_historical':('score_semantic','score_case'),'sealed_reuse':('score_reuse','case_score')}
 if meta['study'] not in entries:raise NotImplementedError('Global-only original scorer requires whole-batch lifecycle: '+meta['study'])
 module,function=entries[meta['study']]
 code='import sys; from pathlib import Path; sys.path.insert(0,'+repr(str(root/'code'))+'); import '+module+'; '+module+'.'+function+'(Path('+repr(str(root/'cases'/name))+'))'
 log=root/'logs'/(name+'_portable_score.log')
 with log.open('x') as f:p=subprocess.run([python,'-B','-c',code],cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Fixed-output scoring failed; preserve marker: '+str(log))
 return {'case':name,'scored_once':True,'scores':str(root/'cases'/name/'scores.json'),'metric_entry':'unchanged code/'+module+'.py:'+function}
def smoke(root,name,gpu):
 """Own a fresh model service for one full real case, then audit and score once."""
 root=Path(root).resolve();status=doctor(root)
 if status['missing'] or status['unresolved_absolute_paths']:raise ValueError('Doctor blocks execution: '+json.dumps(status))
 meta=read(root/'audit/portable_materialization.json')
 if meta['study'] not in {'llm_digit70','new_errors','advisor_supplement','robustness_retinopathy','semantic_historical','sealed_reuse'}:raise NotImplementedError('This study requires global/offline lifecycle, not single-case smoke')
 if name not in meta['case_names']:raise ValueError('Unregistered case')
 if meta['study']=='sealed_reuse' and read(root/'cases'/name/'case.json')['gpu']!=gpu:raise ValueError('Original reuse case GPU lane differs; choose its registered GPU')
 python=meta['machine'].get('run_python',sys.executable);devices=meta['machine'].get('gpu_devices',{'0':'0','1':'1'})
 # Parent remains alive throughout model startup/native/repair and closes its own
 # service before references can be read. Short procfs IPC alias stays anchored.
 os.chdir(root);env={**os.environ,'EF_GPU':str(gpu),'EF_DEVICES':str(devices[str(gpu)]),'TMPDIR':f'/proc/{os.getpid()}/cwd/tmp','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 previous={k:os.environ.get(k) for k in env};os.environ.update(env)
 service=root/'services'/('gpu'+str(gpu));helper=root/'code/model_service.py'
 def invoke(action):
  if action=='stop':
   import socket
   owner=read(service/'owner.json')
   if owner['host']!=socket.gethostname():raise ValueError('Wrong host: refuse to stop another machine service')
   proc=Path('/proc')/str(owner['pid'])
   if (proc/'cmdline').exists() and (proc/'cmdline').read_bytes():
    if owner['name'].encode() not in (proc/'cmdline').read_bytes() or (proc/'cwd').resolve()!=root:raise ValueError('PID/cwd/name ownership differs; refuse stop')
  log=root/'logs'/('portable_service_'+str(gpu)+'_'+action+'.log')
  with log.open('x') as f:p=subprocess.run([python,'-B',str(helper),action],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError('Owned service '+action+' failed: '+str(log))
 try:
  invoke('start');result=run_case(root,name,gpu)
 finally:
  if (service/'owner.json').exists() and not (service/'stopped.json').exists():invoke('stop')
 if meta['study']=='sealed_reuse':
  for d in (root/'services').glob('gpu*'):
   if (d/'owner.json').exists():assert (d/'stopped.json').exists(),'All owned models must be off before reuse'
  save(root/'status/inference_disabled.json',{'host':__import__('socket').gethostname(),'at':time.time(),'all_owned_services_stopped':True})
  with (root/'logs'/(name+'_portable_offline_execute.log')).open('x') as f:r=subprocess.run([python,'-B',str(root/'code/execute_reuse.py'),'--case',str(root/'cases'/name)],cwd=root,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('Original offline reuse execution failed; preserve sealed state')
 audit=audit_case(root,name);score=score_case(root,name)
 return {'case_execution':result,'audit':audit,'score':score,'real_model_calls':audit['real_calls'],'service_shutdown_verified':True}
def run_batch(root,gpu,max_real_calls):
 """Fresh sequential batch: source/native/output -> model off -> audit -> once score.

 No historical controller/PID is executed. This is an opt-in executable command;
 preparing or verifying the repository never launches it.
 """
 import socket
 root=Path(root).resolve();meta=read(root/'audit/portable_materialization.json');study=meta['study'];check=doctor(root)
 if check['missing'] or check['unresolved_absolute_paths'] or check.get('missing_display_fonts'):raise ValueError('Doctor blocks batch; resolve declared paths/display fonts before spending model calls')
 if study not in {'new_errors','llm_digit70','advisor_supplement','robustness_retinopathy','semantic_historical','example_budget'}:raise NotImplementedError('Distinct baseline/offline batch protocol required: '+study)
 names=meta['case_names'];planned=0
 for name in names:
  c=root/'cases'/name
  if any((c/'work').iterdir()) or any((c/'outputs').iterdir()) or any((c/'status').iterdir()):raise ValueError('Partial/existing case cannot be replayed: '+name)
  cfg=read(c/'case.json');planned+=10
 if planned>max_real_calls:raise ValueError(f'Planned {planned} calls exceed explicit limit {max_real_calls}')
 if study=='example_budget' and len(names)!=54:raise ValueError('Original global example-budget scorer requires all54 planned cases')
 save(root/'status/portable_batch_started.json',{'host':socket.gethostname(),'pid':os.getpid(),'study':study,'cases':names,'maximum_real_calls':planned,'budget_limit':max_real_calls,'at':time.time()})
 os.chdir(root);machine=meta['machine'];python=machine.get('run_python',sys.executable);devices=machine.get('gpu_devices',{'0':'0','1':'1'})
 env={**os.environ,'EF_GPU':str(gpu),'EF_DEVICES':str(devices[str(gpu)]),'TMPDIR':f'/proc/{os.getpid()}/cwd/tmp','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'};os.environ.update(env)
 service=root/'services'/('gpu'+str(gpu))
 def owned(action):
  if action=='stop':
   owner=read(service/'owner.json');proc=Path('/proc')/str(owner['pid'])
   if owner['host']!=socket.gethostname():raise ValueError('Wrong stop host')
   if (proc/'cmdline').exists() and (proc/'cmdline').read_bytes():
    assert owner['name'].encode() in (proc/'cmdline').read_bytes() and (proc/'cwd').resolve()==root,'Wrong process ownership'
  with (root/'logs'/('portable_batch_service_'+action+'.log')).open('x') as f:r=subprocess.run([python,'-B',str(root/'code/model_service.py'),action],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('Owned model service '+action+' failed')
 try:
  owned('start')
  for name in names:run_case(root,name,gpu)
 finally:
  if (service/'owner.json').exists() and not (service/'stopped.json').exists():owned('stop')
 audits=[audit_case(root,name) for name in names]
 assert sum(a['real_calls'] for a in audits)==planned
 receipt={'passed':True,'cases':audits,'formal_reference_rows_parsed':0,'all_owned_models_stopped':True,'at':time.time()}
 for rel in ['audit/all_outputs_ready_integrity.json','audit/all_outputs_ready.json']:save(root/rel,receipt)
 for rel in ['status/all_model_outputs_closed.json','status/all_outputs_closed.json']:save(root/rel,{'cases':len(names),'at':time.time(),'formal_reference_read':False})
 entry={'new_errors':'score.py','llm_digit70':'score.py','advisor_supplement':'score_v2.py','robustness_retinopathy':'score.py','semantic_historical':'score_semantic.py','example_budget':'score_once.py'}[study]
 with (root/'logs/portable_global_score.log').open('x') as f:r=subprocess.run([python,'-B',str(root/'code'/entry)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Original fixed-output scoring/report failed; preserve markers and inspect log; do not repeat batch')
 save(root/'status/portable_batch_complete.json',{'at':time.time(),'cases':len(names),'real_model_calls':planned,'scorer':entry,'result_sha256':sha(root/'reports/all_scores.json'),'case_replay_permitted':False})
 return read(root/'status/portable_batch_complete.json')

def run_reuse_batch(root,max_real_calls):
 """Learn all six skills, stop owned models, execute every batch, then score once."""
 import owned_model_service as owned
 root=Path(root).resolve();check=doctor(root);meta=read(root/'audit/portable_materialization.json')
 if meta['study']!='sealed_reuse':raise ValueError('Dedicated sealed-reuse batch required')
 if check['missing'] or check['unresolved_absolute_paths'] or check.get('missing_display_fonts'):raise ValueError('Doctor blocks reuse batch')
 names=meta['case_names'];cases={n:read(root/'cases'/n/'case.json') for n in names}
 if len(names)!=6 or {(c['dataset'],c['seed']) for c in cases.values()}!={(d,s) for d in ['retinopathy','airfoil'] for s in [1013,1014,1015]}:raise ValueError('All six prescribed reuse cases required')
 planned=10*len(names)
 if max_real_calls<planned:raise ValueError('Reuse batch exceeds explicit call limit')
 if (root/'status/portable_reuse_batch_started.json').exists() or (root/'status/inference_disabled.json').exists():raise ValueError('Existing reuse batch cannot be replayed')
 for name in names:
  case=root/'cases'/name
  if any(any((case/folder).iterdir()) for folder in ['work','outputs','status','sealed']):raise ValueError('Partial/existing reuse case cannot be replayed: '+name)
 save(root/'status/portable_reuse_batch_started.json',{'pid':os.getpid(),'at':time.time(),'cases':names,'maximum_real_calls':planned,'budget_limit':max_real_calls})
 os.chdir(root);machine=meta['machine'];python=machine.get('run_python',sys.executable)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=f'/proc/{os.getpid()}/cwd/tmp')
 for gpu in sorted({c['gpu'] for c in cases.values()}):
  try:
   owned.start(root,gpu)
   for name in names:
    if cases[name]['gpu']==gpu:run_case(root,name,gpu)
  finally:
   service=root/'services'/('gpu'+str(gpu))
   if (service/'owner.json').exists() and not (service/'stopped.json').exists():owned.stop(root,gpu)
 for name in names:
  case=root/'cases'/name
  if not (case/'status/learning_closed.json').is_file() or not (case/'sealed/capsule.json').is_file():raise ValueError('All skills must close before any query batch')
 for service in (root/'services').glob('gpu*'):
  if (service/'owner.json').exists() and not (service/'stopped.json').exists():raise ValueError('All owned model services must be stopped')
 save(root/'status/inference_disabled.json',{'at':time.time(),'all_six_skills_closed':True,'all_owned_services_stopped':True,'formal_reference_read':False})
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 def invoke(entry,args,log):
  with (root/'logs'/log).open('x') as f:r=subprocess.run([python,'-B',str(root/'code'/entry),*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('Reuse stage failed; retain all partial evidence: '+log)
 for name in names:invoke('execute_reuse.py',['--case',str(root/'cases'/name)],name+'_offline_batch.log')
 audits=[audit_case(root,name) for name in names]
 if sum(a['real_calls'] for a in audits)!=planned:raise ValueError('Unexpected reuse learning call count')
 requests=[]
 for name in names:
  for stage in ['prior','context']:
   tag=read(root/'cases'/name/'work/full'/(stage+'_stage/native_v3/session.json'))['tag']
   for i in range(1,6):
    matches=list((root/'services').glob('gpu*/requests/'+tag+'call'+str(i)))
    if len(matches)!=1:raise ValueError('Unbound model request')
    p=matches[0];response=read(p/'response.json')
    requests.append({'case':name,'gpu':cases[name]['gpu'],'tag':p.name,'request_sha':sha(p/'request.json'),'response_sha':sha(p/'response.json'),'usage':response['usage'],'seconds':read(p/'timing.json')['seconds']})
 if len(requests)!=planned or len([p for p in (root/'services').glob('gpu*/requests/*') if p.is_dir()])!=planned:raise ValueError('Unexpected extra model request')
 save(root/'audit/outputs_ready.json',{'passed':True,'at':time.time(),'cases':audits,'requests':requests,'formal_rows_read':0,'all_owned_services_stopped':True})
 invoke('score_reuse.py',[],'reuse_score_once.log')
 invoke('report_reuse.py',[],'reuse_fixed_result_report.log')
 result={'cases':len(names),'batches':18,'real_model_calls':planned,'all_learning_preceded_query_execution':True,'all_owned_models_stopped':True,'scores_sha256':sha(root/'reports/all_scores.json'),'at':time.time(),'case_replay_permitted':False}
 save(root/'status/portable_reuse_batch_complete.json',result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('prepare');q.add_argument('--study',choices=sorted(WORKERS),required=True);q.add_argument('--work',required=True);q.add_argument('--machine',required=True);q.add_argument('--inputs',required=True)
 q=s.add_parser('doctor');q.add_argument('--work',required=True)
 q=s.add_parser('run-case');q.add_argument('--work',required=True);q.add_argument('--case',required=True);q.add_argument('--gpu',type=int,choices=[0,1],required=True)
 for action in ['audit-case','score-case']:
  q=s.add_parser(action);q.add_argument('--work',required=True);q.add_argument('--case',required=True)
 q=s.add_parser('run-batch');q.add_argument('--work',required=True);q.add_argument('--gpu',type=int,choices=[0,1],required=True);q.add_argument('--max-real-calls',type=int,required=True)
 q=s.add_parser('run-reuse-batch');q.add_argument('--work',required=True);q.add_argument('--max-real-calls',type=int,required=True)
 q=s.add_parser('smoke');q.add_argument('--work',required=True);q.add_argument('--case',required=True);q.add_argument('--gpu',type=int,choices=[0,1],required=True)
 a=p.parse_args()
 if a.command=='prepare':result=materialize(a.study,a.work,a.machine,a.inputs)
 elif a.command=='doctor':result=doctor(a.work)
 elif a.command=='run-case':result=run_case(a.work,a.case,a.gpu)
 elif a.command=='audit-case':result=audit_case(a.work,a.case)
 elif a.command=='smoke':result=smoke(a.work,a.case,a.gpu)
 elif a.command=='run-batch':result=run_batch(a.work,a.gpu,a.max_real_calls)
 elif a.command=='run-reuse-batch':result=run_reuse_batch(a.work,a.max_real_calls)
 else:result=score_case(a.work,a.case)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
