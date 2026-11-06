#!/usr/bin/env python3
"""Threshold scan over newly learned A skills; no old predictions or model calls.

Run the public downstream and A--F ablation entries for all three seeds first.
This adapter invokes the unchanged original five-offset worker and scorer. Zero
must match the fresh A output bytes and its independently archived score.
"""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,tarfile,time,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import supplement_runtime as runtime
R=Path(__file__).resolve().parents[1]
load=runtime.read;save=runtime.save;sha=runtime.sha

def prepare(main_runs,work,machine):
 main_runs=Path(main_runs).resolve();work=Path(work).resolve();cfg=load(machine)
 if work.exists():raise FileExistsError('Fresh threshold directory required; preserve markers')
 protocol=load(R/'experiments/paper_runs/threshold/original_config/protocol.json');cases=[];files={}
 for seed in [1013,1014,1015]:
  main=main_runs/f'ablation_seed{seed}';down=main_runs/f'downstream_seed{seed}'
  if not (main/'status/scoring_complete.json').exists():raise ValueError('Fresh main ablation must be complete: '+str(main))
  for ds in protocol['datasets']:
   sel=main/'work'/ds/'A/selection.json';selection=load(sel);skill=Path(selection['skill'])
   if not (main==skill or main in skill.parents):raise ValueError('A skill must belong to the selected fresh run')
   case={'dataset':ds,'seed':seed,'skill':str(skill),'qualified':selection['passed'],
         'input':str(down/'datasets'/ds/'public/repair_dirty.csv'),
         'declaration':str(down/'datasets'/ds/'public/declaration.json'),
         'reference':str(down/'datasets'/ds/'private/repair_reference.csv'),
         'original_output':str(main/'outputs'/ds/'A/repair_output.csv'),
         'original_score':str(main/'reports/scores.json')}
   for path in [sel,*[Path(case[k]) for k in ['input','declaration','reference','original_output','original_score']]]:
    if not path.is_file():raise FileNotFoundError(path)
    files[str(path)]=sha(path)
   for p in skill.rglob('*'):
    if p.is_file() and p.suffix!='.pyc':files[str(p)]=sha(p)
   cases.append(case)
 work.mkdir(parents=True)
 for name in ['code','audit','status','configs','outputs','logs','reports','dependencies']:(work/name).mkdir()
 import shutil
 shutil.copytree(R/'engine',work/'engine');core=runtime.core_check(work/'engine')
 spec=next(x for x in load(R/'provenance/paper_experiments.json')['studies'] if x['id']=='threshold');expected={x['path']:x['sha256'] for x in spec['execution_code_files']};adapted=[]
 with tarfile.open(R/'experiments/paper_runs/threshold/execution_code.tar.gz') as tf:
  for member in runtime.safe_members(tf):
   blob=tf.extractfile(member).read();assert hashlib.sha256(blob).hexdigest()==expected[member.name]
   path=work/member.name;path.parent.mkdir(exist_ok=True,parents=True)
   text,bindings=runtime.strings_rebind(blob.decode(),path,{spec['server_root']:str(work),'/root/fsas/home/anonymous/envs/dataforge-run/bin/python':cfg['run_python']})
   path.write_text(text);adapted.append({'path':member.name,'original_sha256':expected[member.name],'materialized_sha256':sha(path),'bindings':bindings})
 save(work/'configs/cases.json',cases);save(work/'configs/protocol.json',protocol)
 save(work/'audit/core124.json',{'files':core});save(work/'audit/input_skill_snapshot.json',{'files':files,'fresh_A_skills_only':True})
 save(work/'audit/code_freeze.json',{'files':{str(p.relative_to(work)):sha(p) for p in [*list((work/'code').glob('*.py')),*list((work/'configs').glob('*.json'))]}})
 save(work/'audit/portable_threshold_binding.json',{'main_runs':str(main_runs),'adapted_code':adapted,'machine':cfg,'cases':len(cases),'offsets':[-.1,-.05,0.,.05,.1],'model_calls':0,'old_server_artifacts_read':False})
 return {'work':str(work),'cases':len(cases),'outputs_planned':len(cases)*5,'model_calls':0}

def run(work):
 work=Path(work).resolve();cfg=load(work/'audit/portable_threshold_binding.json')['machine'];python=cfg['run_python'];cases=load(work/'configs/cases.json')
 if len(cases)!=18:raise ValueError('Original scorer requires all18 paired dataset-seed cases')
 if (work/'status/portable_execution_started.json').exists() or any((work/'outputs').iterdir()):raise RuntimeError('No automatic replay of partial threshold execution')
 for p,h in load(work/'audit/input_skill_snapshot.json')['files'].items():assert sha(p)==h
 save(work/'status/portable_execution_started.json',{'pid':os.getpid(),'at':time.time(),'model_calls':0})
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
 for i,c in enumerate(cases):
  for offset in [-.1,-.05,0.,.05,.1]:
   log=work/'logs'/f'case{i}_offset{offset}.log'
   with log.open('x') as f:r=subprocess.run([python,'-B',str(work/'code/worker.py'),'--case',str(i),'--offset',str(offset)],cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError('Preserve partial fixed outputs and inspect '+str(log))
 # Original scorer independently verifies zero-byte parity, freezes, once-only
 # reference reads, fixed seed counts and every original numerical formula.
 save(work/'status/all_outputs_closed.json',{'at':time.time(),'outputs':90,'model_calls':0})
 with (work/'logs/original_score_once.log').open('x') as f:r=subprocess.run([python,'-B',str(work/'code/score_once.py')],cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Fixed-output scoring/report failed; preserve markers, do not rerun')
 return {'scored':True,'model_calls':0,'scores_sha256':sha(work/'reports/all_scores.json')}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare');q.add_argument('--main-runs',required=True);q.add_argument('--work',required=True);q.add_argument('--machine',required=True)
 q=sub.add_parser('run');q.add_argument('--work',required=True);a=p.parse_args()
 print(json.dumps(prepare(a.main_runs,a.work,a.machine) if a.command=='prepare' else run(a.work),ensure_ascii=False,indent=2))
