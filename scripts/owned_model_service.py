#!/usr/bin/env python3
"""Fresh owned inference lifecycle only; original study transport controls sampling.

This helper never constructs learning prompts or changes an original model request.
It is used when the historical study inherited somebody else's live service.
"""
from pathlib import Path
import argparse,json,os,subprocess,socket,signal,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def selected(machine,gpu):
 devices=machine.get('gpu_devices',{'0':'0','1':'1'})[str(gpu)]
 ids=[int(x) for x in devices.split(',')]
 if len(ids)!=1:raise ValueError('Paper same9B replacement uses one TP1 GPU per lane')
 return ids,devices

def inventory(ids):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 GPUs={int(a):{'uuid':b.strip(),'memory':int(c)} for a,b,c in (line.split(',') for line in rows.splitlines())}
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 active={line.split(',')[0].strip() for line in compute.splitlines() if ',' in line}
 for i in ids:
  if i not in GPUs or GPUs[i]['memory']>=512 or GPUs[i]['uuid'] in active:raise RuntimeError('Configured GPU occupied; never stop someone else')
 return {'gpu':GPUs,'compute':compute}

def start(root,gpu):
 meta=read(root/'audit/portable_materialization.json');m=meta['machine'];service=root/'services'/f'gpu{gpu}';service.mkdir(parents=True,exist_ok=True)
 if (service/'owner.json').exists():raise RuntimeError('Existing owner; do not duplicate startup')
 ids,devices=selected(m,gpu);before=inventory(ids);port=int(m.get('service_base_port',8511))+gpu
 with socket.socket() as so:so.bind(('127.0.0.1',port))
 name=f'evidforge-{meta["study"]}-repro-gpu{gpu}-9b';model=Path(m['model_dir'])
 with (service/'weights_check.txt').open('x') as f:r=subprocess.run(['sha256sum','-c',str(root/'configs/model-files-9B.sha256')],cwd=model,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Pinned model files differ')
 inventory(ids)
 cmd=[m['infer_python'],'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--served-model-name',name,'--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--dtype','bfloat16','--language-model-only','--max-model-len','131072','--max-num-seqs','1','--gpu-memory-utilization','.90','--reasoning-parser','qwen3','--seed','1013','--default-chat-template-kwargs','{"enable_thinking":false}']
 env={**os.environ,'CUDA_VISIBLE_DEVICES':devices,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_USE_FLASHINFER_SAMPLER':'0','PYTHONDONTWRITEBYTECODE':'1'}
 # The server's own PID anchors short IPC paths throughout all its children.
 program="import os,sys;os.environ['TMPDIR']=f'/proc/{os.getpid()}/cwd/tmp';os.execv(sys.argv[1],sys.argv[1:])"
 launch=[m['infer_python'],'-c',program,*cmd]
 with (service/'server.log').open('x') as f:p=subprocess.Popen(launch,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 write(service/'owner.json',{'host':socket.gethostname(),'pid':p.pid,'pgid':p.pid,'gpu':gpu,'physical_devices':devices,'port':port,'name':name,'command':cmd,'cwd':str(root),'initial_inventory':before,'sampling':'original study model_service.py:call unchanged'})
 for _ in range(int(m.get('startup_checks',900))):
  if p.poll() is not None:raise RuntimeError('Owned inference server exited; preserve log')
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as f:models=json.load(f)
   if not any(x['id']==name for x in models['data']):raise ValueError('Wrong model endpoint')
   write(service/'ready.json',{'models':models,'at':time.time()});return
  except OSError:time.sleep(2)
 raise TimeoutError('Cold startup timed out; inspect this owner, do not duplicate')

def stop(root,gpu):
 service=root/'services'/f'gpu{gpu}';owner=read(service/'owner.json')
 if owner['host']!=socket.gethostname():raise RuntimeError('Wrong host; cannot stop a remote owner')
 p=Path('/proc')/str(owner['pid'])
 if (p/'cmdline').exists() and (p/'cmdline').read_bytes():
  text=(p/'cmdline').read_bytes()
  if owner['name'].encode() not in text or (p/'cwd').resolve()!=root or os.getpgid(owner['pid'])!=owner['pgid']:raise RuntimeError('Owner PID/cwd/group mismatch')
  os.killpg(owner['pid'],signal.SIGTERM)
 for _ in range(90):
  with socket.socket() as so:active=so.connect_ex(('127.0.0.1',owner['port']))==0
  group=[]
  for proc in Path('/proc').glob('[0-9]*'):
   try:
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if int(fields[2])==owner['pgid'] and fields[0]!='Z':group.append(proc.name)
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  if not active and not group:
   write(service/'stopped.json',{'at':time.time(),'pid':owner['pid'],'port_closed':True,'owned_group_exited':True});return
  time.sleep(1)
 raise RuntimeError('Owned process group has not exited; block score and retain owner')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','stop']);p.add_argument('--work',type=Path,default=ROOT);p.add_argument('--gpu',type=int,default=int(os.environ.get('EF_GPU','0')));a=p.parse_args();root=a.work.resolve();{'start':start,'stop':stop}[a.action](root,a.gpu)
