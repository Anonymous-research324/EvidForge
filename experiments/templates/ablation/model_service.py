"""Owned single-GPU BF16 9B; all operational seeds are 1014."""
import os,sys,json,subprocess,time,socket,signal,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPLICA=os.environ.get('EVIDFORGE_MODEL_REPLICA','primary')
if REPLICA not in ['primary','secondary']:raise ValueError('Unknown owned model replica')
SERVICE=ROOT/('models_service_gpu1' if REPLICA=='secondary' else 'models_service');SEED=1014;PORT=8091 if REPLICA=='secondary' else 8081
WEIGHTS='@MODEL_DIR@'
NAME='evidforge-ablation-seed1014'
PYTHON='@INFER_PYTHON@'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def start():
    owner=SERVICE/'owner.json'
    if owner.exists():raise RuntimeError('Existing service receipt: inspect before retry')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    free=[int(r.split(',')[0]) for r in used.splitlines() if int(r.split(',')[1])<512]
    wanted=0
    if wanted not in free:raise RuntimeError('Requested GPU occupied; do not stop another service')
    free=[wanted]
    with socket.socket() as s:s.bind(('127.0.0.1',PORT))
    manifest=Path('@MODEL_MANIFEST@')
    check=subprocess.run(['sha256sum','-c',str(manifest)],cwd=WEIGHTS,capture_output=True,text=True)
    (SERVICE/'weights_check.txt').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('9B weight fingerprint failed')
    cmd=[PYTHON,'-m','vllm.entrypoints.openai.api_server','--model',WEIGHTS,'--served-model-name',NAME,
        '--host','127.0.0.1','--port',str(PORT),'--tensor-parallel-size','1','--dtype','bfloat16','--language-model-only',
        '--max-model-len','131072','--max-num-seqs','1','--gpu-memory-utilization','.90','--reasoning-parser','qwen3',
        '--seed',str(SEED),'--default-chat-template-kwargs','{"enable_thinking":false}']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(free[0]),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'OMP_NUM_THREADS':'4','VLLM_USE_FLASHINFER_SAMPLER':'0','PYTHONDONTWRITEBYTECODE':'1'}
    with (SERVICE/'server.log').open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(owner,{'pid':p.pid,'pgid':p.pid,'gpu':free[0],'port':PORT,'name':NAME,'seed':SEED,'command':cmd,'created_at':time.time()})
    for _ in range(240):
        if p.poll() is not None:raise RuntimeError('Owned server exited; retain server.log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=2) as r:models=json.load(r)
            save(SERVICE/'ready.json',{'models':models,'seed':SEED,'ready_at':time.time()});print('9B_READY seed1014',flush=True);return
        except OSError:time.sleep(2)
    raise TimeoutError('9B startup timeout; preserve process receipt')
def call():
    body=json.load(sys.stdin);tag=body.pop('_trace_tag');old=body.get('seed');body.update(model=NAME,seed=SEED,temperature=0)
    if not tag.isalnum():raise ValueError('Alphanumeric trace tag required')
    out=SERVICE/'requests'/tag;out.mkdir(parents=True,exist_ok=False)
    save(out/'request.json',body);save(out/'operational_seed.json',{'input_seed':old,'effective_seed':SEED,'explicit_user_override':True})
    t=time.monotonic()
    try:
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as r:reply=json.load(r)
        save(out/'response.json',reply);print(json.dumps(reply,ensure_ascii=False),flush=True)
    except Exception as e:save(out/'failure.json',{'error':repr(e)});raise
    finally:save(out/'timing.json',{'seconds':time.monotonic()-t})
def stop():
    owner=json.loads((SERVICE/'owner.json').read_text());p=Path('/proc')/str(owner['pid'])/'cmdline'
    if p.exists():
        text=p.read_bytes().replace(b'\0',b' ').decode()
        if NAME not in text or '--port '+str(PORT) not in text or os.getpgid(owner['pid'])!=owner['pgid']:raise RuntimeError('Ownership mismatch')
        os.killpg(owner['pid'],signal.SIGTERM)
    save(SERVICE/'stopped.json',{'pid':owner['pid'],'at':time.time()})
if __name__=='__main__':{'start':start,'call':call,'stop':stop}[sys.argv[1]]()
