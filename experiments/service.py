"""Owned local vLLM transport. It never reads dataset labels or references."""
from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]
CASE=json.loads((ROOT/"case.json").read_text());CFG=CASE["machine"];SEED=CASE["seed"]
SERVICE=ROOT/"models_service";NAME=f"evidforge-{CASE['study']}-seed{SEED}";PORT=CFG["port"]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def start():
 if (SERVICE/"owner.json").exists():raise RuntimeError("Owner receipt exists; inspect the previous service before restart")
 used=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
 gpu=CFG["gpu"];rows={int(s.split(",")[0]):int(s.split(",")[1]) for s in used.splitlines()}
 if rows.get(gpu,100000)>=512:raise RuntimeError("Configured GPU is occupied; do not stop another service")
 with socket.socket() as s:s.bind(("127.0.0.1",PORT))
 model=Path(CFG["model_dir"]);manifest=Path(CFG["model_manifest"])
 with (SERVICE/"weights_check.txt").open("x") as f:r=subprocess.run(["sha256sum","-c",str(manifest)],cwd=model,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError("Model fingerprint mismatch")
 cmd=[CFG["infer_python"],"-m","vllm.entrypoints.openai.api_server","--model",str(model),"--served-model-name",NAME,"--host","127.0.0.1","--port",str(PORT),"--tensor-parallel-size","1","--dtype","bfloat16","--language-model-only","--max-model-len","131072","--max-num-seqs","1","--gpu-memory-utilization",".90","--reasoning-parser","qwen3","--seed",str(SEED),"--default-chat-template-kwargs",'{"enable_thinking":false}']
 env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"4","VLLM_USE_FLASHINFER_SAMPLER":"0","PYTHONDONTWRITEBYTECODE":"1"}
 with (SERVICE/"server.log").open("x") as f:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(SERVICE/"owner.json",{"pid":proc.pid,"pgid":proc.pid,"gpu":gpu,"port":PORT,"name":NAME,"seed":SEED,"command":cmd})
 for _ in range(300):
  if proc.poll() is not None:raise RuntimeError("Owned server exited; retain server.log")
  try:
   with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=2) as r:models=json.load(r)
   save(SERVICE/"ready.json",{"models":models,"ready_at":time.time()});return
  except OSError:time.sleep(2)
 raise TimeoutError("Inspect owned process and logs; do not duplicate start")
def call():
 body=json.load(sys.stdin);tag=body.pop("_trace_tag")
 if not tag.isalnum():raise ValueError("Alphanumeric trace tag required")
 body.update(model=NAME,seed=SEED,temperature=0);out=SERVICE/"requests"/tag;out.mkdir(parents=True,exist_ok=False)
 save(out/"request.json",body);start=time.monotonic()
 try:
  request=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
  with urllib.request.urlopen(request,timeout=300) as r:reply=json.load(r)
  save(out/"response.json",reply);print(json.dumps(reply,ensure_ascii=False),flush=True)
 except Exception as e:save(out/"failure.json",{"error":repr(e)});raise
 finally:save(out/"timing.json",{"seconds":time.monotonic()-start})
def stop():
 if not (SERVICE/"owner.json").exists():return
 owner=json.loads((SERVICE/"owner.json").read_text());p=Path("/proc")/str(owner["pid"])/"cmdline"
 if p.exists():
  text=p.read_bytes().replace(b"\0",b" ").decode()
  if NAME not in text or "--port "+str(PORT) not in text or os.getpgid(owner["pid"])!=owner["pgid"]:raise RuntimeError("Ownership mismatch: refusing to stop")
  os.killpg(owner["pid"],signal.SIGTERM)
  for _ in range(60):
   with socket.socket() as s:occupied=s.connect_ex(("127.0.0.1",PORT))==0
   if not occupied:break
   time.sleep(1)
 save(SERVICE/"stopped.json",{"pid":owner["pid"],"time":time.time()})
if __name__=="__main__":{"start":start,"call":call,"stop":stop}[sys.argv[1]]()
