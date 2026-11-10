"""Relocatable orchestration; numerical methods live in the hash-locked engine."""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,os,shlex,shutil,socket,subprocess,sys,time,urllib.request,zipfile
REPO=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((REPO/"experiments/protocol.json").read_text())
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_engine():
 files=load(REPO/"provenance/core_sha256.json")["files"]
 for r,h in files.items():
  if sha(REPO/"engine"/r)!=h:raise ValueError("Engine integrity mismatch: "+r)
 return len(files)
def defaults():
 return {"run_python":sys.executable,"infer_python":str(REPO/".venvs/inference/bin/python"),"codex":str(REPO/".tools/node_modules/.bin/codex"),"model_dir":str(REPO/"models/Qwen3.5-9B"),"environments":str(REPO/".venvs"),"sources":str(REPO/".sources"),"postgres_bin":"/usr/lib/postgresql/15/bin","model_manifest":str(REPO/"provenance/model-files-9B.sha256"),"gpu":0,"port":8091,"pg_port":55441}
def settings(path):
 x=defaults()
 if path:x.update(load(path))
 for k in ["run_python","infer_python","codex","model_dir","environments","sources","postgres_bin"]:
  x[k]=str(Path(x[k]).expanduser().absolute())
 return x
def case_path(work,study,seed):return Path(work).resolve()/f"{study}_seed{seed}"
def initialize(work,study,seed,cfg,names=None):
 check_engine()
 allowed=PROTOCOL['datasets'][study if study in PROTOCOL['datasets'] else 'downstream']
 if names and (len(names)!=len(set(names)) or not set(names)<=set(allowed)):raise ValueError('Datasets outside the declared paper scope')
 root=case_path(work,study,seed);meta=root/"case.json"
 identity={"study":study,"seed":seed,"datasets":names or PROTOCOL["datasets"][study if study in PROTOCOL["datasets"] else "downstream"],"machine":cfg,"release_core":sha(REPO/"provenance/core_sha256.json")}
 if meta.exists():
  if load(meta)!=identity:raise ValueError("Existing case has different configuration; use a new work directory")
  binding=load(root/"audit/release_binding.json")
  for group in ["core","code"]:
   for rel,h in binding[group].items():
    if sha(root/rel)!=h:raise ValueError("Existing case frozen source changed: "+rel)
  return root
 if root.exists():raise FileExistsError("Partially initialized directory: inspect, do not overwrite")
 root.mkdir(parents=True)
 for n in ["code","configs","logs","audit","status","outputs","models_service","dependencies","frozen_workspace"]:
  (root/n).mkdir()
 shutil.copytree(REPO/"engine",root/"frozen_workspace/DataForge_UnseenTable_v23")
 core_files={str(f.relative_to(root)):sha(f) for f in (root/"frozen_workspace").rglob("*.py")}
 save(root/"audit/frozen_v23_files.json",{"files":core_files})
 save(root/"audit/core124.json",{"files":{str(Path(r).relative_to("frozen_workspace/DataForge_UnseenTable_v23")):h for r,h in core_files.items()}})
 replacements={"@RUN_PYTHON@":cfg["run_python"],"@CODEX@":cfg["codex"],"@INFER_PYTHON@":cfg["infer_python"],"@MODEL_DIR@":cfg["model_dir"],"@MODEL_MANIFEST@":str(REPO/"provenance/model-files-9B.sha256"),"@DOWNSTREAM_ROOT@":str(case_path(work,"downstream",seed)),"@CURATOR_ARCHIVE@":str(root/"generator_inputs/private"),"@RUN_ENV@":str(Path(cfg["run_python"]).parent.parent),"@PYTHON_BASE@":subprocess.check_output([cfg["run_python"],"-c","import sys;print(sys.base_prefix)"],text=True).strip()}
 template=REPO/"experiments/templates"/study
 for f in template.glob("*.py"):
  s=f.read_text().replace("1014",str(seed))
  for oldport in ["55421","55431"]:s=s.replace(oldport,str(cfg["pg_port"]))
  for a,b in replacements.items():s=s.replace(a,b)
  if "@" in s and any(a in s for a in replacements):raise ValueError("Unbound runtime path")
  if f.name=="model_service.py":s=(REPO/"experiments/service.py").read_text()
  # File-access allowlist follows this interpreter's actual standard library path.
  if f.name=="tool_worker.py":
   s=s.replace("'/usr/lib/','/usr/share/zoneinfo/'",repr(str(Path(replacements["@PYTHON_BASE@"])))+",'/usr/lib/','/usr/share/zoneinfo/'")
  ast.parse(s);(root/"code"/f.name).write_text(s)
 config=load(template/"experiment.json")
 def bind(x):
  if isinstance(x,str):
   x=x.replace("1014",str(seed))
   for a,b in replacements.items():x=x.replace(a,b)
   return x
  if isinstance(x,list):return [bind(v) for v in x]
  if isinstance(x,dict):return {k:bind(v) for k,v in x.items()}
  return seed if type(x) is int and x==1014 else x
 config=bind(config)
 config["datasets"]=([n for n in identity["datasets"]] if study not in ["downstream","reuse"] else {n:config["datasets"][n] for n in identity["datasets"]})
 save(root/"configs/experiment.json",config)
 if study!="repair":
  # The published ablation adapter contains the original eight-table catalog.
  # Narrow only the scheduling catalog to this study's paper scope.
  with (root/"code/catalog.py").open("a") as f:f.write("\nCATALOG={k:CATALOG[k] for k in "+repr(identity["datasets"])+"}\nNAMES=list(CATALOG)\nCLASSIFICATION=[n for n in NAMES if CATALOG[n]['task']=='classification']\n")
 if study=="ablation":
  old=case_path(work,"downstream",seed)
  if not (old/"datasets").is_dir():raise FileNotFoundError("Run downstream for this seed first")
  (root/"datasets").symlink_to(old/"datasets")
  # score_ablation's list is an operational catalog, never a numerical formula.
  p=root/"code/score_ablation.py";s=p.read_text();s=s.replace("NAMES=['real_estate','concrete','airfoil','tic_tac_toe','car_evaluation','retinopathy']","NAMES="+repr(identity["datasets"]));p.write_text(s)
  p=root/"code/report.py";s=p.read_text();s=s.replace("NAMES=['tic_tac_toe','car_evaluation','retinopathy','real_estate','concrete','airfoil']","NAMES="+repr(identity["datasets"]));p.write_text(s)
 for n in ["raha","holoclean","garf","cdc"]:
  env=Path(cfg["environments"])/n
  (root/"environments").mkdir(exist_ok=True)
  (root/"environments"/n).symlink_to(env)
 (root/"environments/postgres").symlink_to(Path(cfg["postgres_bin"]).parent)
 (root/"sources").symlink_to(Path(cfg["sources"]))
 wrapper=root/"code/v23-python"
 wrapper.write_text("#!/bin/sh\nexec "+shlex.quote(cfg["run_python"])+" -B "+shlex.quote(str(root/"code/seed_runtime.py"))+' "$@"\n')
 wrapper.chmod(0o755)
 save(meta,identity)
 save(root/"audit/release_binding.json",{"code":{str(p.relative_to(root)):sha(p) for p in (root/"code").iterdir()},"core":core_files,"numerical_engine_bytes_unchanged":True,"operational_changes":["paths","seed","selected-table catalog","owned model/PG transport"]})
 return root
def download_raw(root,study,names):
 expected=load(REPO/"provenance/raw_sources.json")
 spec=load(root/"configs/experiment.json")["datasets"]
 rawurls={
 "bc_original":"https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
 "mammographic":"https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
 "ilpd":"https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20%28ILPD%29.csv",
 "climate":"https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat",
 "voting":"https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
 "qsar":"https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"}
 for n in names:
  folder=root/("curator_cache/private" if study=="repair" else "generator_inputs/private")/n;folder.mkdir(parents=True,exist_ok=True)
  target=folder/("source.raw" if study=="repair" else "source.zip")
  if not target.exists():
   if study=="repair" and n in ["monks1","cmc"]:
    u="https://archive.ics.uci.edu/static/public/"+("70/monk+s+problems" if n=="monks1" else "30/contraceptive+method+choice")+".zip"
    import io
    with urllib.request.urlopen(u,timeout=120) as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
    member="monks-1.test" if n=="monks1" else "cmc.data"
    matches=[f for f in z.namelist() if Path(f).name==member]
    if len(matches)!=1:raise ValueError("Ambiguous archive member")
    target.write_bytes(z.read(matches[0]))
   else:
    u=rawurls[n] if study=="repair" else f"https://archive.ics.uci.edu/static/public/{spec[n]['id']}/{spec[n]['slug']}.zip"
    with urllib.request.urlopen(u,timeout=120) as r:target.write_bytes(r.read())
  if sha(target)!=expected[n]["sha256"]:raise ValueError("Official raw bytes differ from frozen benchmark: "+n+"; retain file and investigate, do not silently continue")
  save(folder/"provenance.json",{"sha256":sha(target),"origin":"official UCI, generator-only verified raw bytes","generator_only":True})
def invoke(root,cfg,entry,args=(),label=None,python=None):
 label=label or entry.removesuffix(".py")
 marker=root/"status/stages"/(label+".json");log=root/"logs"/(label+".log")
 if marker.exists():
  m=load(marker)
  if m.get("status")=="complete":return
  raise RuntimeError("Prior failed/incomplete stage; inspect before resuming: "+str(log))
 marker.parent.mkdir(parents=True,exist_ok=True)
 save(marker,{"status":"started","time":time.time()})
 cmd=[python or cfg["run_python"],"-B",str(root/"code"/entry),*map(str,args)]
 with log.open("x") as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"4"})
 save(marker,{"status":"complete" if r.returncode==0 else "failed","returncode":r.returncode,"command":cmd,"time":time.time()})
 if r.returncode:raise RuntimeError("Stage failed; original logs preserved: "+str(log))
def generate(root,study,cfg,names):
 if study=="ablation":return
 download_raw(root,study,names)
 for n in names:
  if (root/"datasets"/n/"input_freeze.json").exists():continue
  invoke(root,cfg,"generate_benchmarks.py",["--datasets",n],"generate_"+n)
def pg(root,cfg,start):
 state=root/"status/postgres.json";data=root/"outputs/postgres/cluster";bin=Path(cfg["postgres_bin"])
 if not start:
  if state.exists() and (data/"postmaster.pid").exists():
   assert Path(load(state)["cluster"])==data
   subprocess.run([str(bin/"pg_ctl"),"-D",str(data),"-m","fast","stop"],check=True)
  return
 if state.exists():
  if (data/"postmaster.pid").exists():return
  raise RuntimeError("Stopped owned PG receipt exists; inspect before restart")
 port=cfg["pg_port"];user="dataforge"+str(load(root/"case.json")["seed"])
 with socket.socket() as sock:sock.bind(("127.0.0.1",port))
 data.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run([str(bin/"initdb"),"-D",str(data),"-U",user,"--auth-local=trust","--auth-host=trust"],check=True)
 # TCP only avoids the Unix socket path limit on long checkout locations.
 subprocess.run([str(bin/"pg_ctl"),"-D",str(data),"-l",str(root/"logs/postgres.log"),"-o",f"-p {port} -h 127.0.0.1 -k ''","start"],check=True)
 save(state,{"port":port,"host":"127.0.0.1","user":user,"cluster":str(data),"owned":True})
def service(root,cfg,action):
 subprocess.run([cfg["run_python"],"-B",str(root/"code/model_service.py"),action],cwd=root,check=True)
def run_methods(root,study,cfg,names):
 if study=="ablation":
  old=case_path(root.parent,"downstream",load(root/"case.json")["seed"])
  for n in names:
   selected=load(old/"outputs/v23"/n/"selection.json")["skill"]
   invoke(root,cfg,"execute_arm.py",["--dataset",n,"--arm","A_saved","--skill",selected,"--out",root/"outputs"/n/"A_saved"],n+"_A_saved")
  service(root,cfg,"start")
  try:
   for n in names:
    for a in ["A"]:
     invoke(root,cfg,"learn_variant.py",["--dataset",n,"--arm",a],n+"_learn_"+a)
     skill=load(root/"work"/n/a/"selection.json")["skill"]
     for arm in (["A","B","D"] if a=="A" else [a]):
      invoke(root,cfg,"execute_arm.py",["--dataset",n,"--arm",arm,"--skill",skill,"--out",root/"outputs"/n/arm],n+"_execute_"+arm)
  finally:service(root,cfg,"stop")
  save(root/"status/all_outputs_closed.json",{"time":time.time(),"datasets":names,"outputs_per_table":4,"scored_arms":["A","B","D"],"replay_arm":"A_saved"})
  return
 if all((root/"status"/("model_done_"+n+".json")).exists() for n in names):
  return run_baselines(root,study,cfg,names)
 service(root,cfg,"start")
 try:
  for n in names:invoke(root,cfg,"run_v23_learning.py",["--dataset",n],n+"_learning")
  for n in names:
   if (root/"outputs"/n/"primary_llm/output_freeze.json").exists():continue
   try:invoke(root,cfg,"run_primary_llm_v3.py",["--dataset",n],n+"_primary")
   except RuntimeError:
    log=(root/"logs"/(n+"_primary.log")).read_text()
    reviewed=study=="repair" and load(root/"case.json")["seed"]==1015 and n=="qsar" and "Incomplete real LLM output; do not score as zero" in log
    if not reviewed:raise
    invoke(root,cfg,"resume_primary_format.py",["--dataset",n],n+"_primary_reviewed_format_continuation")
  for n in names:
   invoke(root,cfg,"run_tool_llm_v5.py",["--dataset",n],n+"_tool")
   if study=="downstream" and load(root/"configs/experiment.json")["datasets"][n]["task"]=="classification":
    env={**os.environ,"EVIDFORGE_SUPERVISED_COMPONENT":"caafe"}
    before=os.environ.get("EVIDFORGE_SUPERVISED_COMPONENT");os.environ["EVIDFORGE_SUPERVISED_COMPONENT"]="caafe"
    try:invoke(root,cfg,"run_caafe.py",["--dataset",n],n+"_caafe")
    finally:
     if before is None:os.environ.pop("EVIDFORGE_SUPERVISED_COMPONENT",None)
     else:os.environ["EVIDFORGE_SUPERVISED_COMPONENT"]=before
   save(root/"status"/("model_done_"+n+".json"),{"closed":True})
 finally:service(root,cfg,"stop")
 run_baselines(root,study,cfg,names)
def run_baselines(root,study,cfg,names):
 for n in names:
  for method in ["garf","cdc"]:
   try:invoke(root,cfg,"run_"+method+".py",["--dataset",n],n+"_"+method,python=str(root/"environments"/method/"bin/python"))
   except RuntimeError:
    log=(root/"logs"/(n+"_"+method+".log")).read_text()
    known=study=="repair" and load(root/"case.json")["seed"]==1015 and n=="qsar" and method=="cdc" and "AssertionError" in log and ("exactly 2" in log or "exactly two" in log)
    if not known:raise
    save(root/"status/closed_unavailable/qsar_cdc.json",{"status":"N/A","reason":"Previously reviewed native binary single-class source-fit assertion; no constant model/split change","log":"logs/qsar_cdc.log"})
  invoke(root,cfg,"run_raha_detection.py",["--dataset",n],n+"_raha",python=str(root/"environments/raha/bin/python"))
 pg(root,cfg,True)
 try:
  for n in names:invoke(root,cfg,"run_holoclean.py",["--dataset",n],n+"_holo",python=str(root/"environments/holoclean/bin/python"))
 finally:pg(root,cfg,False)
def score(root,study,cfg):
 entry={"repair":"score_all.py","downstream":"score_downstream.py","ablation":"score_ablation.py"}[study]
 invoke(root,cfg,entry,["--close-incomplete"] if study=="repair" else [],"independent_scoring")
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
 sub.add_parser("doctor")
 t=sub.add_parser("tables");t.add_argument("--source",choices=["recorded","runs"],default="recorded");t.add_argument("--work",default="runs");t.add_argument("--output",default="results/generated")
 u=sub.add_parser("reuse");u.add_argument("--study",choices=["repair","downstream"],required=True);u.add_argument("--seed",type=int,choices=PROTOCOL["seeds"],required=True);u.add_argument("--work",default="runs");u.add_argument("--config");u.add_argument("--dataset",required=True)
 for cmd in ["prepare","data","run","score"]:
  s=sub.add_parser(cmd);s.add_argument("--study",choices=["repair","downstream","ablation"],required=True);s.add_argument("--seed",type=int,choices=PROTOCOL["seeds"],required=True);s.add_argument("--work",default="runs");s.add_argument("--config");s.add_argument("--datasets",nargs="+")
 a=p.parse_args()
 if a.command=="doctor":
  print(json.dumps({"python":sys.version,"core_files_verified":check_engine(),"platform":sys.platform,"machine":defaults(),"expected_results_are_not_fresh_runs":True},indent=2));return
 if a.command=="tables":
  from tables import build
  build(REPO,Path(a.output),a.source,Path(a.work));return
 cfg=settings(a.config)
 if a.command=="reuse":
  root=case_path(a.work,a.study,a.seed)
  if load(root/"case.json")["machine"]!=cfg:raise ValueError("Machine configuration differs")
  shutil.copyfile(REPO/"experiments/reuse.py",root/"code/reuse.py")
  invoke(root,cfg,"reuse.py",["--dataset",a.dataset,"--out",root/"outputs"/a.dataset/"reuse"],a.dataset+"_reuse")
  return
 root=initialize(a.work,a.study,a.seed,cfg,a.datasets)
 names=load(root/"case.json")["datasets"]
 if a.command=="prepare":print(root);return
 if a.command in ["data","run"]:generate(root,a.study,cfg,names)
 if a.command=="run":run_methods(root,a.study,cfg,names)
 if a.command in ["run","score"]:score(root,a.study,cfg)
