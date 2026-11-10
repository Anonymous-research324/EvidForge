from pathlib import Path
import json,hashlib,csv,io,os,time,uuid
ROOT=Path(__file__).resolve().parents[1]
import sys
if sys.version_info >= (3,12):sys.path.insert(0,str(ROOT/"dependencies"))
if Path(sys.argv[0]).name not in ["generate_benchmarks.py","score_downstream.py","self_check.py"]:
 from role_guard import install as install_reference_guard
 install_reference_guard()
if Path(sys.argv[0]).name == 'score_downstream.py':
 while (ROOT/'status/caafe_recovery_pending.json').exists():time.sleep(2)
SEED=1014
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def read_csv(p):
 with Path(p).open(newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def write_csv(p,fields,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def status(phase,**info):save(ROOT/'status/progress.json',{'phase':phase,'timestamp':time.time(),'seed':SEED,**info})
def freeze(folder):
 folder=Path(folder);return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.endswith('.pyc')}
