from pathlib import Path
import sys,json,os,csv,hashlib,time,uuid
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('@DOWNSTREAM_ROOT@')
sys.path.insert(0,str(ROOT/'dependencies'))
if Path(sys.argv[0]).name not in ['score_ablation.py']:
 from role_guard import install
 install()
def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def read_csv(p):
 with Path(p).open(newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def write_csv(p,fields,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def freeze(p):return {str(x.relative_to(p)):digest(x) for x in sorted(Path(p).rglob('*')) if x.is_file() and x.suffix!='.pyc'}
def status(phase,**kw):save(ROOT/'status'/('progress_'+str(os.getpid())+'.json'),{'phase':phase,'at':time.time(),**kw})
