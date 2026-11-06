"""Execute only a checked LLM repair function on provided teaching/query targets."""
from pathlib import Path
import sys,json,ast,os,random,time
import numpy as np
random.seed(1014);np.random.seed(1014)
INPUT=Path(sys.argv[1]).resolve();OUTPUT=Path(sys.argv[2]).resolve()
BASE=Path(__file__).resolve().parents[1];WORK=INPUT.parent
SPEC=json.loads(INPUT.read_text())
ALLOWED={'re','math','decimal','statistics','collections','numpy','pandas','sklearn.ensemble','sklearn.linear_model','sklearn.neighbors','sklearn.impute','sklearn.preprocessing','sklearn.pipeline','sklearn','sklearn.metrics','sklearn.model_selection','random'}
BANNED={'open','eval','exec','compile','globals','locals','getattr','setattr','__import__','vars','dir','input','print','breakpoint'}
tree=ast.parse(SPEC['program'])
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom)):
  names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
  if any(x not in ALLOWED for x in names):raise ValueError('Unsupported import: '+','.join(str(x) for x in names if x not in ALLOWED))
 if isinstance(n,ast.Name) and (n.id in BANNED or n.id.startswith('__')):raise ValueError('Unsafe name')
 if isinstance(n,ast.Attribute) and n.attr.startswith('_'):raise ValueError('Private attribute not allowed')
 if isinstance(n,(ast.Global,ast.Nonlocal)):raise ValueError('Global mutation is not allowed')
 if isinstance(n,ast.Call):
  for kw in n.keywords:
   if kw.arg in ('random_state','random_seed','seed') and isinstance(kw.value,ast.Constant) and kw.value.value not in (1014,None):raise ValueError('Use random seed1014')
   if kw.arg=='n_jobs' and isinstance(kw.value,ast.UnaryOp) and isinstance(kw.value.op,ast.USub):raise ValueError('Use at most4 CPU workers')
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).resolve()
  if BASE==p or BASE in p.parents:
   if WORK!=p and WORK not in p.parents and (BASE/'environments') not in p.parents and p!=Path(__file__).resolve():raise PermissionError('Worker cannot read other project data')
  elif not any(str(p).startswith(prefix) for prefix in ['@RUN_ENV@/','@PYTHON_BASE@/','/usr/lib/','/usr/share/zoneinfo/','/dev/null','/proc/self/']):
   raise PermissionError('Worker files restricted to interpreter and execution directory')
 if event in ('socket.connect','socket.bind'):raise PermissionError('Worker has no network access')
sys.addaudithook(audit)
space={}
exec(compile(tree,'<llm-cleaning-program>','exec'),space)
f=space.get('repair')
if not callable(f):raise ValueError('Define repair(rows, summary, teaching)')
rows=SPEC['rows'];before=[dict(r) for r in rows]
summary=dict(SPEC['summary']);summary['learned_summary']=SPEC['summary']
answer=f([dict(r) for r in rows],summary,SPEC['teaching'])
if isinstance(answer,dict) and all(isinstance(k,str) and k.startswith('row_') for k in answer):answer={'repairs':answer}
if isinstance(answer,dict) and 'repairs' in answer:
 result=[dict(r) for r in before]
 for k,patch in answer['repairs'].items():
  i=int(k.split('_')[-1])
  if not 0<=i<len(rows) or not set(patch)<=set(rows[i]):raise ValueError('Invalid patch coordinate')
  result[i].update(patch)
else:result=answer
if not isinstance(result,list) or len(result)!=len(rows):raise ValueError('Return same row count')
clean=[]
for old,r in zip(before,result):
 if not isinstance(r,dict) or set(r)!=set(old):raise ValueError('Target fields changed')
 clean.append({c:'' if v is None or isinstance(v,(float,np.floating)) and np.isnan(v) else str(v) for c,v in r.items()})
OUTPUT.write_text(json.dumps(clean,allow_nan=False))
