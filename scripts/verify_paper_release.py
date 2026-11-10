"""Verify the immutable system/environment and the eight-figure publication subset."""
from pathlib import Path
import json,hashlib,statistics,math,tarfile
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 manifest=load(R/'provenance/paper_release_files.json')['files']
 actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.pyc' and p!=R/'provenance/paper_release_files.json'}
 assert actual==set(manifest),'Unexpected or missing release files'
 for rel,h in manifest.items():assert sha(R/rel)==h,rel
 core=load(R/'provenance/core_sha256.json')['files']
 for rel,h in core.items():assert sha(R/'engine'/rel)==h,rel
 env=load(R/'provenance/environment_sha256.json')['files']
 for rel,h in env.items():assert sha(R/rel)==h,rel
 scopes=load(R/'provenance/paper_scope.json')['studies']
 assert {p.name for p in (R/'results/studies').iterdir() if p.is_dir()}==set(scopes)
 assert {p.name for p in (R/'experiments/paper_runs').iterdir() if p.is_dir()}==set(scopes)
 checks={'F1':0,'mean_sd':0}
 def walk(x):
  if isinstance(x,list):
   for v in x:walk(v)
  elif isinstance(x,dict):
   if all(k in x for k in ['errors','changes','exact','F1']) and isinstance(x['F1'],(int,float)):
    n=x['errors']+x['changes'];assert math.isclose(x['F1'],2*x['exact']/n if n else 0.,abs_tol=1e-10);checks['F1']+=1
   values=next((x[k] for k in ['values','per_seed','seed_values','paired_values'] if k in x),None)
   if isinstance(values,dict):values=list(values.values())
   if isinstance(values,list) and len(values)>1 and all(isinstance(v,(int,float)) for v in values) and isinstance(x.get('mean'),(int,float)):
    assert math.isclose(x['mean'],statistics.mean(values),abs_tol=1e-10)
    sd=x.get('sd',x.get('sample_sd'))
    if sd is not None:assert math.isclose(sd,statistics.stdev(values),abs_tol=1e-10)
    checks['mean_sd']+=1
   for v in x.values():walk(v)
 for p in (R/'results').rglob('*.json'):walk(load(p))
 for study in load(R/'provenance/paper_experiments.json')['studies']:
  for rec in study['source_records']:assert sha(R/rec['published_file'])==rec['published_sha256']
  archive=R/'experiments/paper_runs'/study['id']/'execution_code.tar.gz'
  if archive.exists():
   expected={x['path']:x['sha256'] for x in study['execution_code_files']};seen=set()
   with tarfile.open(archive) as tf:
    for member in tf:
     assert member.isfile() and member.name in expected
     assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==expected[member.name];seen.add(member.name)
   assert seen==set(expected)
 for item in load(R/'provenance/retained_records.json')['files']:
  obj=load(R/item['path'])
  for key in item['container']:obj=obj[key]
  assert len(obj)==len(item['record_sha256'])
  got=[hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest() for x in obj]
  assert got==item['record_sha256'],item['path']
 return {'passed':True,'files':len(manifest),'core':len(core),'environment_files':len(env),'study_groups':len(scopes),**checks,'model_experiments_rerun':False}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
