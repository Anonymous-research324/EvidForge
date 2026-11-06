from pathlib import Path
import json,hashlib,statistics,math,tarfile,re
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(R/'provenance/paper_release_files.json')
for path,h in manifest['files'].items():assert sha(R/path)==h,path
core=load(R/'provenance/core_sha256.json')['files']
for path,h in core.items():assert sha(R/'engine'/path)==h,path
binding=load(R/'provenance/paper_binding.json')
for path,h in {**binding['table_snippets'],**binding['paper_figures']}.items():assert sha(R/path)==h,path
checks={'F1':0,'mean_sd':0}
def walk(x):
 if isinstance(x,list):
  for v in x:walk(v)
 elif isinstance(x,dict):
  if all(k in x for k in ['errors','changes','exact','F1']) and isinstance(x['F1'],(int,float)):
   n=x['errors']+x['changes'];assert math.isclose(x['F1'],2*x['exact']/n if n else 0.,abs_tol=1e-10);checks['F1']+=1
  v=x.get('values',x.get('per_seed',x.get('seed_values')))
  if isinstance(v,dict):v=list(v.values())
  if isinstance(v,list) and len(v)>1 and all(isinstance(a,(int,float)) for a in v) and isinstance(x.get('mean'),(int,float)):
   assert math.isclose(x['mean'],statistics.mean(v),abs_tol=1e-10)
   sd=x.get('sd',x.get('sample_sd'))
   if sd is not None:assert math.isclose(sd,statistics.stdev(v),abs_tol=1e-10)
   checks['mean_sd']+=1
  for v in x.values():walk(v)
for p in (R/'results').rglob('*.json'):walk(load(p))
for s in load(R/'provenance/paper_experiments.json')['studies']:
 for x in s['source_records']:assert (R/x['published_file']).exists()
 archive=R/'experiments/paper_runs'/s['id']/'execution_code.tar.gz'
 if archive.exists():
  expected={x['path']:x['sha256'] for x in s['execution_code_files']}
  with tarfile.open(archive) as tf:
   seen=set()
   for m in tf:
    assert m.isfile() and m.name in expected
    assert hashlib.sha256(tf.extractfile(m).read()).hexdigest()==expected[m.name];seen.add(m.name)
  assert seen==set(expected)
print(json.dumps({'passed':True,'release_files':len(manifest['files']),'core':len(core),'paper_tables':len(binding['table_snippets']),**checks,'formal_reference_read':False,'new_model_calls':0},ensure_ascii=False,indent=2))
