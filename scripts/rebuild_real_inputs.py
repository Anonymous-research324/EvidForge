#!/usr/bin/env python3
"""Build final real-error cases from public author pairs and fixed row-role config.

The configuration contains row numbers, synthetic entity groups, schema and SHA
only. No old case feature/reference values, predictions or scores are consumed.
The caller supplies the public author CSVs named in public_data_sources.json.
"""
from pathlib import Path
import argparse,csv,gzip,json,hashlib,shutil
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def csvwrite(p,fields,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def build(source_root,output):
 src=Path(source_root).resolve();out=Path(output).resolve();study=R/'experiments/paper_runs/real_dirty';sources=read(study/'public_data_sources.json');roles=json.loads(gzip.decompress((study/'row_partition_config.json.gz').read_bytes()))['cases']
 if out.exists():raise FileExistsError('Fresh output required; preserve existing input/failure evidence')
 for item in sources['source_files']:
  p=src/item['path']
  if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Public author source absent or changed: '+str(p))
 loaded={}
 for ds in {v['metadata']['dataset'] for v in roles.values()}:
  parent=sources['dirty_only_fixed_subsets'].get(ds,{}).get('parent_dataset',{'library_official':'address_library'}.get(ds,ds));f,d=csvread(src/'sources'/parent/'dirty.csv');cf,t=csvread(src/'sources'/parent/'reference.csv')
  if len(f)!=len(cf) or len(d)!=len(t):raise ValueError('Author pair shape differs '+ds)
  mapping=dict(zip(f,cf))
  if ds=='beers':
   by={r['index']:r for r in t};assert len(by)==len(t);t=[by[r['index']] for r in d]
  elif 'id' in f and ds!='movies':
   by={r['id']:r for r in t};assert len(by)==len(t);t=[by[r['id']] for r in d]
  truth=[{k:r[mapping[k]] for k in f} for r in t]
  if ds in sources['dirty_only_fixed_subsets']:
   cfg=sources['dirty_only_fixed_subsets'][ds];indices=cfg['selection_indices'];assert len(d)==cfg['parent_rows'] and len(indices)==cfg['actual_rows'];d=[d[i] for i in indices];truth=[truth[i] for i in indices]
  loaded[ds]=(f,d,truth)
 out.mkdir(parents=True)
 for name,spec in roles.items():
  meta=spec['metadata'];fields,dirty,truth=loaded[meta['dataset']];c=out/'cases'/name;schema=spec['static_public_json']['declaration.json'];targets=schema['targets'];allfields=['entity_id','__rowkey__',*fields]
  def row(i,g):return {'entity_id':g,'__rowkey__':f'r{i:06d}',**dirty[i]}
  gids={i:g for entries in spec['zone_rows'].values() for i,g in entries};bundle={}
  for zone in ['learning','validation']:
   bundle[zone]=[]
   for group in spec['groups'][zone]:
    g={k:v for k,v in group.items() if k!='source_indices'};g['records']=[{'row_key':f'r{i:06d}','dirty':row(i,gids[i]),'truth':{k:truth[i][k] for k in targets}} for i in group['source_indices']];bundle[zone].append(g)
  write(c/'public/development_bundle.json',bundle)
  for n,values in spec['static_public_json'].items():write(c/'public'/n,values)
  expected=spec['expected_input_freeze']
  if 'learning_only.json' in expected['public']:write(c/'public/learning_only.json',{'learning':bundle['learning']})
  if 'validation_only.json' in expected['public']:write(c/'public/validation_only.json',{'validation':bundle['validation']})
  for zone,entries in spec['zone_rows'].items():
   csvwrite(c/'public'/(zone+'_dirty.csv'),allfields,[row(i,g) for i,g in entries])
   if zone in ['validation','repair']:
    ts=[{k:truth[i][k] for k in targets} for i,g in entries];write(c/'private'/(zone+'_truth.json'),{'row_keys':[f'r{i:06d}' for i,g in entries],'truth':ts,'known_mask':[[k for k in targets if k in t] for t in ts]})
  write(c/'case.json',meta)
  for zone in ['public','private']:
   actual={str(p.relative_to(c/zone)):sha(p) for p in (c/zone).rglob('*') if p.is_file()}
   if actual!=expected[zone]:raise ValueError('Original input hash mismatch: '+name+'/'+zone+'; preserve evidence, never accept a different split or silently patch values')
  assert sha(c/'case.json')==expected.get('case_sha',expected.get('case_sha256'))
  write(c/'input_freeze.json',expected)
 names=list(roles);write(out/'configs/cases.json',names)
 for p in (study/'original_config').iterdir():shutil.copyfile(p,out/'configs'/p.name)
 shutil.copyfile(R/'provenance/model-files-9B.sha256',out/'configs/model-files-9B.sha256')
 files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
 write(out/'input_export.json',{'study':'real_dirty','files':files,'kind':'PUBLIC_AUTHOR_PAIRS_FIXED_ROLE_METADATA','case_count':len(names),'all_original_input_hashes_match':True,'old_case_values_or_scores_read':False,'model_calls':0,'fits':0})
 return {'cases':len(names),'original_input_hashes_match':True,'model_calls':0,'fits':0,'input_export':str(out/'input_export.json')}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.source_root,a.output),ensure_ascii=False,indent=2))
