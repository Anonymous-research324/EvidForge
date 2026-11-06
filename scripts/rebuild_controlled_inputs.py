"""Rebuild fixed controlled-error inputs from official raw UCI archives and public role IDs.

No historical private references, predictions, skills or scores are consumed.
The original curator is executed with a clean source/role facade; all generated
public and private input bytes must match the published SHA contract before use.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,shutil,sys,tarfile,tempfile,types,zipfile
REPO=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,fields,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def csvread(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def freeze(p):return {str(q.relative_to(p)):sha(q) for q in sorted(Path(p).rglob('*')) if q.is_file() and q.suffix!='.pyc'}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rebuild(study,raw_dir,output,selected=None):
 if study not in {'new_errors','llm_digit70'}:raise ValueError('Only these original controlled curators are implemented')
 output=Path(output).resolve();raw_dir=Path(raw_dir).resolve()
 if output.exists():raise FileExistsError('Fresh output required; failed evidence is retained')
 specs=load(REPO/'provenance/controlled_rebuild_specs.json')[study];roles=load(REPO/'provenance/reproduction_public_roles_seed1013.json')['datasets']
 names=selected or list(specs['cases'])
 if any(n not in specs['cases'] for n in names):raise ValueError('Unknown original case')
 output.parent.mkdir(parents=True,exist_ok=True);output.mkdir();scratch=output/'curator_only';scratch.mkdir();code=scratch/'code';code.mkdir();run=scratch/'run';(run/'cases').mkdir(parents=True)
 for sub in ['configs','audit','reports','dependencies']:(run/sub).mkdir()
 save(run/'configs/protocol.json',specs['protocol'])
 needed={'code/curator.py','code/original_curator_functions.py','code/corruption_functions.py'}
 with tarfile.open(REPO/'experiments/paper_runs'/study/'execution_code.tar.gz') as tf:
  for m in tf:
   if m.name not in needed:continue
   (code/Path(m.name).name).write_bytes(tf.extractfile(m).read())
 common=types.ModuleType('common');common.ROOT=run;common.ENGINE=REPO/'engine';common.load=load;common.save=save;common.sha=sha;common.freeze=freeze;common.read_csv=csvread;common.write_csv=csvwrite;sys.modules['common']=common
 sys.path[:0]=[str(code),str(REPO/'engine'),str(REPO/'engine/source_runtime')]
 funcs=module('original_curator_functions',code/'original_curator_functions.py');curator=module('curator',code/'curator.py')
 for ds in {specs['cases'][n]['case_metadata']['dataset'] for n in names}:
  role=roles[ds];raw=raw_dir/(ds+'.zip')
  if not raw.exists() or sha(raw)!=role['raw_zip_sha256']:raise ValueError('Official raw archive absent/different: '+str(raw))
  with zipfile.ZipFile(raw) as z:values,labels,ids,*_=funcs.parse(z,curator.SPECS[ds])
  del labels
  groups=funcs.make_groups(values,ids);columns=role['declaration']['targets'];fields=['entity_id',*columns]
  source=scratch/'EvidForge_Downstream_Study_seed1013/datasets'/ds;(source/'private').mkdir(parents=True);shutil.copyfile(raw,source/'private/source.zip');save(source/'public/declaration.json',role['declaration'])
  bundle={}
  for zone in ['learning','validation']:
   bundle[zone]=[]
   for group in role[zone]:
    ix=groups[group['group']];keys=[f'{ds}-{i:05d}' for i in ix]
    if keys!=group['row_keys']:raise ValueError('Raw group/row identity mismatch')
    bundle[zone].append({'group':group['group'],'records':[{'row_key':k,'dirty':{'entity_id':group['group'],**dict(zip(columns,values[i]))},'truth':dict(zip(columns,values[i]))} for i,k in zip(ix,keys)]})
  save(source/'public/development_bundle.json',bundle)
  repair=[{'entity_id':g,**dict(zip(columns,values[i]))} for g in role['repair_group_order'] for i in groups[g]]
  if [r['entity_id'] for r in repair]!=role['repair_entity_order']:raise ValueError('Repair role order differs')
  csvwrite(source/'public/repair_dirty.csv',fields,repair)
 # Original curator resolves its old dependency as ROOT.parent/<source-role root>.
 if study=='llm_digit70':curator.B=scratch
 result=[];files={}
 for name in names:
  original=specs['cases'][name];meta=original['case_metadata'];generated=curator.create(meta['dataset'],meta['seed'],meta['profile']) if study=='new_errors' else curator.create(meta['dataset'],meta['seed'])
  if generated!=name:raise ValueError('Original case identity differs')
  c=run/'cases'/name;made=load(c/'case.json')
  if made['counts']!=meta['counts']:raise ValueError('Pollution count contract differs')
  check=original['expected_input_freeze']
  for zone in ['public','private']:
   if freeze(c/zone)!=check[zone]:raise ValueError('Reconstructed '+zone+' bytes differ for '+name)
  # Reuse immutable original metadata only after ALL actual method/evaluation input
  # bytes match. It binds original role provenance, not a claim of reading old files.
  save(c/'case.json',meta);save(c/'input_freeze.json',check)
  if sha(c/'case.json')!=check['case_sha256']:raise ValueError('Case metadata serialization differs')
  target=output/'cases'/name;target.parent.mkdir(exist_ok=True)
  for sub in ['public','private']:shutil.copytree(c/sub,target/sub)
  for f in ['case.json','input_freeze.json']:shutil.copyfile(c/f,target/f)
  result.append({'case':name,'public_private_byte_parity':True,'pollution_counts_equal':True})
 # Match the portable worker input-export contract, with actual original scientific protocol.
 save(output/'configs/protocol.json',specs['protocol']);save(output/'configs/cases.json',names);save(output/'configs/lanes.json',{'0':names[::2],'1':names[1::2]});shutil.copyfile(REPO/'provenance/model-files-9B.sha256',output/'configs/model-files-9B.sha256')
 for p in output.rglob('*'):
  if p.is_file() and 'curator_only' not in p.relative_to(output).parts:files[str(p.relative_to(output))]=sha(p)
 save(output/'input_export.json',{'study':study,'kind':'OFFICIAL_RAW_PLUS_PUBLIC_ROLE_REBUILD','files':files,'cases':result,'old_private_reference_read':False,'old_observed_feature_values_read':False,'new_model_calls':0,'new_fits':0})
 return {'study':study,'cases':len(names),'files':len(files),'all_original_input_bytes_match':True,'old_formal_reference_read':False,'model_calls':0,'fits':0}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=['new_errors','llm_digit70'],required=True);p.add_argument('--raw-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',nargs='+');a=p.parse_args();print(json.dumps(rebuild(a.study,a.raw_dir,a.output,a.cases),ensure_ascii=False,indent=2))
