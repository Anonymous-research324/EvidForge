#!/usr/bin/env python3
"""Recreate the final21 TabClean-comparison inputs with the original UCI curator.

Downloads only official raw archives. No model calls, source fitting, old case
values, old predictions or recorded performance scores are read. Source/role
fingerprints must match the original paper cases before the export is usable.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,tarfile,shutil
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def build(output,raw_cache=None):
 out=Path(output).resolve()
 if out.exists():raise FileExistsError('Fresh output required; preserve partial evidence')
 out.mkdir(parents=True);(out/'code').mkdir();(out/'cases').mkdir();(out/'configs').mkdir()
 study=R/'experiments/paper_runs/tabclean_same9b';spec=read(study/'original_config/simple_protocol_v1.json');expected=read(study/'input_fingerprints.json')['cases']
 with tarfile.open(study/'execution_code.tar.gz') as tf:
  for rel in ['code/common.py','code/data.py']:
   data=tf.extractfile(rel).read();(out/rel).write_bytes(data)
 write(out/'configs/protocol_v1.json',read(study/'input_source_catalog.json'))
 sys.path[:0]=[str(out/'code'),str(R/'engine/source_runtime')]
 # Original common module computes ROOT from the copied file, and original data
 # functions are used unchanged. Do not run data.py's old pilot main entry.
 for name in ['common','data']:sys.modules.pop(name,None)
 import common,data
 names=[]
 for dataset in spec['datasets']:
  if raw_cache:
   cache=Path(raw_cache);src=cache/(dataset+'.zip');src=src if src.exists() else cache/dataset/'original.zip';dst=out/'sources'/dataset/'original.zip'
   if not src.is_file():raise FileNotFoundError(src)
   dst.parent.mkdir(parents=True);shutil.copyfile(src,dst)
  for seed in spec['seeds']:
   name=data.build(dataset,'confirmation',seed,'mixed');names.append(name);case=out/'cases'/name;made=read(case/'input_freeze.json')
   if name not in expected or made!=expected[name]:raise ValueError('Input bytes differ from final paper: '+name+'; preserve, do not silently accept or retry')
   for zone in ['public','private']:
    for rel,h in expected[name][zone].items():assert sha(case/zone/rel)==h
 for rel in ['simple_protocol_v1.json','run_config.json']:shutil.copyfile(study/'original_config'/rel,out/'configs'/rel)
 write(out/'configs/cases.json',names);write(out/'configs/simple_cases.json',names);shutil.copyfile(R/'provenance/model-files-9B.sha256',out/'configs/model-files-9B.sha256')
 # Export only per-case method/scorer inputs and static configuration. Raw author
 # data and curator code/cache stay outside the method input allowlist.
 files={}
 for name in names:
  case=out/'cases'/name
  for p in case.rglob('*'):
   if p.is_file() and (p.parent.name in ['public','private'] or p.name in ['case.json','input_freeze.json']):files[str(p.relative_to(out))]=sha(p)
 for p in (out/'configs').iterdir():
  if p.is_file():files[str(p.relative_to(out))]=sha(p)
 write(out/'input_export.json',{'study':'tabclean_same9b','files':files,'kind':'OFFICIAL_UCI_ORIGINAL_CURATOR','case_count':len(names),'all_original_input_sha_match':True,'new_model_calls':0,'new_fits':0})
 return {'cases':len(names),'original_input_hashes_match':True,'model_calls':0,'fits':0,'input_export':str(out/'input_export.json')}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--raw-cache');a=p.parse_args();print(json.dumps(build(a.output,a.raw_cache),ensure_ascii=False,indent=2))
