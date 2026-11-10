#!/usr/bin/env python3
"""Reconstruct paper inputs locally from pinned official UCI archives.

Only original pure curator functions are executed. No saved input rows, formal
answers, predictions, fitted skills, model calls or experiment scores are read.
Historical input hashes verify the newly generated scientific bytes. Fresh
case/cache provenance is explicitly rebound instead of copying server metadata.
"""
from pathlib import Path
import argparse,contextlib,hashlib,importlib.util,json,shutil,sys,tarfile,types
from rebuild_controlled_inputs import csvread,csvwrite,freeze,load,save,sha

REPO=Path(__file__).resolve().parents[1]
SPEC=REPO/'provenance/supplement_rebuild_specs.json'
STUDIES=['advisor_supplement','robustness_retinopathy','sealed_reuse']

@contextlib.contextmanager
def curator_namespace(run,code):
 names=['common','catalog','source_generator','curate_reuse','rebuild_generator']
 previous={n:sys.modules.get(n) for n in names};path=list(sys.path)
 common=types.ModuleType('common')
 for key,value in {'ROOT':run,'SEED':1014,'ENGINE':REPO/'engine','load':load,'save':save,'sha':sha,'digest':sha,'freeze':freeze,'read_csv':csvread,'write_csv':csvwrite}.items():setattr(common,key,value)
 sys.modules['common']=common;sys.path.insert(0,str(code))
 try:yield common
 finally:
  sys.path[:]=path
  for name,value in previous.items():
   if value is None:sys.modules.pop(name,None)
   else:sys.modules[name]=value

def module(name,path,source=None):
 result=types.ModuleType(name);result.__file__=str(path);sys.modules[name]=result
 exec(compile(source if source is not None else path.read_text(),str(path),'exec'),result.__dict__)
 return result

def extract_curators(code):
 study=next(x for x in load(REPO/'provenance/paper_experiments.json')['studies'] if x['id']=='sealed_reuse')
 expected={x['path']:x['sha256'] for x in study['execution_code_files']}
 with tarfile.open(REPO/'experiments/paper_runs/sealed_reuse/execution_code.tar.gz') as tf:
  for name in ['code/catalog.py','code/source_generator.py','code/curate_reuse.py']:
   data=tf.extractfile(name).read()
   if hashlib.sha256(data).hexdigest()!=expected[name]:raise ValueError('Archived curator source changed: '+name)
   (code/Path(name).name).write_bytes(data)

def official_cache(base,dataset,raw_dir):
 expected=load(REPO/'provenance/raw_sources.json')[dataset]['sha256']
 raw=raw_dir/(dataset+'.zip')
 if not raw.is_file() or sha(raw)!=expected:raise ValueError('Official raw source absent/different: '+str(raw))
 destination=base/'generator_inputs/private'/dataset;destination.mkdir(parents=True)
 shutil.copyfile(raw,destination/'source.zip')
 save(destination/'provenance.json',{'sha256':expected,'origin':'official UCI raw archive; verified locally for publication reconstruction','generator_only':True})

def verify_scientific(case,expected):
 checked=0;fresh=[]
 for zone,files in expected.items():
  actual=freeze(case/zone)
  if set(actual)!=set(files):raise ValueError('Rebuilt input file inventory differs: '+str(case)+'/'+zone)
  for name,h in files.items():
   if zone=='private' and name=='source_provenance.json':fresh.append(zone+'/'+name);continue
   if actual[name]!=h:raise ValueError('Rebuilt scientific input bytes differ: '+str(case)+'/'+zone+'/'+name)
   checked+=1
 return checked,fresh

def rebuild(study,raw_dir,output):
 if study not in STUDIES:raise ValueError('Unsupported paper study')
 raw_dir=Path(raw_dir).resolve();out=Path(output).resolve()
 if out.exists():raise FileExistsError('Fresh output required; preserve previous or failed evidence')
 specs=load(SPEC);spec=specs['studies'][study]
 declared=next(x['case_ids'] for x in load(REPO/'provenance/reproduction_data_sources.json')['studies'] if x['study']==study)
 if spec['case_ids']!=declared:raise ValueError('Curator scope differs from the paper catalog')
 out.mkdir(parents=True);scratch=out/'curator_only';code=scratch/'code';run=scratch/'run'
 code.mkdir(parents=True);(run/'cases').mkdir(parents=True);extract_curators(code)
 source=code/'source_generator.py'
 if sha(source)!=specs['generator_sha256']:raise ValueError('Original generator fingerprint differs')
 source_text=source.read_text();rows=[]
 with curator_namespace(run,code) as common:
  module('catalog',code/'catalog.py')
  if study=='sealed_reuse':
   generator=module('source_generator',source)
   curator=module('curate_reuse',code/'curate_reuse.py')
   curator.OLD=scratch/'official_seed_source'
   for ds in sorted({x['parameters']['dataset'] for x in spec['cases'].values()}):official_cache(curator.OLD,ds,raw_dir)
  for name in declared:
   item=spec['cases'][name];params=item['parameters'];ds=params['dataset'];seed=params['seed']
   case=run/'cases'/name
   if study=='sealed_reuse':
    if curator.make(ds,seed)!=name:raise ValueError('Original reuse case identity differs')
    if sha(case/'case.json')!=item['expected_deterministic_case_sha256']:raise ValueError('Original reuse data seed, batch identities or metadata differ')
   else:
    case.mkdir();official_cache(case,ds,raw_dir)
    # Preserve the source/schema namespace recorded in the original input hashes.
    # The advisor robustness cases retained 1014 labels across model seeds.
    text=source_text.replace('1014',str(params['source_namespace_seed']))
    if params['suite']!='main':
     old='round(.7*len(eligible))'
     if text.count(old)!=1:raise ValueError('Unrecognized original pollution-rate binding')
     text=text.replace(old,'round('+repr(params['rate'])+'*len(eligible))',1)
     if params['error_type']!='mixed':
      old="['blank','digit','scale','affix']"
      if text.count(old)!=1:raise ValueError('Unrecognized original error-family binding')
      text=text.replace(old,repr([params['error_type']]),1)
    generator=module('rebuild_generator',source,text);generator.ROOT=case;generator.SEED=seed;generator.generate(ds)
    made=case/'datasets'/ds
    for zone,names in item['expected_original_input_sha256'].items():
     (case/zone).mkdir()
     for rel in names:
      target=case/zone/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(made/zone/rel,target)
    save(case/'case.json',{'id':name,**params,'provenance':{'kind':'OFFICIAL_UCI_LOCAL_REBUILD','generator_sha256':specs['generator_sha256'],'raw_archive_sha256':load(REPO/'provenance/raw_sources.json')[ds]['sha256'],'no_historical_input_rows_or_answers_read':True}})
   count,fresh=verify_scientific(case,item['expected_original_input_sha256'])
   target=out/'cases'/name;target.mkdir(parents=True)
   frozen={}
   for zone in item['expected_original_input_sha256']:
    shutil.copytree(case/zone,target/zone);frozen[zone]=freeze(target/zone)
   shutil.copyfile(case/'case.json',target/'case.json')
   frozen['case_sha' if study=='sealed_reuse' else 'case_sha256']=sha(target/'case.json')
   save(target/'input_freeze.json',frozen)
   rows.append({'case':name,'original_input_hashes_matched':count,'fresh_cache_metadata':fresh,'scientific_input_bytes_match':True})
 configs=out/'configs';configs.mkdir()
 for path in (REPO/'experiments/paper_runs'/study/'original_config').iterdir():shutil.copyfile(path,configs/path.name)
 save(configs/'cases.json',declared);shutil.copyfile(REPO/'provenance/model-files-9B.sha256',configs/'model-files-9B.sha256')
 files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'curator_only' not in p.relative_to(out).parts}
 save(out/'input_export.json',{'study':study,'kind':'OFFICIAL_UCI_LOCAL_REBUILD','files':files,'cases':rows,'old_private_reference_read':False,'old_observed_feature_values_read':False,'new_model_calls':0,'new_fits':0,'metadata_policy':'Case and cache provenance are freshly generated; original scientific input hashes are checked before use.'})
 return {'study':study,'cases':len(rows),'original_scientific_input_hashes_matched':sum(x['original_input_hashes_matched'] for x in rows),'fresh_cache_metadata_files':sum(len(x['fresh_cache_metadata']) for x in rows),'old_formal_reference_read':False,'model_calls':0,'fits':0}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',choices=STUDIES,required=True);parser.add_argument('--raw-dir',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();print(json.dumps(rebuild(args.study,args.raw_dir,args.output),ensure_ascii=False,indent=2))
