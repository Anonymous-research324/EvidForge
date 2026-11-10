"""Frozen-before-download benchmark curator. No repair method imports this module."""
from pathlib import Path
from decimal import Decimal
import csv,io,zipfile,urllib.request,time,random,collections,shlex,hashlib,json,math
from common import ROOT,SEED,save,write_csv,digest,freeze
from catalog import CATALOG,NAMES,CAR_VOCAB,TIC_VOCAB
def num(x):
 try:
  d=Decimal(str(x));return d if d.is_finite() else None
 except Exception:return None
def canonical(x):return format(num(x).normalize(),'f') if num(x) is not None else x
def raw_archive(name,spec,private):
 f=private/'source.zip'
 cache=ROOT/'generator_inputs/private'/name/'source.zip'
 info=__import__('json').loads((cache.parent/'provenance.json').read_text())
 blob=cache.read_bytes()
 if hashlib.sha256(blob).hexdigest()!=info['sha256']:raise ValueError('Curator raw cache fingerprint mismatch')
 if f.exists():raise FileExistsError('Do not overwrite an existing source')
 f.write_bytes(blob)
 save(private/'source_provenance.json',{**info,'reused_raw_archive_only':True,'cached_at':time.time(),'old_formal_answers_used':False})
 return zipfile.ZipFile(f)

def member(z,basename):
 names=[n for n in z.namelist() if Path(n).name.casefold()==basename.casefold()]
 if len(names)!=1:raise ValueError('Missing/ambiguous official archive member '+basename)
 return z.read(names[0])
def parse(z,spec):
 mode=spec['layout'];values=[];ys=[];ids=[];sources=[]
 if mode in ['estate','concrete']:
  import pandas as pd
  if mode=='estate':
   candidates=[n for n in z.namelist() if n.lower().endswith('.xlsx') and not Path(n).name.startswith('~')]
   if len(candidates)!=1:raise ValueError('Expected unique estate workbook')
   df=pd.read_excel(io.BytesIO(z.read(candidates[0])),engine='openpyxl');a=df.iloc[:,1:7];y=df.iloc[:,7];ident=df.iloc[:,0]
  else:
   df=pd.read_excel(io.BytesIO(member(z,'Concrete_Data.xls')),engine='xlrd');a=df.iloc[:,:8];y=df.iloc[:,8];ident=[None]*len(df)
  for row,truth,identity in zip(a.itertuples(index=False,name=None),y,ident):
   values.append(['' if pd.isna(v) else str(v) for v in row]);ys.append(str(truth));ids.append(None if identity is None else str(identity));sources.append('official_workbook')
 else:
  if mode=='spectf':
   texts=[('original_train',member(z,'SPECTF.train').decode('utf-8-sig')),('original_test',member(z,'SPECTF.test').decode('utf-8-sig'))]
  elif mode=='retino':texts=[('official_arff',member(z,'messidor_features.arff').decode('utf-8-sig'))]
  else:
   name={'car':'car.data','tic':'tic-tac-toe.data','auto':'auto-mpg.data','airfoil':'airfoil_self_noise.dat'}[mode]
   texts=[('official_data',member(z,name).decode('utf-8-sig'))]
  for source,text in texts:
   in_data=mode!='retino'
   for line in text.splitlines():
    line=line.strip()
    if not line or line.startswith('%'):continue
    if not in_data:
     if line.casefold().startswith('@data'):in_data=True
     continue
    identity=None
    if mode=='auto':
     a=shlex.split(line);truth=a[0];row=a[1:8];identity=' '.join(a[8:])
    elif mode=='airfoil':
     a=line.split();row=a[:-1];truth=a[-1]
    else:
     a=[v.strip() for v in next(csv.reader([line]))]
     if mode=='spectf':truth=a[0];row=a[1:]
     else:row=a[:-1];truth=a[-1]
    values.append(['' if v in ['?','NA','NaN','nan',''] else v for v in row]);ys.append(truth);ids.append(identity);sources.append(source)
 if len(values)!=spec['rows'] or any(len(r)!=spec['columns'] for r in values):raise ValueError('Official source shape mismatch')
 vocab=CAR_VOCAB if mode=='car' else TIC_VOCAB if mode=='tic' else None
 if vocab:
  values=[[str(vocab[j].index(v)) for j,v in enumerate(r)] for r in values]
 if any(v and num(v) is None for row in values for v in row):raise ValueError('Unexpected nonnumeric value after registered reversible type encoding')
 if spec['task']=='regression' and any(num(v) is None for v in ys):raise ValueError('Unknown regression target')
 return values,ys,ids,sources,vocab
def make_groups(values,ids):
 parent=list(range(len(values)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 seen={};identities={}
 for i,row in enumerate(values):
  key=tuple(map(canonical,row))
  if key in seen:union(i,seen[key])
  else:seen[key]=i
  if ids[i] is not None:
   if ids[i] in identities:union(i,identities[ids[i]])
   else:identities[ids[i]]=i
 grouped=collections.defaultdict(list)
 for i in range(len(values)):grouped[find(i)].append(i)
 return {f'g{k:05d}':ix for k,ix in enumerate(sorted(grouped.values(),key=lambda x:x[0]))}
def choose(pool,n,groups,strata,rng):
 if len(pool)<n:raise ValueError('Insufficient independent complete-case teaching groups')
 buckets=collections.defaultdict(list)
 for g in pool:buckets[collections.Counter(strata[i] for i in groups[g]).most_common(1)[0][0]].append(g)
 for a in buckets.values():rng.shuffle(a)
 exact={k:n*len(v)/len(pool) for k,v in buckets.items()};quota={k:int(v) for k,v in exact.items()}
 for k in sorted(buckets,key=lambda k:(-(exact[k]-quota[k]),k))[:n-sum(quota.values())]:quota[k]+=1
 result=[g for k in sorted(buckets) for g in buckets[k][:quota[k]]];rng.shuffle(result);return result
def damage(value,kind,rng,domain=None):
 if kind=='blank':return '',kind
 if kind=='affix':return 'ERR'+value+'BAD',kind
 if domain is not None:
  if kind=='digit':return rng.choice([x for x in domain if num(x)!=num(value)]),'category_substitution'
  return format(num(value)+100,'f'),'category_invalid_code'
 if kind=='scale' and num(value)!=0:return format(num(value)*100,'f'),kind
 indices=[i for i,c in enumerate(value) if c.isdigit()]
 if not indices:raise ValueError('No digit for registered numeric corruption')
 i=rng.choice(indices);d=rng.choice([v for v in '0123456789' if v!=value[i]])
 return value[:i]+d+value[i+1:],'digit'
def generate(name):
 spec=CATALOG[name];base=ROOT/'datasets'/name
 if (base/'input_freeze.json').exists():print('GENERATION_ALREADY_FROZEN',name,flush=True);return
 private=base/'private';public=base/'public';private.mkdir(parents=True,exist_ok=True);public.mkdir(exist_ok=True)
 with raw_archive(name,spec,private) as z:values,ys,ids,source_splits,vocab=parse(z,spec)
 columns=[f'feature_{i+1:02d}' for i in range(spec['columns'])]
 groups=make_groups(values,ids);rng=random.Random(SEED);ng=len(groups)
 if spec['task']=='regression':
  import numpy as np
  cuts=np.unique(np.quantile([float(x) for x in ys],[i/10 for i in range(1,10)]))
  strata=[str(int(np.searchsorted(cuts,float(x),side='right'))) for x in ys]
 else:strata=ys
 complete=[g for g,ix in groups.items() if all(all(values[i]) for i in ix)]
 learning=choose(complete,max(20,round(.1*ng)),groups,strata,rng)
 validation=choose([g for g in complete if g not in learning],max(10,round(.1*ng)),groups,strata,rng)
 remaining=[g for g in groups if g not in set(learning+validation)]
 holdout=choose(remaining,max(1,round(.16*ng)),groups,strata,rng);repair=[g for g in remaining if g not in holdout]
 zones=dict(learning=learning,validation=validation,repair=repair,holdout=holdout)
 group_of={i:g for g,ix in groups.items() for i in ix}
 clean=[{'entity_id':group_of[i],**dict(zip(columns,row))} for i,row in enumerate(values)]
 dirty=[dict(r) for r in clean];events=[]
 for zone in ['learning','validation','repair']:
  ix=[i for g in zones[zone] for i in groups[g]]
  for j,c in enumerate(columns):
   eligible=[i for i in ix if clean[i][c]!=''];rng.shuffle(eligible);chosen=eligible[:round(.7*len(eligible))]
   kinds=(['blank','digit','scale','affix']*((len(chosen)+3)//4))[:len(chosen)];rng.shuffle(kinds)
   domain=[str(i) for i in range(len(vocab[j]))] if vocab else None
   for i,k in zip(chosen,kinds):
    after,actual=damage(clean[i][c],k,rng,domain)
    if canonical(after)==canonical(clean[i][c]):raise ValueError('No-op contamination')
    dirty[i][c]=after;events.append(dict(zone=zone,row=i,field=c,kind=actual))
 fields=['entity_id',*columns]
 declaration={'id':name+'-seed1014','provenance':'UCI input roles; reversible public schema encoding; no handwritten correction rules','entity_fields':['entity_id'],'targets':columns,'protected_fields':[],'visible_context_fields':[],
  'fields':{'entity_id':{'role':'identity','type':'string','identity':True},**{c:{'role':'target','type':'number','meaning':'anonymous source input feature; numeric/reversibly encoded categorical'} for c in columns}}}
 context={'dataset':name,'source':'uci_downstream_seed1014_'+name}
 bundle={zone:[{'group':g,'provenance':'authorized synthetic teaching reference; not formal test','records':[{'row_key':f'{name}-{i:05d}','dirty':dirty[i],'truth':{c:clean[i][c] for c in columns}} for i in groups[g]]} for g in zones[zone]] for zone in ['learning','validation']}
 for filename,obj in [('development_bundle.json',bundle),('learning_only.json',{'learning':bundle['learning']}),('validation_only.json',{'validation':bundle['validation']}),('declaration.json',declaration),('context.json',context)]:save(public/filename,obj)
 for zone in ['learning','validation','repair','holdout']:
  ix=[i for g in zones[zone] for i in groups[g]]
  if zone!='holdout':write_csv(public/(zone+'_dirty.csv'),fields,[dirty[i] for i in ix])
  write_csv(private/(zone+'_reference.csv'),[*fields,'class'],[{**clean[i],'class':ys[i]} for i in ix])
 repair_ix=[i for g in repair for i in groups[g]]
 write_csv(base/'downstream_training/labels.csv',['entity_id','class'],[{'entity_id':group_of[i],'class':ys[i]} for i in repair_ix])
 save(private/'corruption_events.json',events);save(private/'original_source_split.json',source_splits)
 categorical=columns if vocab else [columns[6]] if name=='auto_mpg' else []
 metadata={'name':name,'task':spec['task'],'seed':SEED,'source_id':spec['id'],'rows':len(values),'targets':columns,'categorical_columns':categorical,
  'zones':{z:{'groups':len(gs),'rows':sum(len(groups[g]) for g in gs)} for z,gs in zones.items()},'contamination_counts':dict(collections.Counter(e['kind'] for e in events)),
  'reversible_category_vocabulary':dict(zip(columns,vocab)) if vocab else {},'natural_unknown_cells':sum(v=='' for row in values for v in row),
  'no_predictive_target_or_id_in_repair_model':True,'no_holdout_public_file':True,'teaching_complete_case_groups_only':True,'all_original_rows_preserved':True,
  'original_train_test_files_repartitioned':name=='spectf','source_library_cold_start':True,'grouping':'public entity/car-name identifier if supplied, plus exact canonical feature duplicates; latent correlation without an ID is not guaranteed separated'}
 save(public/'dataset_metadata.json',metadata)
 save(base/'input_freeze.json',{'public':freeze(public),'private':freeze(private),'supervised_labels':freeze(base/'downstream_training'),'generator_sha256':digest(__file__),'seed':SEED})
 print('NEW_DATASET_FROZEN',name,json.dumps(metadata['zones']),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--datasets',nargs='+',default=NAMES);args=a.parse_args()
 for n in args.datasets:generate(n)
