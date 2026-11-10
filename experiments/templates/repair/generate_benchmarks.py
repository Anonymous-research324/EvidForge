"""Seed1014 data generation; full references are written only to private.
Raw data are public UCI; methods receive dirty targets and authorized teaching.
Existing dataset raw source is reused by this GENERATOR, never read by a model.
"""
from pathlib import Path
import csv,io,json,hashlib,random,urllib.request,zipfile,collections,math,time
from decimal import Decimal
from common import ROOT,SEED,save,write_csv,digest,freeze
ARCHIVE=Path('@CURATOR_ARCHIVE@')
CATALOG={
 'bc_original':{'id':15,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data','columns':9,'rows':699,'sep':',','layout':'id_first'},
 'mammographic':{'id':161,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data','columns':5,'rows':961,'sep':',','layout':'last'},
 'ilpd':{'id':225,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20%28ILPD%29.csv','columns':10,'rows':583,'sep':',','layout':'ilpd'},
 'climate':{'id':252,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat','columns':18,'rows':540,'sep':None,'layout':'climate'},
 'voting':{'id':105,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data','columns':16,'rows':435,'sep':',','layout':'voting'},
 'qsar':{'id':254,'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv','columns':41,'rows':1055,'sep':';','layout':'last'},
 'monks1':{'columns':6,'rows':432,'archive':str(ARCHIVE/'DataForge_UnseenTable_v23/benchmarks/monks1/local_private/source.zip'),'member':'monks-1.test','layout':'monks','sep':None},
 'cmc':{'columns':9,'rows':1473,'archive':str(ARCHIVE/'DataForge_UnseenTable_v23/benchmarks/cmc/local_private/source.zip'),'member':'cmc.data','layout':'last','sep':','}}
def get_raw(name,spec,private):
 p=ROOT/'curator_cache/private'/name/'source.raw'
 provenance=json.loads((p.parent/'provenance.json').read_text())
 blob=p.read_bytes()
 assert hashlib.sha256(blob).hexdigest()==provenance['sha256'],'Cached source changed'
 (private/'source.raw').write_bytes(blob)
 save(private/'source_provenance.json',{**provenance,'origin':'same raw bytes as original benchmark; generator-only cache','acquired_at':time.time(),'generator_only':True,'original_reference_to_models':False})
 return blob.decode('utf-8-sig')
def parse(text,spec):
 rows=[];ys=[];ids=[]
 for n,line in enumerate(text.splitlines()):
  if not line.strip():continue
  a=line.split() if spec['sep'] is None else next(csv.reader([line],delimiter=spec['sep']))
  a=[x.strip() for x in a];mode=spec['layout'];rid=None
  if mode=='climate':
   if len(a)!=21 or a[-1] not in ['0','1']:continue
   rid=a[0]+':'+a[1];x=a[2:20];y=a[-1]
  elif mode=='monks':y=a[0];x=a[1:7];rid=a[7]
  elif mode=='id_first':rid=a[0];x=a[1:-1];y=a[-1]
  elif mode=='voting':y=a[0];x=[{'y':'1','n':'0','?':''}[v] for v in a[1:]]
  else:
   x=a[:-1];y=a[-1]
   if mode=='ilpd':x[1]={'male':'1','female':'0'}[x[1].casefold()]
  x=['' if v in ['?','NA','NaN','nan',''] else v for v in x]
  if len(x)!=spec['columns']:raise ValueError('Source width differs')
  for v in x:
   if v and not Decimal(v).is_finite():raise ValueError('Nonfinite source feature')
  rows.append(x);ys.append(y);ids.append(rid)
 if len(rows)!=spec['rows']:raise ValueError('Unexpected raw row count: '+str(len(rows)))
 return rows,ys,ids
def groups_for(rows,ids):
 parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 by_values={};by_id={}
 for i,x in enumerate(rows):
  signature=tuple(x)
  if signature in by_values:union(i,by_values[signature])
  else:by_values[signature]=i
  if ids[i] is not None:
   if ids[i] in by_id:union(i,by_id[ids[i]])
   else:by_id[ids[i]]=i
 groups=collections.defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 return {f'g{n:05d}':v for n,v in enumerate(sorted(groups.values(),key=lambda v:v[0]))}
def choose(pool,n,groups,ys,rng):
 if len(pool)<n:raise ValueError('Not enough eligible independent groups')
 buckets=collections.defaultdict(list)
 for g in pool:
  label=collections.Counter(ys[i] for i in groups[g]).most_common(1)[0][0];buckets[label].append(g)
 for a in buckets.values():rng.shuffle(a)
 exact={k:n*len(v)/len(pool) for k,v in buckets.items()};quota={k:int(v) for k,v in exact.items()}
 for k in sorted(buckets,key=lambda k:(-(exact[k]-quota[k]),k))[:n-sum(quota.values())]:quota[k]+=1
 selected=[g for k in sorted(buckets) for g in buckets[k][:quota[k]]];rng.shuffle(selected)
 return selected
def change(v,kind,rng):
 if kind=='blank':return '',kind
 if kind=='affix':return 'ERR'+v+'BAD',kind
 if kind=='scale' and Decimal(v)!=0:return format(Decimal(v)*100,'f'),kind
 indices=[i for i,c in enumerate(v) if c.isdigit()]
 if not indices:raise ValueError('No decimal digit')
 i=rng.choice(indices);d=rng.choice([x for x in '0123456789' if x!=v[i]])
 return v[:i]+d+v[i+1:],'digit'
def generate(name):
 spec=CATALOG[name];base=ROOT/'datasets'/name
 if base.exists():raise FileExistsError('Preserve existing dataset generation')
 private=base/'private';public=base/'public';private.mkdir(parents=True);public.mkdir()
 text=get_raw(name,spec,private);values,labels,original_ids=parse(text,spec)
 cols=[f'feature_{i+1:02d}' for i in range(spec['columns'])]
 groups=groups_for(values,original_ids);rng=random.Random(SEED)
 complete=[g for g,ix in groups.items() if all(all(values[i]) for i in ix)]
 n_learn=max(20,round(.1*len(groups)));n_val=max(10,round(.1*len(groups)))
 learning=choose(complete,n_learn,groups,labels,rng)
 complete=[g for g in complete if g not in learning]
 validation=choose(complete,n_val,groups,labels,rng)
 remaining=[g for g in groups if g not in set(learning+validation)]
 holdout=choose(remaining,max(1,round(.16*len(groups))),groups,labels,rng)
 repair=[g for g in remaining if g not in holdout]
 zone_groups={'learning':learning,'validation':validation,'repair':repair,'holdout':holdout}
 gid={i:g for g,ix in groups.items() for i in ix}
 clean=[{'entity_id':gid[i],**dict(zip(cols,x))} for i,x in enumerate(values)]
 dirty=[dict(r) for r in clean];events=[]
 for zone in ['learning','validation','repair']:
  row_ids=[i for g in zone_groups[zone] for i in groups[g]]
  for c in cols:
   eligible=[i for i in row_ids if clean[i][c]!=''];rng.shuffle(eligible)
   selected=eligible[:round(.7*len(eligible))]
   kinds=['blank','digit','scale','affix']*((len(selected)+3)//4);kinds=kinds[:len(selected)];rng.shuffle(kinds)
   for i,kind in zip(selected,kinds):
    value,actual=change(clean[i][c],kind,rng)
    if Decimal(value)==Decimal(clean[i][c]) if value and actual in ['digit','scale'] else value==clean[i][c]:raise ValueError('No-op contamination')
    dirty[i][c]=value;events.append({'zone':zone,'row':i,'field':c,'kind':actual})
 fields=['entity_id',*cols]
 schema={'id':name+'-seed1014','provenance':'UCI roles; generic reversible type encoding; no handwritten correction rules',
   'entity_fields':['entity_id'],'targets':cols,'protected_fields':[],'visible_context_fields':[],
   'fields':{'entity_id':{'role':'identity','type':'string','identity':True},**{c:{'role':'target','type':'number','meaning':'public source input feature'} for c in cols}}}
 context={'dataset':name,'source':'uci_benchmark_seed1014_'+name}
 bundle={}
 for zone in ['learning','validation']:
  bundle[zone]=[{'group':g,'provenance':'authorized synthetic teaching reference; not formal test',
     'records':[{'row_key':f'{name}-{i:05d}','dirty':dirty[i],'truth':{c:clean[i][c] for c in cols}} for i in groups[g]]} for g in zone_groups[zone]]
 save(public/'development_bundle.json',bundle);save(public/'learning_only.json',{'learning':bundle['learning']})
 save(public/'validation_only.json',{'validation':bundle['validation']});save(public/'declaration.json',schema);save(public/'context.json',context)
 for zone in ['learning','validation','repair','holdout']:
  ix=[i for g in zone_groups[zone] for i in groups[g]]
  write_csv(public/(zone+'_dirty.csv'),fields,[dirty[i] for i in ix])
  write_csv(private/(zone+'_reference.csv'),[*fields,'class'],[{**clean[i],'class':labels[i]} for i in ix])
 # Full references, masks and labels remain generator/scorer-only.
 save(private/'corruption_events.json',events)
 stats={z:{'groups':len(gs),'rows':sum(len(groups[g]) for g in gs)} for z,gs in zone_groups.items()}
 kinds=collections.Counter(e['kind'] for e in events)
 save(public/'dataset_metadata.json',{'name':name,'seed':SEED,'source_id':spec.get('id'),'rows':len(clean),'targets':cols,'zones':stats,
   'contamination_counts':dict(kinds),'protected_class_removed_from_repair_input':True,'natural_unknown_cells':sum(v=='' for x in values for v in x),
   'teaching_complete_case_groups_only':True,'all_original_rows_preserved':True,'cold_dataset_specific_library':True,
   'encoded_categorical_inputs':{'voting':{'y':'1','n':'0'},'ilpd':{'Male':'1','Female':'0'}}.get(name)})
 save(base/'input_freeze.json',{'public':freeze(public),'private':freeze(private),'seed':SEED,'roles_isolated':True,'code_sha256':digest(__file__)})
 print(json.dumps({'dataset':name,'rows':len(clean),'zones':stats,'corrupted_cells':len(events),'formal_answers_to_models':False}),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
 for name in a.datasets:generate(name)
