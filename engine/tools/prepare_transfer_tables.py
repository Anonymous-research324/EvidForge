"""Create NEW benchmarks only after an algorithm freeze, never look up old tests.

Raw source values and evaluation references stay local_private. Learners see
only public20% labels and dirty repair input. No reference values are printed.
Public corruption specification is fixed, not inferred from private test gold.
"""
from pathlib import Path
from decimal import Decimal
from collections import Counter
from io import BytesIO
from zipfile import ZipFile
import argparse,random,urllib.request,ssl,sys,json
import certifi
from sklearn.model_selection import train_test_split
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import fresh,load,dump,sha256,write_table,fingerprint,validate_declaration,normalize_bundle,canonical

SPECS={
 'cmc':{'url':'https://archive.ics.uci.edu/static/public/30/contraceptive%2Bmethod%2Bchoice.zip','member':'cmc.data','rows':1473,'features':9,'delimiter':',','uci':'https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice'},
 'monks1':{'url':'https://archive.ics.uci.edu/static/public/70/monk%2Bs%2Bproblems.zip','member':'monks-1.test','rows':432,'features':6,'delimiter':None,'label_first':True,'drop_original_id':True,'uci':'https://archive.ics.uci.edu/dataset/70/monk+s+problems'},
}


def damage(rows,targets,rng):
    dirty=[dict(r) for r in rows];total=round(.7*len(rows)*len(targets));quotient,remainder=divmod(total,len(targets))
    chosen=[];field_counts={}
    for j,c in enumerate(targets):
        ids=list(range(len(rows)));rng.shuffle(ids);count=quotient+int(j<remainder);field_counts[c]=count
        chosen.extend((i,c) for i in ids[:count])
    rng.shuffle(chosen);kinds=['blank','digit','scale','affix'];assigned=[kinds[i%4] for i in range(len(chosen))]
    for j,((i,c),kind) in enumerate(zip(chosen,assigned)):
        if kind=='scale' and Decimal(rows[i][c])==0:
            replacement=next(k for k,((r,f),t) in enumerate(zip(chosen,assigned)) if t!='scale' and Decimal(rows[r][f])!=0)
            assigned[j],assigned[replacement]=assigned[replacement],assigned[j]
    changes=[]
    for (i,c),kind in zip(chosen,assigned):
        old=rows[i][c]
        if kind=='blank':new=''
        elif kind=='scale':new=str(Decimal(old)*10)
        elif kind=='affix':new='ERR:'+old
        else:
            positions=[j for j,x in enumerate(old) if x.isdigit()];pos=rng.choice(positions)
            digit=rng.choice([str(n) for n in range(10) if str(n)!=old[pos]]);new=old[:pos]+digit+old[pos+1:]
        if new and not new.startswith('ERR:') and Decimal(new)==Decimal(old):raise ValueError('No-op corruption forbidden')
        dirty[i][c]=new;changes.append({'row':i,'field':c,'kind':kind})
    assert max(field_counts.values())-min(field_counts.values())<=1
    return dirty,{'target_cells':len(rows)*len(targets),'errors':len(changes),'fields':field_counts,'types':dict(Counter(assigned))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(SPECS),required=True);p.add_argument('--out',required=True);p.add_argument('--algorithm-freeze',required=True);a=p.parse_args()
    fixed=load(a.algorithm_freeze)
    if any(sha256(n)!=v for n,v in fixed['files'].items()):raise ValueError('Algorithm changed before fresh source download')
    spec=SPECS[a.dataset];root=fresh(a.out);private=fresh(root/'local_private');public=fresh(root/'public_inputs')
    dump(root/'download_preregistration.json',{'algorithm_freeze_sha256':sha256(a.algorithm_freeze),'source':spec,
        'seed':20260927,'learning_fraction':.1,'validation_fraction':.1,'validation_not_for_refit':True,'repair_fraction_of_remaining':.8,'corruption_fraction':.7,
        'protected_fields':['id','class'],'independent_new_dataset_before_any_repair_scoring':True,
        'reference_values_visible_to_model':False,'source_not_for_looking_up_any_old_test_answer':True})
    context=ssl.create_default_context(cafile=certifi.where())
    request=urllib.request.Request(spec['url'],headers={'User-Agent':'DataForge research benchmark preparation'})
    with urllib.request.urlopen(request,context=context,timeout=60) as response:archive=response.read(2000000)
    (private/'source.zip').write_bytes(archive)
    with ZipFile(BytesIO(archive)) as z:
        members=sorted(n for n in z.namelist() if n.lower().endswith(spec['members_suffix'])) if 'members_suffix' in spec else [next(n for n in z.namelist() if Path(n).name==spec['member'])]
        raw=b'\n'.join(z.read(n) for n in members)
    lines=[line.strip() for line in raw.decode().splitlines() if line.strip()]
    if spec.get('skip_header'):lines=lines[1:]
    cols=[f'feature_{i+1:02d}' for i in range(spec['features'])]
    fields=['id']+cols+['class'];rows=[]
    for i,line in enumerate(lines):
        values=line.split(spec['delimiter']) if spec['delimiter'] else line.split()
        if spec.get('drop_original_id'):values=values[:-1]
        if spec.get('label_first'):values=values[1:]+values[:1]
        if len(values)!=len(cols)+1:raise ValueError('Unexpected source schema')
        for value in values[:-1]:
            if not Decimal(value).is_finite():raise ValueError('Non-finite source value')
        rows.append(dict(zip(fields,[f'{a.dataset}-{i:04d}']+values)))
    if len(rows) not in spec.get('rows_allowed',[spec.get('rows')]):raise ValueError('Source sample count changed')
    # Exact duplicate observations must never be separated across teaching,
    # repair and holdout. Keep every source row; do not silently drop duplicates.
    grouped={}
    for row in rows:
        key=fingerprint([canonical(row[c]) for c in cols]);grouped.setdefault(key,[]).append(row)
    ids=sorted(grouped);y={k:grouped[k][0]['class'] for k in ids}
    entity_names={k:f'entity-{i:04d}' for i,k in enumerate(ids)}
    teach,rest=train_test_split(ids,train_size=round(.2*len(ids)),stratify=[y[k] for k in ids],random_state=20260927)
    learn,check=train_test_split(teach,test_size=.5,stratify=[y[k] for k in teach],random_state=20260927)
    evaluation,holdout=train_test_split(rest,train_size=.8,stratify=[y[k] for k in rest],random_state=20260927)
    partitions={n:[dict(r) for k in partition for r in grouped[k]] for n,partition in [('learning',learn),('validation',check),('evaluation',evaluation),('holdout',holdout)]}
    ownership={r['id']:k for k,rr in grouped.items() for r in rr}
    rng=random.Random(20260927)
    # The declaration carries generic roles, not a hand-written scientific rule.
    declaration={'id':a.dataset+'-numeric','provenance':'UCI numeric field roles confirmed for fresh benchmark; no hand-written scientific relation',
        'entity_fields':['id'],'targets':cols,'protected_fields':['class'],
        'visible_context_fields':[],'fields':{f:{'role':'entity' if f=='id' else 'protected' if f=='class' else 'target',
            'type':'string' if f in ['id','class'] else 'number','meaning':f} for f in fields}}
    labels={};counts={}
    for name in ['learning','validation']:
        corrupted,count=damage(partitions[name],cols,rng);counts[name]=count
        label_groups={}
        for d,r in zip(corrupted,partitions[name]):
            key=ownership[r['id']];original_id=r['id'];d=dict(d);d['id']=entity_names[key]
            label_groups.setdefault(key,[]).append({'row_key':original_id,'dirty':d,'truth':{c:r[c] for c in cols}})
        labels[name]=[{'group':entity_names[key],'provenance':'Preselected first10% learning or second10% independent validation; no refit after validation; formal references isolated',
                       'records':rr} for key,rr in label_groups.items()]
    eval_dirty,counts['evaluation']=damage(partitions['evaluation'],cols,rng)
    for rr in [eval_dirty,partitions['evaluation'],partitions['holdout']]:
        for r in rr:r['id']=entity_names[ownership[r['id']]]
    validate_declaration(declaration,fields);normalize_bundle(labels,declaration)
    write_table(public/'repair_evaluation.csv',fields,eval_dirty)
    write_table(private/'repair_evaluation_reference.csv',fields,partitions['evaluation'])
    write_table(private/'downstream_holdout_reference.csv',fields,partitions['holdout'])
    dump(public/'development_labels.json',labels);dump(public/'declaration.json',declaration)
    dump(public/'context.json',{'source_id':a.dataset,'task':'numeric cell repair','partition':'fresh benchmark'})
    dump(root/'benchmark_manifest.json',{'source_uci':spec['uci'],'license':'CC BY4.0','source_archive_sha256':sha256(private/'source.zip'),
        'seed':20260927,'counts':{n:len(v) for n,v in partitions.items()},'corruption':counts,
        'independent_observation_groups':len(grouped),'duplicate_rows_preserved':len(rows)-len(grouped),
        'learning_fraction_by_groups':len(teach)/len(grouped),'learning_fraction_by_rows':(len(partitions['learning'])+len(partitions['validation']))/len(rows),
        'schema':{'target_features':len(cols),'protected':['id','class']},'public_files':{str(q.relative_to(root)):sha256(q) for q in public.iterdir()},
        'private_reference_not_visible_to_model':True,'classes_not_used_for_repair_rules':True,
        'first_local_raw_download_after_algorithm_freeze':True,'metadata_only_before_freeze':True,'original_paper_train_test_split_used':False,'sampling_note':'All rows from declared member re-split by observation groups; MONKS1 uses its complete432-row domain file, not the original paper split'})
    print('NEW_BENCHMARK_CREATED',a.dataset,{n:len(v) for n,v in partitions.items()},'features',len(cols),'corruption counts',counts,flush=True)

if __name__=='__main__':main()
