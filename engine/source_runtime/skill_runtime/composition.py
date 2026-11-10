"""Compare finite compositions on public source-held proposals only.

Existing proposals are deterministic measurement replays. Choosing a recipe
here reuses public development cases; it is not an independent assessment.
The model receives aggregate scores and operator IDs, never individual labels.
"""
from itertools import combinations
from pathlib import Path
from unseen_table.common import load,equal,canonical,fingerprint
from .learning_tools import measurements

ROOT=Path(__file__).resolve().parents[1]


def compare(dataset,maximum=4):
    evidence=measurements(dataset);eligible={k for k,v in evidence.items() if v['source_cross_check_passed']}
    source=ROOT/'research'/(dataset+'_public_audit')/'source_held_cache.json';rows=[];baseline={}
    for fold in load(source):
        offset=len(rows);rows+=fold['checking_records']
        for d in fold['decisions']:baseline[(offset+d['row'],d['field'])]=d
    offers={k:{} for k in eligible}
    for suffix in ['refinement_public','relations_public','conditional_public','uncertain_context_public_v2','mixture_public']:
        path=ROOT/'research'/(dataset+'_'+suffix)/'proposals_public_only.json'
        if not path.is_file():continue
        raw=load(path)
        pairs=((q['clause'],q) for q in raw) if isinstance(raw,list) else ((name,q) for name,qs in raw.items() for q in qs)
        for name,q in pairs:
            if name in eligible:offers[name][(q['row'],q['field'])]=q['after']
    errors=normal=old_changes=old_correct=old_damage=0
    for (i,col),d in baseline.items():
        error=not equal(rows[i]['dirty'][col],rows[i]['truth'][col]);changed=not equal(d['before'],d['after'])
        errors+=error;normal+=not error;old_changes+=changed;old_correct+=error and changed and equal(d['after'],rows[i]['truth'][col]);old_damage+=not error and changed
    candidates=[]
    for size in range(1,min(maximum,len(eligible))+1):
        for operators in combinations(sorted(eligible),size):
            merged={}
            for op in operators:
                for key,value in offers[op].items():merged.setdefault(key,{})[canonical(value)]=value
            changed=correct=damage=0
            for (i,col),values in merged.items():
                if len(values)!=1:continue
                value=next(iter(values.values()));old=baseline[(i,col)]
                if not equal(old['before'],old['after']) or equal(value,old['before']):continue
                error=not equal(rows[i]['dirty'][col],rows[i]['truth'][col]);changed+=1
                correct+=error and equal(value,rows[i]['truth'][col]);damage+=not error
            if not changed or correct/changed<.98 or damage:continue
            p=(old_correct+correct)/(old_changes+changed);r=(old_correct+correct)/errors
            candidates.append({'recipe_id':fingerprint(list(operators))[:16],'operators':list(operators),
                'additional_proposals':changed,'additional_exact_repairs':correct,'additional_precision':correct/changed,
                'additional_normal_damage':damage,'repair_precision':p,'repair_recall':r,'repair_F1':2*p*r/(p+r),
                'normal_damage_fraction':old_damage/normal,'operator_count':size})
    candidates.sort(key=lambda q:(-q['repair_F1'],q['operator_count'],q['operators']))
    # Distinct residual coverage levels, rather than three copies of the same
    # recipe differing only in a redundant stricter threshold.
    recommendations=[];seen=set()
    for q in candidates:
        marker=(q['additional_exact_repairs'],q['operator_count'])
        if marker not in seen:recommendations.append(q);seen.add(marker)
        if len(recommendations)==3:break
    return {'recipes':recommendations,'measured_operators':list(evidence.values()),
            'scope':'reused public development composition selection; not independent performance',
            'normal_damage_target':.005,'additional_precision_target':.98,
            'raw_reference_cases_visible':False,'formal_reference_read':False,
            'baseline_errors':errors,'baseline_exact_repairs':old_correct,'finite_compositions_checked':len(candidates)}
