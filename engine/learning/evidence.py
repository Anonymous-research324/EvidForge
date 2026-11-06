"""Build observations-only inference features; source answers stay in fitting.

No query truth argument exists. Each original alternative and KEEP participates.
Protected row identifiers/class labels never enter candidate arithmetic.
"""
from collections import defaultdict
import math
from .environment import RUNTIME
from .observation_model import fit as fit_observation,rank_row,channels
from .arithmetic import FIELDS, CONTEXT_FIELDS
from unseen_table.common import equal,number
from skill_runtime.codec import fit as fit_codec


def build(rows,source_records,base,cells,baseline):
    codec=fit_codec(source_records,base['declaration']);blocks=defaultdict(list)
    for c in cells:blocks[c['row']].append(c)
    models={mode:fit_observation(source_records,base,mode) for mode in ['raw','signed_log']}
    noisy={mode:{} for mode in models}
    for i,row in enumerate(rows):
        for mode,model in models.items():
            for q in rank_row(row,blocks[i],model):noisy[mode][(i,q['column'])]=q
    lookup={(d['row'],d['field']):d for d in baseline};frames=[]
    for c in cells:
        i,col=c['row'],c['column'];d=lookup[(i,col)];old=rows[i][col];options=[]
        for q in c['options']:
            matched=channels(old,q['value'],col,codec)
            f={'rank_score':float(q['score']),'is_keep':float(equal(old,q['value'])),
               'is_baseline':float(equal(d['after'],q['value'])),'baseline_changed':float(not equal(old,d['after'])),
               'forward_explainable':float(any(x!='unknown' for x in matched)),
               **{f'is_{k}':float(k in q['kinds']) for k in ['affix','scale','shift','edit']},
               'old_numeric':float(number(old) is not None),'old_blank':float(not old.strip()),'score_gap':0.}
            for mode,label in [('raw','raw'),('signed_log','signed')]:
                e=noisy[mode].get((i,col))
                answer=next((x for x in e['options'] if equal(x['value'],q['value'])),None) if e else None
                f[f'noisy_{label}_log_score']=max(-30,min(30,answer['log_score'])) if answer else -30
                f[f'noisy_{label}_context_fields']=float(e['context_fields']) if e else 0
                if e and answer:
                    best=next(x['log_score'] for x in e['options'] if equal(x['value'],e['value']))
                    odds=answer['log_score']-best+e['unknown_log_odds']
                else:odds=-30
                f[f'noisy_{label}_unknown_odds']=max(-30,min(30,odds))
            f.update({name:0.0 for name in CONTEXT_FIELDS})
            if set(f)!=FIELDS:raise AssertionError('Finite feature interface changed')
            options.append({'value':q['value'],'features':f})
        frames.append({'row':i,'column':col,'original':old,'baseline_after_local':d['after'],
                       'group':f'query-{i}','options':options})
    return frames


def labelled(rows,source_records,base,cells,baseline,query_source_records):
    """Only authorized teaching callers attach labels AFTER inference features."""
    fs=build(rows,source_records,base,cells,baseline)
    for f in fs:
        r=query_source_records[f['row']];f['group']=r['group'];f['expected_local_only']=r['truth'][f['column']]
    return fs
