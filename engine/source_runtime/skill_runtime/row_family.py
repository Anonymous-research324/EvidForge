"""Discover within-row rounded rational patterns; exclude queried field."""
from collections import Counter
import numpy as np
from unseen_table.common import number,canonical
from unseen_table.learn_transforms import apply_rule
from .codec import possible
QS=np.arange(2,513,dtype=float)
TOL=.00000501

def nonpoint(v):
    n=number(v);return n is not None and n!=n.to_integral()
def matches(v):
    a=np.asarray(v,dtype=float);return np.abs(a[:,None]-np.rint(a[:,None]*QS[None,:])/QS[None,:])<=TOL

def fit(records,schema):
    hits=Counter();misses=Counter();support_rows=0
    for r in records:
        values={c:v for c,v in r['truth'].items() if c in schema['targets'] and nonpoint(v)}
        if len(values)<5 or len(set(canonical(v) for v in values.values()))<5:continue
        m=matches([float(number(v)) for v in values.values()]);count=m.sum(axis=0);eligible=np.flatnonzero((count>=5)&(count>=.8*len(values)))
        if not len(eligible):continue
        best=int(max(count[eligible]));qi=int(next(k for k in eligible if count[k]>=best-1));support_rows+=1
        for col,hit in zip(values,m[:,qi]):(hits if hit else misses)[col]+=1
    exceptions=[c for c in schema['targets'] if misses[c]>=3 and misses[c]/(misses[c]+hits[c])>=.5]
    return {'family':'row_rational_rounding','max_denominator':512,'tolerance':TOL,'exceptions':exceptions,'support_rows':support_rows,'exception_support':{c:{'hits':hits[c],'misses':misses[c]} for c in schema['targets']}}

def basic(row,col,codec):
    values=[row[col]]
    for rule in codec['rules']:
        if col in rule['fields']:
            v=apply_rule(rule,row,col)
            if v is not None and possible(row[col],v,col,codec):values.append(v)
    return list({canonical(v):v for v in values if nonpoint(v) and codec['low'] is not None and codec['low']<=number(v)<=codec['high']}.values())

def prove(row,columns,cells,codec,family):
    values={c:basic(row,c,codec) for c in columns if c not in family['exceptions']}
    masks={c:matches([float(number(v)) for v in vs]).any(axis=0) for c,vs in values.items() if vs};total=np.sum(list(masks.values()),axis=0) if masks else np.zeros(len(QS),dtype=int);result={}
    for c in cells:
        col=c['column'];proof={'value':None,'support':0,'fraction':0,'distinct':0,'query_field_excluded':True}
        if col in family['exceptions'] or not family['support_rows']:result[col]=proof;continue
        count=total-masks.get(col,0);peak=int(max(count));qi=int(np.flatnonzero(count>=peak)[0]);q=int(QS[qi]);contributors=[v for k,vs in values.items() if k!=col for v in vs if abs(float(number(v))-round(float(number(v))*q)/q)<=TOL]
        allowed={canonical(p['value']):p['value'] for p in c['options'] if possible(row[col],p['value'],col,codec) and number(p['value']) is not None and abs(float(number(p['value']))-round(float(number(p['value']))*q)/q)<=TOL}
        proof.update(value=next(iter(allowed.values())) if len(allowed)==1 else None,support=peak,fraction=peak/max(1,len(masks)-(col in masks)),distinct=len(set(canonical(v) for v in contributors)),denominator=q,matching_candidates=len(allowed));result[col]=proof
    return result
