"""Rebuild row hypotheses from verified repairs, preserving original anchors.

Use only nonquery fields as denominator evidence. Trusted values remain
proposals with source-calibrated provenance; no original test value is read.
This produces exact-pool candidates and cannot recover an arbitrary blank.
"""
import numpy as np
from unseen_table.common import number,canonical,equal
from .row_family import matches,QS,TOL,nonpoint
from .codec import possible


def hypotheses(original,cells,confirmed,codec,family):
    if not family['support_rows']:return {}
    seeds={c:v for c,v in confirmed.items() if c not in family['exceptions'] and nonpoint(v)}
    masks={c:matches([float(number(v))])[0] for c,v in seeds.items()}
    total=np.sum(list(masks.values()),axis=0) if masks else np.zeros(len(QS),dtype=int)
    result={}
    for cell in cells:
        col=cell['column']
        if col in family['exceptions']:continue
        counts=total-masks.get(col,0);other={c:v for c,v in seeds.items() if c!=col}
        if len(other)<4:continue
        peak=int(counts.max());qi=int(np.flatnonzero(counts==peak)[0]);q=int(QS[qi])
        support=[v for v in other.values() if abs(float(number(v))-round(float(number(v))*q)/q)<=TOL]
        distinct=len({canonical(v) for v in support});fraction=peak/len(other)
        candidates={canonical(p['value']):p['value'] for p in cell['options']
            if number(p['value']) is not None and possible(original[col],p['value'],col,codec)
            and abs(float(number(p['value']))-round(float(number(p['value']))*q)/q)<=TOL}
        if len(candidates)!=1:continue
        value=next(iter(candidates.values()))
        if equal(original[col],value):continue
        offered={}
        for minimum in [4,6,8,12]:
            if peak>=minimum and distinct>=minimum and fraction>=.9:
                offered['verified_row_grid_'+str(minimum)]={'value':value,'denominator':q,
                    'trusted_support':peak,'trusted_fraction':fraction,'distinct_values':distinct,
                    'query_field_excluded':True,'original_value_is_evidence_anchor':True}
        if offered:result[col]=offered
    return result
