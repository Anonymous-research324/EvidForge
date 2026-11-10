"""Review the weak base digit-edit channel before admitting new rules.

Only source-held teaching predictions select gates. Other base channels are
preserved. IDs/labels and inferred replacement values are never inspected.
"""
from .environment import RUNTIME
from unseen_table.common import equal

GRIDS={'rank_score':[0,.5,.7,.8,.9,.95,.97,.99,.995,.999],
       'raw_gap':[-60,-4,-2,0,1,2,4,8,12],
       'signed_gap':[-60,-4,-2,0,1,2,4,8,12]}


def point(frame):
    q=next((q for q in frame['options'] if equal(q['value'],frame['baseline_after_local'])),None)
    keep=next(q for q in frame['options'] if q['features']['is_keep'])
    if q is None or equal(frame['original'],frame['baseline_after_local']):return None
    f=q['features']
    if not (f['old_numeric'] and f['is_edit'] and not (f['is_affix'] or f['is_scale'] or f['is_shift'])):return None
    return {'rank_score':f['rank_score'],
            'raw_gap':f['noisy_raw_log_score']-keep['features']['noisy_raw_log_score'],
            'signed_gap':f['noisy_signed_log_score']-keep['features']['noisy_signed_log_score']}


def fit(frames):
    proposals=[(f,point(f)) for f in frames];proposals=[(f,q) for f,q in proposals if q is not None]
    best=None
    for name,grid in GRIDS.items():
        for threshold in grid:
            selected=[f for f,q in proposals if q[name]>=threshold]
            groups={f['group'] for f in selected}
            if len(selected)<8 or len(groups)<4:continue
            good=sum(equal(f['baseline_after_local'],f['expected_local_only']) for f in selected)
            damage=sum(equal(f['original'],f['expected_local_only']) for f in selected)
            if good/len(selected)<.98 or damage:continue
            key=(good,threshold)
            if best is None or key>best[0]:best=(key,{'feature':name,'threshold':threshold,'checked':len(selected),
                'source_groups':len(groups),'P':good/len(selected)})
    return {'digit_only_gate':best[1] if best else None,'source_digit_changes':len(proposals),
            'scope':'inner source-only gate, pending outer check','other_base_channels_unchanged':True}


def apply(frames,policy):
    output=[]
    for f in frames:
        p=point(f);gate=policy['digit_only_gate'];after=f['baseline_after_local']
        if p is not None and not (gate and p[gate['feature']]>=gate['threshold']):after=f['original']
        options=[{**q,'features':{**q['features'],'is_baseline':float(equal(q['value'],after)),
                   'baseline_changed':float(not equal(f['original'],after))}} for q in f['options']]
        output.append({**f,'baseline_after_local':after,'options':options})
    return output
