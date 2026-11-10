"""A separate classifier-training view of unresolved numerical corruption.

No replacement value is guessed or written back to the repaired table. Source
held predictions calibrate numeric-only error flags; current uncertain values
become missing ONLY in the classifier training view. IDs/classes are excluded
from detection and stay unchanged in the exported training view.
"""
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from .environment import RUNTIME
from unseen_table.common import load,dump,equal,number
from unseen_table.rank_candidates import predict_rows
from skill_runtime.package import open_skill
from skill_runtime.codec import fit as fit_codec
from skill_runtime.row_family import fit as fit_family
from skill_runtime.execute import propose_from_cells

THRESHOLDS=[.5,.7,.8,.9,.95,.98,.99,.995,.999]


def fit(path):
    root=Path(path);config=load(root/'configuration.json');basepath=Path(config['base_skill_path'])
    _,base,_,_,policy=open_skill(basepath);records=load(root/'source_records_backend.json')['records'];points=[]
    for fold,(u,v) in enumerate(GroupKFold(3).split(records,groups=[r['group'] for r in records])):
        source=[records[int(i)] for i in u];query=[records[int(i)] for i in v];model=load(basepath/f'fold-{fold}-model.json')
        if set(model['numeric_context']['training_groups'])!={r['group'] for r in source}:raise ValueError('Held source overlap')
        model['policy']=base['policy']
        original=[dict(r['dirty']) for r in query];cells=predict_rows(original,model)
        ds=propose_from_cells(original,cells,model,fit_codec(source,base['declaration']),fit_family(source,base['declaration']),policy)
        repaired=[dict(r) for r in original]
        for d in ds:repaired[d['row']][d['field']]=d['after']
        confirmed=[{**r['dirty'],**r['truth']} for r in query]
        for scenario,rows in [('dirty',original),('base_repaired',repaired),('confirmed',confirmed)]:
            for c in predict_rows(rows,model):
                old=rows[c['row']][c['column']]
                if number(old) is None:continue
                r=query[c['row']];points.append({'group':r['group'],'scenario':scenario,
                    'probability':c['error_probability'],'error':not equal(old,r['truth'][c['column']])})
    candidates=[]
    for threshold in THRESHOLDS:
        selected=[q for q in points if q['probability']>=threshold]
        wrong=sum(q['error'] for q in selected);normal=len(selected)-wrong;groups={q['group'] for q in selected}
        if len(selected)<8 or len(groups)<4 or normal or wrong/len(selected)<.98:continue
        residual=sum(q['error'] and q['scenario']=='base_repaired' for q in selected)
        if residual<4:continue
        candidates.append((residual,wrong,threshold,{'threshold':threshold,'numeric_checks':len(selected),
             'source_groups':len(groups),'source_flag_P':wrong/len(selected),'source_normal_flags':normal,
             'residual_errors_identified':residual}))
    result={'enabled':bool(candidates),'gate':max(candidates,key=lambda q:q[:3])[3] if candidates else None,
        'calibration_scenarios':['dirty','base_repaired','confirmed'],'numeric_only':True,
        'use':'classifier TRAINING view only; never a cell repair output or an exact repair claim',
        'formal_reference_read':False,'raw_reference_cases_visible_to_model':False}
    return result


def apply(rows,path,policy):
    output=[dict(r) for r in rows];flags=[]
    if not policy['enabled']:return output,flags
    config=load(Path(path)/'configuration.json');_,base,*_=open_skill(config['base_skill_path'])
    for c in predict_rows(rows,base):
        value=rows[c['row']][c['column']]
        if number(value) is not None and c['error_probability']>=policy['gate']['threshold']:
            output[c['row']][c['column']]='';flags.append({'row':c['row'],'field':c['column'],'error_score':c['error_probability']})
    return output,flags
