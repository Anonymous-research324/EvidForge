"""Sealed skills, and an explicitly whitelisted model view without raw cases."""
from pathlib import Path
from unseen_table.common import load,sha256,number,canonical,fingerprint
from unseen_table.save_skill import read
from .codec import decode

def open_skill(path):
    root=Path(path);manifest=load(root/'manifest.json')
    if any(sha256(root/n)!=v for n,v in manifest['files'].items()):raise ValueError('Learned skill files changed')
    base=read(root/'base');extra=load(root/'configuration.json')
    if extra['seed']!=20260927 or not extra['raw_reference_visible_to_model'] is False:raise ValueError('Experiment visibility/seed changed')
    return root,base,decode(extra['codec']),extra['row_family'],extra['policy']

def model_view(path):
    root,base,codec,family,policy=open_skill(path);columns={}
    for col in base['declaration']['targets']:
        p=base['profiles'][col];freq=base['value_modes'][col].get('frequent',{})
        points=[{'value':v['value'],'groups':v['groups']} for v in freq.values() if number(v['value']) is not None and number(v['value'])==number(v['value']).to_integral()][:8]
        columns[col]={'meaning':base['declaration']['fields'][col].get('meaning',col),'empirical_min':p['min'],'empirical_max':p['max'],'observed_precisions':list(p['precision']),'repeated_points':points}
    rules=[{k:r[k] for k in ['kind','fields','factor','offset','prefix','suffix'] if k in r} for r in codec['rules']]
    # No training matrices, source IDs, original rows, or paired examples are exported.
    qualification=load(root/'configuration.json').get('component_qualification',{})
    return {'columns':columns,'learned_transform_families':rules,'supported_literal_precisions':codec['precisions'],'row_family':{'description':'On supported rows, multiple other nonpoint attributes agree on a rounded rational grid. Never infer this solely from the queried cell. Coincidental finite-decimal alignment is insufficient; check the clause qualification.','exceptions':family['exceptions'],'supported_learning_rows':family['support_rows']},
            'validated_new_clause_families':qualification,
            'instructions':['KEEP is a genuine alternative; existence of an inverse does not mean it is correct.','Empirical numeric ranges are learned teaching facts, not physical laws.','For a blank unique continuous measurement, a nearby estimate is not exact recovery.','Do not use IDs or classification labels; do not access original references.'],
            'protected_roles':base['declaration']['entity_fields']+base['declaration'].get('protected_fields',[]),'learning_rule_summary_only':True,'original_reference_cases_visible':False}

def verify_request(payload,columns):
    allowed={'rows','learned_summary','skill_instructions','approved_proposals','batch','previous_answer','format_error'}
    if not set(payload)<=allowed:raise ValueError('Unexpected model-visible input section')
    banned={'truth','correct','paired_teaching','paired_learning_rows','trusted_examples','learning_rows','training_rows','validation_rows','examples','reference_rows','numeric_context','row_state','models'}
    def visit(v):
        if isinstance(v,dict):
            if set(v)&banned:raise ValueError('Raw reference or model matrix entered model request')
            for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    visit(payload)
    rows=payload.get('rows',{})
    for row in rows.values():
        if set(row)!=set(columns) or any(not isinstance(v,str) for v in row.values()):raise ValueError('Model rows must be target-only current dirty inputs')
    return True
