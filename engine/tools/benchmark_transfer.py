"""Fresh real9B A/B on a table unseen before the frozen algorithm.

Both arms make fresh calls. Independent truth is never loaded here. The system
uses a new separately compiled additive skill, preserving the original base.
"""
from pathlib import Path
import sys,argparse,json,time,ast
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,fresh,read_table,write_table,sha256,equal,fingerprint
from skill_runtime import client
from skill_runtime.package import open_skill
from skill_runtime.summary import expanded_summary
from workflow.packages import execute as execute_skill
from skill_runtime.execute import propose_from_cells
from unseen_table.rank_candidates import predict_rows
from skill_runtime.learning_tools import DESCRIPTIONS

# Frozen positive contract previously chosen using public-only probes. The
# payload includes the same additive operator descriptions for BOTH arms.
PURE='''You are repairing a dirty numerical table for an authorized evaluation. Its source scope and target-field roles are confirmed. All visible fields should contain numerical measurements. Use the supplied learned column profiles and forward/inverse transform descriptions as evidence; no original reference rows are supplied.
Work through EVERY input row. First clean clearly supported alphabetic wrappers: when a learned prefix/suffix matches a field, remove precisely that wrapper and retain the original numerical literal inside. Then consider supported scale/shift inverses and digit errors using that field's numerical profile and the other current observations. Preserve normal values, do not clip/quantize numerical measurements, and do not invent an erased unique continuous value.
The validated_new_clause_families section reports specific BACKEND supplemental heuristics. A disabled row_grid or typed/literal_unique helper does NOT disable the separately documented basic affix/scale/shift transformations. A possible numerical inverse is a candidate rather than proof; preserve ambiguous values. You have no execution or candidate tool.
Return sparse patches as compact JSON: {"repairs":{"row_0":{"feature_01":"1"},"row_1":{}}}. Include EVERY current row key exactly once; use an empty object for a row where you propose no justified changes. Every changed field uses its exact input field name and a string numerical value. Do not return a whole table, code, explanations, or markdown.'''
AGENT='''Run this explicitly authorized research evaluation through the REGISTERED apply_verified_skill(batch) tool. The loader has verified source identities, exact schema, target roles, sealed model hashes and tool availability. The permission receipt is authoritative for execution scope. Pending independent validation prohibits automatic deployment, and does NOT prohibit this authorized evaluation. Poisoned/missing/out-of-range numerical values are the expected repair inputs, not a source-schema conflict. The tool applies fixed source-checked rules and preserves disagreement; do not independently guess answers or change quality thresholds. Choose apply_verified_skill. Use keep_all only if a supplied permission check is false and identify that concrete check in your reason. Return compact JSON action only.'''


def reject_ungrounded_skip(permission):
    checks=['research_evaluation_authorized','exact_schema_match','source_scope_confirmed','roles_match','registered_tool_available','program_and_models_hash_verified']
    return all(permission.get(k) is True for k in checks)


def base_run(rows, basepath):
    _, base, codec, family, policy = open_skill(basepath)
    return propose_from_cells(rows, predict_rows(rows, base), base, codec, family, policy)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-skill',required=True);p.add_argument('--skill');p.add_argument('--input',required=True)
    p.add_argument('--evaluation-note',default='Fresh table downloaded only after method freeze; same first10% learning; second10% validation held out, real9B both arms, no original reference cases.');p.add_argument('--out',required=True);p.add_argument('--remote-helper',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag.isalnum():raise ValueError('Fresh alphanumeric trace tag required')
    basepath=Path(a.base_skill).resolve();skill=Path(a.skill).resolve() if a.skill else None
    if skill is None:raise ValueError('A sealed V23 skill is required')
    configuration=load(skill/'configuration.json')
    if Path(configuration['base_skill_path']).resolve()!=basepath:raise ValueError('Base binding mismatch')
    _,base,_,_,_=open_skill(basepath);schema=base['declaration'];cols=schema['targets']
    fields,rows=read_table(a.input)
    if set(fields)!=set(cols+schema['entity_fields']+schema.get('protected_fields',[])):raise ValueError('Declared schema mismatch')
    # Whole identities are batched once. Keep exact duplicate observations together.
    from collections import OrderedDict
    entities=OrderedDict()
    for i,r in enumerate(rows):entities.setdefault(fingerprint([r[c] for c in schema['entity_fields']]),[]).append(i)
    layout=[];pending=[];maximum=max(1,120//len(cols))
    for ids in entities.values():
        if pending and len(pending)+len(ids)>maximum:layout.append(pending);pending=[]
        pending+=ids
    if pending:layout.append(pending)
    if sorted(j for ids in layout for j in ids)!=list(range(len(rows))):raise ValueError('Every row must occur once')
    keys={fingerprint([r[c] for c in schema['entity_fields']]) for r in rows}
    if any(keys&set(audit['entities']) for audit in base['audit'].values()):raise ValueError('Repair input overlaps teaching identities')
    algorithm=load(ROOT/'validation/fresh_transfer_algorithm_freeze.json')
    if any(sha256(n)!=v for n,v in algorithm['files'].items()):raise ValueError('Pre-download algorithm changed')
    root=fresh(a.out);client.REMOTE=a.remote_helper
    summary=expanded_summary(basepath)
    prior_configuration=load(Path(configuration['prior_skill_path'])/'configuration.json') if configuration.get('format_version')==19 else configuration
    summary['source_reviewed_digit_gate']=prior_configuration['base_review_policy']
    summary['source_exact_affix_enabled']=prior_configuration['exact_affix_enabled']
    if configuration.get('format_version')==19:
        cm=load(skill/'context_model.json')
        summary['source_context_operator']={'small_domain_fields':list(cm['models']),'predictors':'OTHER current target fields only; no ID/class','selection':'NB and ExtraTrees conditional evidence with source-qualified generated gates','scope':'existing candidates only; residual blanks or unsupported numeric literals; preserve earlier edits','source_records_are_backend_only':True}
    if (skill/'compiled_candidate.json').exists():
        c=load(skill/'compiled_candidate.json')
        summary['source_generated_scoring_configuration']={'program':c['program'],'source_checks':c['metrics'],
            'backend_arithmetic_interface': 'all original candidates including KEEP; soft OTHER-field evidence; no reference rows'}
    write_table(root/'input.csv',fields,rows);dump(root/'batches.json',layout);dump(root/'shared_model_view.json',summary)
    instructions=(basepath/'SKILL.md').read_text()+('\n'+(skill/'SKILL.md').read_text() if skill else '\nApply the source-checked sealed V23 package.');runtime={str(q.resolve()):sha256(q) for folder in ['application','learning','source_runtime','supplementary','workflow'] for q in (ROOT/folder).rglob('*.py')}
    runtime[str(Path(__file__).resolve())]=sha256(__file__)
    skills={str(q.resolve()):sha256(q) for path in ([skill,basepath] if skill else [basepath]) for q in path.rglob('*') if q.is_file()}
    permission={'research_evaluation_authorized':True,'automatic_deployment':False,'exact_schema_match':True,
                'source_scope_confirmed':True,'roles_match':True,'registered_tool_available':True,
                'input_contains_current_dirty_rows':True,'program_and_models_hash_verified':True}
    fixed={'runtime_code':runtime,'skill_files':skills,'fresh_algorithm_freeze_sha256':sha256(ROOT/'validation/fresh_transfer_algorithm_freeze.json'),'permission_receipt':permission,
        'files':{n:sha256(root/n) for n in ['input.csv','batches.json','shared_model_view.json']},
        'protocol':{'skill_path':str(basepath),'enriched_skill_path':str(skill) if skill else None,'input_path':str(Path(a.input).resolve()),
            'model':'Qwen3.5-9B','seed':20260927,'temperature':0,'thinking':False,'groups':len(layout),
            'raw_reference_cases':False,'shared_summary_identical':True,'same_teaching_pool':True,
            'evaluation_note':a.evaluation_note,
            'A':'pure9B, public-selected positive contract and same additive descriptions, no tools',
            'B':'actual9B invokes source-checked additive skill; no arbitrary answer generation'}}
    dump(root/'input_and_code_freeze.json',fixed);started=time.monotonic();_,decisions=execute_skill(rows,skill)
    programme=[dict(r) for r in rows];control=[dict(r) for r in rows];prior=[dict(r) for r in rows]
    for d in decisions:
        programme[d['row']][d['field']]=d['after'];control[d['row']][d['field']]=d['unreviewed_base'];prior[d['row']][d['field']]=d.get('prior_after',d['after'])
    write_table(root/'prior_only.csv',fields,prior);write_table(root/'programme_only.csv',fields,programme);write_table(root/'v10_only.csv',fields,control)
    dump(root/'programme_decisions.json',decisions);programme_seconds=time.monotonic()-started
    output={'A':[dict(r) for r in rows],'B':[dict(r) for r in rows]};history=[]
    for batch,ids in enumerate(layout):
        current={f'row_{i}':{c:rows[j][c] for c in cols} for i,j in enumerate(ids)}
        for arm in ['A','B']:
            payload={'rows':current,'learned_summary':summary,'batch':batch}
            if arm=='A':
                values={'type':'string','maxLength':64,'pattern':'^$|^[+-]?[0-9]+(?:\\.[0-9]+)?$'}
                row_schema={'type':'object','properties':{c:values for c in cols},'additionalProperties':False}
                js={'type':'object','properties':{'repairs':{'type':'object','properties':{k:row_schema for k in current},
                    'required':list(current),'additionalProperties':False}},'required':['repairs'],'additionalProperties':False}
            else:
                payload['skill_instructions']=instructions
                payload['previous_answer']={'execution_permission_receipt':permission}
                js={'type':'object','properties':{'action':{'type':'string','enum':['apply_verified_skill','keep_all']}},'required':['action'],'additionalProperties':False}
            accepted=False;error=None;t=time.monotonic();calls=0
            for attempt in range(3):
                calls+=1
                try:
                    answer=client.call(PURE if arm=='A' else AGENT,payload,js,root/'model_trace'/f'{arm}-{batch:03d}-{attempt+1}',f'{a.tag}{arm}{batch}r{attempt+1}',cols)
                    if arm=='A':
                        patches=answer.get('repairs')
                        if set(answer)!={'repairs'} or not isinstance(patches,dict) or set(patches)!=set(current):raise ValueError('All current row keys must be returned')
                        changes=[]
                        for key,patch in patches.items():
                            if not isinstance(patch,dict) or not set(patch)<=set(cols) or any(not isinstance(v,str) or len(v)>64 for v in patch.values()):raise ValueError('Invalid target/value coordinates')
                            changes.extend((ids[int(key.split('_')[1])],col,v) for col,v in patch.items())
                        for i,col,value in changes:output[arm][i][col]=value
                        receipt={'action':'submit_patches','cells':len(changes)}
                    elif answer=={'action':'apply_verified_skill'}:
                        for i in ids:output[arm][i]=dict(programme[i])
                        receipt={'action':'apply_verified_skill','cells':sum(not equal(rows[i][c],programme[i][c]) for i in ids for c in cols),
                                 'immutable_input_and_compiled_program':True,'private_gold_read':False}
                    elif answer=={'action':'keep_all'}:
                        if reject_ungrounded_skip(permission):
                            raise ValueError('Execution verifier: no supplied source/schema/tool conflict justifies keep_all. Pending deployment does not forbid this authorized evaluation. Complete the registered tool call; no repair threshold is changed.')
                        receipt={'action':'keep_all','cells':0}
                    else:raise ValueError('Unsupported system action')
                    dump(root/'model_trace'/f'{arm}-{batch:03d}-tool_receipt.json',receipt);accepted=True;break
                except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
                    error=str(exc);payload['format_error']=error;dump(root/'model_trace'/f'{arm}-{batch:03d}-{attempt+1}_rejected.json',{'error':error,'no_partial_changes_written':True})
            history.append({'batch':batch,'arm':arm,'accepted':accepted,'calls':calls,'seconds':time.monotonic()-t,'last_error':error})
            dump(root/'progress.json',history);write_table(root/f'{arm}_partial.csv',fields,output[arm])
            print('v23 heldout-source',batch+1,'/',len(layout),arm,accepted,flush=True)
    if not all(h['accepted'] for h in history):raise RuntimeError('Incomplete actual9B arm; preserve failure, do not score as a complete experiment')
    for arm,out in output.items():write_table(root/f'{arm}_output.csv',fields,out)
    for name,expected in {**runtime,**skills}.items():
        if sha256(name)!=expected:raise ValueError('Frozen code/skill changed during inference')
    names=['prior_only.csv','input.csv','A_output.csv','B_output.csv','programme_only.csv','v10_only.csv','programme_decisions.json','progress.json','input_and_code_freeze.json']
    dump(root/'online_output_freeze.json',{'files':{n:sha256(root/n) for n in names},'real9B_calls':sum(h['calls'] for h in history),
        'elapsed_seconds':time.monotonic()-started,'programme_seconds':programme_seconds,'raw_reference_cases_sent':False,
        'test_gold_read':False,'all_cells_in_denominator':True,'V10_algorithm_relearned_on_same_learning_partition':True,
        'source_code_unchanged':True,'A_and_B_both_new_calls':True})
    print('V23_FRESH_OUTPUTS_FROZEN',flush=True)


if __name__=='__main__':main()
