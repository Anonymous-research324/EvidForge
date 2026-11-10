from pathlib import Path
import json,time,argparse,sys
from common import ROOT,SEED,load,save,read_csv,write_csv,freeze,digest
import seed_runtime
seed_runtime.install()
from skill_runtime import client
from unseen_table.common import number,equal
from run_primary_llm_v3 import PURE
def validate(answer, current, ids, rows, cols):
    patches = answer['repairs']
    if set(answer) != {'repairs'} or set(patches) != set(current):
        raise ValueError('Incomplete row keys')
    changes = []
    for key, p in patches.items():
        if not isinstance(p, dict) or not set(p) <= set(cols):
            raise ValueError('Protected field change')
        index = ids[int(key.split('_')[1])]
        for c, v in p.items():
            if isinstance(v, str) and equal(v, rows[index][c]):
                continue
            if not isinstance(v, str) or len(v) > 96 or (v and number(v) is None):
                raise ValueError(f'Invalid output type at repairs.{key}.{c}: a changed value must be a finite numeric string or blank. Echoing the unchanged original value is allowed; a new ERR/BAD string is not a numeric replacement. Return complete row keys; use empty dict when abstaining.')
            changes.append((index, c, v))
    return changes

def resume(name):
    out = ROOT / 'outputs' / name / 'primary_llm'
    assert not (out / 'output_freeze.json').exists(), 'Already closed; do not repeat'
    pub = ROOT / 'datasets' / name / 'public'
    fields, rows = read_csv(pub / 'repair_dirty.csv')
    cols = load(pub / 'declaration.json')['targets']
    layout = load(out / 'batch_layout.json')
    history = load(out / 'progress.json')
    summary = load(out / 'shared_summary.json')
    _, output = read_csv(out / 'pure_partial.csv')
    save(out / 'progress_before_contract_recovery_v3.json', history)
    old_calls = sum((x['calls'] for x in history))
    new_calls = 0
    details = []
    for batch, ids in enumerate(layout):
        if batch < len(history) and history[batch]['accepted']:
            continue
        current = {f'row_{i}': {c: rows[j][c] for c in cols} for i, j in enumerate(ids)}
        val = {'type': 'string', 'maxLength': 96}
        rs = {'type': 'object', 'properties': {c: val for c in cols}, 'additionalProperties': False}
        schema = {'type': 'object', 'properties': {'repairs': {'type': 'object', 'properties': {k: rs for k in current}, 'required': list(current), 'additionalProperties': False}}, 'required': ['repairs'], 'additionalProperties': False}
        ok = False
        reason = None
        started = time.monotonic()
        calls = 0
        retained = history[batch]['calls'] if batch < len(history) else 0
        for attempt in range(3):
            p = out / 'trace' / f'pure_{batch:04d}_{attempt}_response.json'
            if not p.exists():
                continue
            try:
                resp = load(p)
                if resp['choices'][0]['finish_reason'] != 'stop':
                    continue
                ans = client.strict_json(resp['choices'][0]['message']['content'])
                changes = validate(ans, current, ids, rows, cols)
                ok = True
                details.append({'batch': batch, 'first_legal_original_attempt': attempt, 'extra_calls': 0})
                break
            except (ValueError, KeyError, TypeError) as e:
                reason = str(e)
        if not ok:
            payload = {'rows': current, 'learned_summary': summary}
            if reason:
                payload['format_error'] = reason
            for attempt in range(3):
                calls += 1
                new_calls += 1
                try:
                    ans = client.call(PURE, payload, schema, out / 'trace' / f'contract_v3_{batch:04d}_{attempt}', f'QsarPure1015ContractV3B{batch}R{attempt}', cols)
                    changes = validate(ans, current, ids, rows, cols)
                    ok = True
                    break
                except (ValueError, KeyError, RuntimeError, OSError, TypeError) as e:
                    save(out / 'trace' / f'contract_v3_rejected_{batch}_{attempt}.json', {'error': repr(e), 'no_partial_batch_write': True})
                    payload['format_error'] = str(e)
            if ok:
                details.append({'batch': batch, 'first_legal_continuation_attempt': attempt, 'extra_calls': calls, 'retained_failed_calls': retained})
        if not ok:
            raise RuntimeError('Still invalid after bounded format continuation; retain all output, no grading')
        for i, c, v in changes:
            output[i][c] = v
        entry = {'batch': batch, 'accepted': True, 'seconds': time.monotonic() - started, 'calls': calls, 'previous_calls_retained': retained, 'format_contract_recovery': True}
        if batch < len(history):
            history[batch] = entry
        else:
            history.append(entry)
        save(out / 'progress.json', history)
        write_csv(out / 'pure_partial.csv', fields, output)
        print('CONTRACT_V3_BATCH_CLOSED', batch + 1, '/', len(layout), 'new_calls', calls, flush=True)
    write_csv(out / 'pure_output.csv', fields, output)
    assert digest(out / 'program_only.csv') == digest(out / 'v23_output.csv')
    selected = load(ROOT / 'outputs/v23' / name / 'selection.json')
    save(out / 'format_contract_recovery_v3_audit.json', {'old_calls': old_calls, 'new_calls': new_calls, 'details': details, 'accepted_prior_batches_recalled': False, 'test_gold_read': False, 'original_PURE_prompt_unchanged': True, 'allowed_answer_set_unchanged': True, 'only_error_feedback_more_specific': True, 'extra_failed_batch_format_attempt_budget': 3, 'not_strictly_equal_total_model_compute': True})
    save(out / 'receipt.json', {'seed': SEED, 'dataset': name, 'calls': old_calls + new_calls + 1, 'program_seconds': None, 'program_seconds_note': 'initial unpersisted duration not fabricated', 'real_v23_dispatch_calls': 1, 'formal_reference_read': False, 'shared_teaching_summary_identical': True, 'system_output_equals_fresh_program': True, 'automatic_deployment_qualified': selected['passed'], 'system_native_learning_separately_recorded': True, 'contract_recovery': True, 'old_calls': old_calls, 'new_calls': new_calls, 'extra_format_retry_budget_disclosed': True})
    save(out / 'output_freeze.json', {'files': freeze(out), 'all_primary_outputs_closed': True, 'test_gold_read': False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
 if SEED!=1015 or a.dataset!='qsar':raise ValueError('Only the previously reviewed seed1015 QSAR format continuation is authorised')
 resume(a.dataset)
