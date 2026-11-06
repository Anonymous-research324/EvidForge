"""Frozen v6 execution, with killable inference and full-denominator partial output."""
import json, time
from pathlib import Path
from .common import fresh, dump, load, read_table, write_table, sha256, fingerprint, observation_signatures, atomic_bytes
from .save_skill import validate
from .validate_skill import permitted, region
from .workflow_budget import run_worker

def batches(count, groups):
    if not 1 <= groups <= count:
        raise ValueError('批次数必须不大于行数')
    q, r = divmod(count, groups)
    start = 0
    result = []
    for i in range(groups):
        size = q + (i < r)
        result.append(list(range(start, start + size)))
        start += size
    return result

def initialize(csv, skill, context, output, *, groups=80, allow_partial=False):
    validate(skill)
    fields, rows = read_table(csv)
    if not skill['quality_gate']['passed'] and (not allow_partial):
        raise ValueError('技能未通过75%等开发目标；仅允许明确标注的诊断运行')
    if skill['scope'] != context:
        raise ValueError('来源适用范围发生变化，需要重新学习或校验')
    if set(fields) != set(skill['declaration']['fields']):
        raise ValueError('字段发生变化，需要明确映射')
    identity = skill['declaration']['entity_fields']
    source = skill['declaration'].get('source_field')
    record_fields = identity + ([source] if source else [])
    keys = [fingerprint([r[c] for c in identity]) for r in rows]
    row_keys = [fingerprint([r[c] for c in record_fields]) for r in rows]
    if len(set(row_keys)) != len(row_keys) or any((not r[c].strip() for r in rows for c in record_fields)):
        raise ValueError('对象标识与来源的组合必须非空且唯一；同一来源重复观察需先声明独立记录标识')
    for audit in skill['audit'].values():
        if set(keys) & set(audit['entities']) or observation_signatures(rows, identity, source) & set(audit['signatures']):
            raise ValueError('正式输入与学习或校验对象重复')
    root = fresh(output)
    write_table(root / 'input.csv', fields, rows)
    dump(root / 'skill.json', skill)
    dump(root / 'batches.json', batches(len(rows), groups))
    dump(root / 'config.json', {'groups': groups, 'seed': skill['seed'], 'format_version': 6, 'diagnostic_only': bool(allow_partial), 'record_fields': record_fields, 'model_workflow_total_call_limit': 12, 'max_output_tokens': 1500, 'input_byte_limit': 6000, 'wall_seconds': 900, 'test_answers_used': False})
    fixed = {n: sha256(root / n) for n in ['input.csv', 'skill.json', 'batches.json', 'config.json']}
    code = {str(p.absolute()): sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    dump(root / 'fixed_files.json', {'files': fixed, 'code': code})
    return root

def verify_fixed(root):
    fixed = load(root / 'fixed_files.json')
    if any((sha256(root / n) != d for n, d in fixed['files'].items())) or any((sha256(n) != d for n, d in fixed['code'].items())):
        raise ValueError('冻结输入、技能或运行代码发生变化')

def run(root, *, client=None, wall_seconds=None):
    root = Path(root)
    if (root / 'started.json').exists() or (root / 'online_output_freeze.json').exists():
        raise ValueError('执行记录已存在，不能覆盖或重置预算')
    verify_fixed(root)
    started = time.monotonic()
    skill = validate(load(root / 'skill.json'))
    fields, rows = read_table(root / 'input.csv')
    config = load(root / 'config.json')
    layout = load(root / 'batches.json')
    if not skill['quality_gate']['passed'] and (not config['diagnostic_only']):
        raise ValueError('未通过质量目标')
    elapsed_learning = skill['learning_summary'].get('seconds', 0)
    budget = max(0, skill['learning_summary'].get('workflow_wall_limit', 900) - elapsed_learning)
    if wall_seconds is not None:
        if not 0 <= wall_seconds <= 900:
            raise ValueError('执行预算须在0至900秒以内')
        budget = min(budget, wall_seconds)
    deadline = started + budget
    with (root / 'run.lock').open('x') as lock:
        lock.write('one execution only\n')
    mode = '9b_learned_program_execution' if skill['learning_summary']['accepted'] else 'program_execution_9b_no_executable_rules' if skill['learning_summary']['model_calls'] else 'offline_learned_program'
    dump(root / 'started.json', {'mode': mode, 'model': getattr(client, 'model', None), 'seed': skill['seed'], 'diagnostic_only': config['diagnostic_only'], 'test_answers_used': False})
    targets = skill['declaration']['targets']
    repaired = [dict(r) for r in rows]
    errors = []
    cells = []
    if budget > 0:
        request = root / 'inference_request.json'
        result = root / 'inference_result.json'
        dump(request, {'action': 'predict', 'rows': rows, 'skill': skill})
        try:
            run_worker(request, result, max(0, deadline - time.monotonic()), root / 'inference.log')
            cells = load(result)
        except TimeoutError as error:
            errors.append(str(error))
    else:
        errors.append('时间预算用完；全表保留并计入评分')
    reviews = []
    changes = []
    lines = []
    conflicts = {}
    model_rule_ids = set(skill['learning_summary'].get('final_model_rule_ids', []))
    model_cells = 0
    if cells and len(cells) != len(rows) * len(targets):
        raise ValueError('工作进程未返回全表结果')
    for c in cells:
        i, col = (c['row'], c['column'])
        cid = f'{i}:{col}'
        apply = permitted(c, skill['policy'])
        if apply:
            repaired[i][col] = c['top']['value']
            changes.append({'cell_id': cid, 'before': rows[i][col], 'after': repaired[i][col], 'rule_ids': c['top']['rule_ids'], 'evidence': c['top']['evidence'], 'stage': c['stage'], 'region': region(c)})
            model_cells += int(bool(model_rule_ids & set(c['top']['rule_ids'])))
        suspicious = c['error_probability'] >= 0.5 or c['top']['kind'] != 'keep'
        reviews.append({'cell_id': cid, 'row': i, 'column': col, 'reviewed': True, 'applied': bool(apply), 'status': 'repair' if apply else 'suspect' if suspicious else 'keep', 'error_probability': c['error_probability'], 'selected': c['top'], 'margin': c['margin'], 'region': region(c), 'stage': c['stage'], 'candidate_count': c['candidate_count'], 'context_proofs': c['context_proofs'], 'estimate': c['context_estimate'] if suspicious and (not apply) else None})
        lines.append(json.dumps({'cell_id': cid, 'candidates': c['options']}, ensure_ascii=False, separators=(',', ':')))
        if suspicious and (not apply) and (region(c) not in conflicts):
            conflicts[region(c)] = {'field': col, 'observed': c['original'], 'choices': [{'value': p['value'], 'kinds': p['kinds']} for p in c['options'][:3]]}
    if not cells:
        reviews = [{'cell_id': f'{i}:{col}', 'row': i, 'column': col, 'reviewed': False, 'applied': False, 'status': 'unreviewed', 'reason': '预算用完，原值保留，仍计入全部错误的分母'} for i in range(len(rows)) for col in targets]
    receipts = [{'batch_id': b, 'rows': ids, 'status': 'complete' if cells else 'unreviewed_budget', 'model_calls': 0} for b, ids in enumerate(layout)]
    notes = []
    if client and skill['quality_gate']['passed'] and (not config['diagnostic_only']):
        client.deadline = min(client.deadline, deadline)
        for start in range(0, min(9, len(conflicts)), 3):
            if time.monotonic() >= deadline or skill['learning_summary']['model_calls'] + getattr(client, 'calls', 0) >= 12:
                break
            try:
                notes.append(client.complete('只返回JSON {"new_rule_notes":["需要的示范或规律"]}；不能批准修改或编造原始答案。', {'representative_conflicts': list(conflicts.values())[start:start + 3]}))
            except Exception as error:
                errors.append(str(error))
                break
    verify_fixed(root)
    if any((repaired[i][c] != rows[i][c] for i in range(len(rows)) for c in fields if c not in targets)):
        raise ValueError('保护字段被修改')
    write_table(root / 'cleaned.csv', fields, repaired)
    dump(root / 'reviews.json', reviews)
    dump(root / 'changes.json', changes)
    dump(root / 'batch_receipts.json', receipts)
    dump(root / 'new_rule_notes.json', notes)
    atomic_bytes(root / 'candidates.jsonl', ('\n'.join(lines) + '\n').encode())
    seconds = time.monotonic() - started
    summary = {'mode': mode, 'seed': skill['seed'], 'groups': len(layout), 'completed_groups': len(layout) if cells else 0, 'target_cells': len(rows) * len(targets), 'reviewed_cells': len(cells), 'unreviewed_cells': len(reviews) - len(cells), 'confirmed_changes': len(changes), 'learning_model_calls': skill['learning_summary']['model_calls'], 'runtime_model_calls': getattr(client, 'calls', 0), 'model_rule_supported_changes': model_cells, 'model_errors': errors, 'quality_gate': skill['quality_gate'], 'diagnostic_only': config['diagnostic_only'], 'seconds': seconds, 'full_execution': bool(cells), 'wall_budget_seconds': budget, 'active_workflow_seconds': elapsed_learning + seconds, 'test_answers_used': False, 'all_rows_retained': True, 'model_does_not_approve_each_cell': True}
    dump(root / 'summary.json', summary)
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    dump(root / 'online_output_freeze.json', {'files': files, 'test_answers_used': False, 'finished': True})
    return summary
