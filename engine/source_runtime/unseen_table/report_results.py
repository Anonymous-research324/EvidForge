"""Sealed-output scoring with the historical five-column main table."""
from pathlib import Path
import statistics
from .common import load, sha256, read_table, equal, fresh, dump

def verify_freeze(run):
    root = Path(run)
    freeze = load(root / 'online_output_freeze.json')
    if freeze.get('test_answers_used') is not False or not freeze.get('finished'):
        raise ValueError('必须先完成并封存运行，再读取正式参考答案')
    if any((not (root / n).is_file() or sha256(root / n) != d for n, d in freeze['files'].items())):
        raise ValueError('封存输出发生变化')
    return freeze

def table(method, metrics):

    def fmt(key, percent=False):
        value = metrics[key]
        return '—' if value is None else f'{100 * value:.2f}' if percent else f'{value:.3f}'
    return '\n'.join(['| 方法 | 修复 P % | 修复 R % | 修复 F1 | 误改率 % ↓ | 二值检测 AUC |', '|---|---:|---:|---:|---:|---:|', f'| {method} | {fmt('repair_precision', True)} | {fmt('repair_recall', True)} | {fmt('repair_f1')} | {fmt('collateral_rate', True)} | {fmt('binary_detection_auc')} |'])

def score(run, reference_path, output, *, holdout_path=None):
    root = Path(run)
    verify_freeze(root)
    skill = load(root / 'skill.json')
    summary = load(root / 'summary.json')
    _, dirty = read_table(root / 'input.csv')
    _, fixed = read_table(root / 'cleaned.csv')
    _, gold = read_table(reference_path)
    identity = load(root / 'config.json')['record_fields']
    targets = skill['declaration']['targets']
    if len(gold) != len(dirty) or len(fixed) != len(dirty):
        raise ValueError('评分记录数不一致')
    if any(([r[c] for c in identity] != [g[c] for c in identity] or [r[c] for c in identity] != [f[c] for c in identity] for r, g, f in zip(dirty, gold, fixed))):
        raise ValueError('评分标识或顺序不一致')
    errors = clean = changes = correct = collateral = detection_tp = 0
    for r, g, f in zip(dirty, gold, fixed):
        for col in targets:
            error = not equal(r[col], g[col])
            changed = not equal(r[col], f[col])
            errors += error
            clean += not error
            changes += changed
            correct += error and changed and equal(f[col], g[col])
            collateral += not error and changed
            detection_tp += error and changed
    p = correct / changes if changes else None
    r = correct / errors if errors else None
    metrics = {'repair_precision': p, 'repair_recall': r, 'repair_f1': 2 * p * r / (p + r) if p is not None and r is not None and p + r else 0 if p is not None and r is not None else None, 'collateral_rate': collateral / clean if clean else None, 'binary_detection_auc': (1 + detection_tp / errors - collateral / clean) / 2 if errors and clean else None, 'errors': errors, 'correct_repairs': correct, 'incorrect_changes': changes - correct, 'clean_cells': clean, 'clean_cells_damaged': collateral, 'changed_cells': changes}
    method = '9B学习＋程序修复' if skill['learning_summary']['accepted'] and skill['learning_summary']['model_calls'] else '程序学习与修复（离线）' if not skill['learning_summary']['model_calls'] else '程序修复（9B未产出有效规则）'
    if summary.get('diagnostic_only'):
        method += '（诊断）'
    report = {'skill_id': skill['skill_id'], 'seed': skill['seed'], 'method': method, 'metrics': metrics, 'summary': summary, 'answers_read_only_after_output_freeze': True, 'estimates_counted_as_exact_repairs': False, 'reference_sha256': sha256(reference_path), 'run_freeze_sha256': sha256(root / 'online_output_freeze.json'), 'definitions': {'P': '正确恢复/实际修改', 'R': '正确恢复/原始错误', 'collateral': '被修改的原本正确格子/原本正确格子', 'AUC': '最终修改0/1标记的AUC，沿用历史口径，不用可疑标记替换'}}
    if holdout_path:
        if skill['declaration'].get('source_field'):
            raise ValueError('多来源表的下游任务需另行明确实体汇总方式')
        from .downstream import classify
        report['downstream'] = classify(gold, dirty, fixed, read_table(holdout_path)[1], targets, identity)
    folder = fresh(output)
    dump(folder / 'results.json', report)
    (folder / '主结果表.md').write_text(f'种子{skill['seed']}；{len(dirty)}行；完成{summary['completed_groups']}/{summary['groups']}批。\n\n' + table(method, metrics) + '\n\n单种子报单值。仅将准确恢复计入修复R；估计不计入。\n', encoding='utf-8')
    return report

def aggregate(paths, output):
    reports = [load(p) for p in paths]
    if len(reports) < 2 or len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('均值与标准差至少需要两个不同种子')
    if len({(r['skill_id'], r['method']) for r in reports}) != 1:
        raise ValueError('只能汇总同一技能和同一方法')
    keys = ['repair_precision', 'repair_recall', 'repair_f1', 'collateral_rate', 'binary_detection_auc']
    if any((r['metrics'][k] is None for r in reports for k in keys)):
        raise ValueError('指标缺失，不能生成均值与标准差')
    metrics = {k: {'mean': statistics.mean((r['metrics'][k] for r in reports)), 'sd': statistics.stdev((r['metrics'][k] for r in reports))} for k in keys}
    cells = [f'{metrics[k]['mean'] * (100 if k in ('repair_precision', 'repair_recall', 'collateral_rate') else 1):.{(2 if k in ('repair_precision', 'repair_recall', 'collateral_rate') else 3)}f} ± {metrics[k]['sd'] * (100 if k in ('repair_precision', 'repair_recall', 'collateral_rate') else 1):.{(2 if k in ('repair_precision', 'repair_recall', 'collateral_rate') else 3)}f}' for k in keys]
    folder = fresh(output)
    result = {'seeds': [r['seed'] for r in reports], 'metrics': metrics, 'sample_sd_ddof': 1}
    dump(folder / 'aggregate.json', result)
    header = table(reports[0]['method'], reports[0]['metrics']).splitlines()[:2]
    (folder / '主结果表.md').write_text('\n'.join(header + ['| ' + reports[0]['method'] + ' | ' + ' | '.join(cells) + ' |']) + '\n', encoding='utf-8')
    return result
