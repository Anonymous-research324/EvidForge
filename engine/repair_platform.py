"""Public versioned workflow; preserve table_repair.py's historical20% mode."""
from pathlib import Path
import argparse, json, subprocess, sys, time
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load, dump, read_table, write_table, fresh, sha256, number
from workflow.packages import describe, execute, code_files
from workflow.catalog import match, qualification, register


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='action', required=True)
    i = sp.add_parser('inspect'); i.add_argument('--input', required=True); i.add_argument('--context')
    a = sp.add_parser('prepare-heldout')
    for n in ('labels', 'declaration', 'context', 'out'): a.add_argument('--' + n, required=True)
    a = sp.add_parser('prepare-context'); a.add_argument('--prior', required=True); a.add_argument('--out', required=True)
    a = sp.add_parser('freeze-validation'); a.add_argument('--workflow', required=True); a.add_argument('--skill', required=True)
    a = sp.add_parser('validate'); a.add_argument('--workflow', required=True); a.add_argument('--out', required=True)
    a = sp.add_parser('register-validation'); a.add_argument('--skill', required=True); a.add_argument('--report', required=True)
    a = sp.add_parser('run')
    for n in ('input', 'skill', 'context', 'out'): a.add_argument('--' + n, required=True)
    a.add_argument('--research', action='store_true')
    for name in ('learn', 'legacy', 'prepare-merged20'):
        a = sp.add_parser(name, add_help=False); a.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.action == 'inspect':
        fields, rows = read_table(a.input)
        result = {'rows': len(rows), 'fields': {c: {'blank': sum(not r[c].strip() for r in rows),
                  'numeric': sum(number(r[c]) is not None for r in rows)} for c in fields},
                  'matching': match(a.input, load(a.context) if a.context else None),
                  'next': 'Confirm source and protected roles. Unmatched sources need corrected learning and separate validation pairs.'}
    elif a.action == 'prepare-heldout':
        from workflow.onboarding import prepare
        result = {'workflow': str(prepare(load(a.labels), load(a.declaration), load(a.context), a.out))}
    elif a.action == 'prepare-context':
        from supplementary.prepare import prepare
        describe(a.prior)
        prepare(a.prior, a.out)
        result = {'prepared_context': str(Path(a.out).resolve())}
    elif a.action == 'freeze-validation':
        from workflow.onboarding import freeze_assessment
        result = {'freeze': str(freeze_assessment(a.workflow, a.skill))}
    elif a.action == 'validate':
        from workflow.onboarding import assess
        result = assess(a.workflow, a.out)
    elif a.action == 'register-validation':
        result = register(a.skill, a.report)
    elif a.action == 'run':
        info = describe(a.skill)
        if info['scope'] != load(a.context):
            raise ValueError('Source changed; the saved qualification cannot transfer automatically')
        gate = qualification(a.skill)
        if not a.research and not gate['auto_run_passed']:
            raise ValueError('This checkpoint needs independent validation before automatic repair')
        fields, rows = read_table(a.input); out = fresh(a.out)
        dump(out / 'input_and_method_freeze.json', {'input_sha256': sha256(a.input), 'skill': info,
                                                  'code': code_files(), 'reference_api': False})
        started = time.monotonic(); output, decisions = execute(rows, a.skill)
        write_table(out / 'input.csv', fields, rows); write_table(out / 'output.csv', fields, output)
        dump(out / 'decisions.json', decisions)
        dump(out / 'output_freeze.json', {'files': {n: sha256(out / n) for n in
             ['input.csv', 'output.csv', 'decisions.json']}, 'seconds': time.monotonic() - started,
             'formal_reference_read': False, 'new9B_calls': 0, 'qualification': gate,
             'scope': 'execute a previously learned frozen skill; no new online reasoning claimed'})
        result = {'out': str(out), 'rows': len(rows), 'automatic_validation_passed': gate['auto_run_passed']}
    else:
        command = {'prepare-merged20': 'prepare'}.get(a.action, a.action)
        raise SystemExit(subprocess.run([sys.executable, '-B', str(ROOT / 'table_repair.py'), command,
                                        *a.arguments]).returncode)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
