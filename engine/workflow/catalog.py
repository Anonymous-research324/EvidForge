"""Match both learned ABIs and retain read-only legacy references."""
from pathlib import Path
from learning.environment import ROOT
from unseen_table.common import load, dump, sha256, read_table
from .packages import describe, verify_files

REGISTRY = ROOT / 'validation/independent_assessments.json'


def qualification(skill):
    info = describe(skill)
    record = (load(REGISTRY) if REGISTRY.is_file() else {}).get(info['manifest_sha256'])
    if not record:
        return {'auto_run_passed': False, 'reason': 'no independent validation bound to this exact checkpoint'}
    verify_files(record['code'])
    verify_files(record['evidence_files'])
    return {'auto_run_passed': record['passed'], 'assessment': record}


def register(skill, report_path):
    info = describe(skill); report = load(report_path)
    if report['frozen_skill'] != info or report['refit_after_validation']:
        raise ValueError('Assessment does not certify this exact unrefitted skill')
    verify_files(report['code'])
    freeze_path = Path(report_path).parent / 'outputs_freeze.json'
    freeze = load(freeze_path)
    evidence = {str((freeze_path.parent / n).resolve()): h for n, h in freeze['files'].items()}
    evidence[str(freeze_path.resolve())] = sha256(freeze_path)
    verify_files(evidence)
    registry = load(REGISTRY) if REGISTRY.is_file() else {}
    registry[info['manifest_sha256']] = {'passed': report['passed'], 'metrics': report['metrics'],
        'scope': info['scope'], 'code': report['code'], 'evidence_files': evidence,
        'report_sha256': sha256(report_path), 'validation_status': report['scope']}
    dump(REGISTRY, registry)
    return qualification(skill)


def match(input_path, context=None):
    fields, rows = read_table(input_path)
    learned = []; invalid = []
    for p in sorted((ROOT / 'skills').iterdir()):
        if not (p / 'configuration.json').is_file() or not (p / 'manifest.json').is_file():
            continue
        try:
            info = describe(p)
            if context == info['scope'] and set(fields) == set(info['declaration']['fields']):
                learned.append({**info, 'qualification': qualification(p)})
        except (ValueError, OSError, KeyError) as exc:
            invalid.append({'path': str(p), 'error': str(exc)})
    from application.catalog import match as legacy_match
    import tempfile
    # The historical catalogue only understands format14. Pass only original
    # legacy references to it, not the format19 registry.
    with tempfile.TemporaryDirectory(prefix='dataforge-legacy-readonly-') as td:
        for p in (ROOT / 'skills').iterdir():
            if (p / 'REFERENCE.json').is_file():
                (Path(td) / p.name).symlink_to(p.resolve(), target_is_directory=True)
        legacy = legacy_match(input_path, context, td)['legacy_candidates']
    return {'status': 'matched' if len(learned) == 1 else 'needs_selection' if learned else
                     'legacy_candidate' if legacy else 'needs_learning',
            'matches': learned, 'legacy_candidates': legacy, 'invalid_packages': invalid,
            'rows': len(rows), 'source_scope_confirmed': context is not None,
            'auto_run_ready': len(learned) == 1 and learned[0]['qualification']['auto_run_passed']}
