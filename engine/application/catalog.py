"""Read-only legacy matching and manifest-bound learned-skill qualification."""
from pathlib import Path
import yaml
from learning.environment import RUNTIME
from unseen_table.common import load, dump, sha256, read_table
from skill_runtime.package import open_skill

ROOT = Path(__file__).resolve().parents[1]
ASSESSMENTS = ROOT / 'validation/skill_assessments.json'


def quality(metrics):
    return (metrics.get('changes', 0) >= 5 and metrics.get('clean', metrics.get('normal', 0)) >= 10
            and metrics.get('P', 0) >= .98 and metrics.get('damage') is not None
            and metrics['damage'] <= .005)


def qualification(path):
    path = Path(path)
    fingerprint = sha256(path / 'manifest.json')
    registry = load(ASSESSMENTS) if ASSESSMENTS.is_file() else {}
    record = registry.get(fingerprint)
    return {'manifest_sha256': fingerprint,
            'auto_run_passed': bool(record and record.get('output_frozen') and quality(record.get('metrics', {}))),
            'assessment': record,
            'note': 'Refit/source-cross-calibration alone cannot grant independent qualification.'}


def register(path, metrics, evidence):
    config=load(Path(path)/'configuration.json');open_skill(config['base_skill_path'])
    if not evidence.get('output_frozen') or not evidence.get('independent_source_groups'):
        raise ValueError('Qualification needs sealed output on sources excluded from fitting')
    registry = load(ASSESSMENTS) if ASSESSMENTS.is_file() else {}
    key = sha256(Path(path) / 'manifest.json')
    registry[key] = {**evidence, 'metrics': metrics, 'quality_passed': bool(quality(metrics)),
                     'scope': 'empirical single-seed qualification of this exact skill; not a universal guarantee'}
    dump(ASSESSMENTS, registry)
    return qualification(path)


def inventory(registry=None):
    result = []
    for path in sorted(Path(registry or ROOT / 'skills').iterdir()):
        if not path.is_dir():
            continue
        if (path / 'REFERENCE.json').is_file():
            ref = load(path / 'REFERENCE.json')
            actual = (ROOT.parent / ref['package_path']).resolve()
            boundary = (ROOT.parent / 'DataForge_SkillPlatform_v2/skills').resolve()
            if boundary not in actual.parents:
                raise ValueError('Legacy reference leaves original frozen library')
            available = all((actual / n).is_file() and sha256(actual / n) == v for n, v in ref['files'].items())
            result.append({'id': ref['id'], 'kind': 'frozen_legacy', 'available': available,
                           'path': str(actual), 'entry': 'legacy_entry.py'})
        elif (path / 'configuration.json').is_file():
            config=load(path/'configuration.json')
            if config.get('format_version')!=14:continue
            _,base,*_=open_skill(config['base_skill_path'])
            gate=qualification(path)
            result.append({'id':base['skill_id'],'kind':'learned_v14','available':True,
                'path':str(path.resolve()),'scope':config['scope'],'fields':list(base['declaration']['fields']),
                'auto_run_quality_passed':gate['auto_run_passed'],'qualification':gate})

    return result


def match(csv, context=None, registry=None):
    fields, rows = read_table(csv)
    exact, legacy = [], []
    for item in inventory(registry):
        if not item['available']:
            continue
        if item['kind'] == 'learned_v14':
            if context is not None and item['scope'] == context and set(fields) == set(item['fields']):
                exact.append(item)
        else:
            schema = yaml.safe_load((Path(item['path']) / 'schema.yaml').read_text())
            mapping = {}
            for name, definition in schema['fields'].items():
                aliases = {name.casefold()} | {a.casefold() for a in definition.get('aliases', [])}
                hits = [f for f in fields if f.casefold() in aliases]
                if len(hits) == 1:
                    mapping[name] = hits[0]
            if len(set(mapping) & set(schema['signature'])) >= schema['min_signature'] and set(schema['required']) <= set(mapping):
                legacy.append({**item, 'proposed_mapping': mapping,
                               'next': 'Use unchanged legacy_entry.py match/profile checks and original execution.'})
    status = 'matched' if len(exact) == 1 else 'needs_selection' if exact else 'legacy_candidate' if legacy else 'needs_learning'
    return {'status': status, 'matches': exact, 'legacy_candidates': legacy, 'rows': len(rows),
            'source_scope_confirmed': context is not None,
            'auto_run_ready': len(exact) == 1 and exact[0]['auto_run_quality_passed'],
            'note': 'Column names/statistics propose candidates. Confirm source and roles; changed sources need new learning/checks.'}
