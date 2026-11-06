"""One checked execution interface for the two existing numerical ABIs."""
from pathlib import Path
from learning.environment import ROOT
from unseen_table.common import load, sha256
from skill_runtime.package import open_skill


def code_files():
    folders = ('application', 'learning', 'supplementary', 'source_runtime', 'workflow')
    paths = list(ROOT.glob('*.py')) + [q for name in folders for q in (ROOT / name).rglob('*.py')]
    return {str(q.resolve()): sha256(q) for q in sorted(paths)}


def verify_files(files):
    changed = [p for p, expected in files.items() if not Path(p).is_file() or sha256(p) != expected]
    if changed:
        raise ValueError('Frozen files changed: ' + ', '.join(changed[:5]))


def describe(path):
    root = Path(path).resolve()
    config = load(root / 'configuration.json')
    version = config.get('format_version')
    if version not in (14, 19):
        raise ValueError('Unsupported learned-skill format; use the unchanged legacy entry for legacy skills')
    manifest = load(root / 'manifest.json')
    if any(sha256(root / n) != h for n, h in manifest['files'].items()):
        raise ValueError('Sealed skill changed')
    _, base, *_ = open_skill(config['base_skill_path'])
    if version == 19:
        prior = describe(config['prior_skill_path'])
        if prior['format_version'] != 14:
            raise ValueError('Context skill requires a format14 prior')
    return {'path': str(root), 'format_version': version, 'scope': config['scope'],
            'declaration': base['declaration'], 'manifest_sha256': sha256(root / 'manifest.json')}


def execute(rows, skill):
    info = describe(skill)
    if any(set(r) != set(info['declaration']['fields']) for r in rows):
        raise ValueError('Input fields differ from the confirmed declaration')
    if info['format_version'] == 19:
        from supplementary.execute import run
    else:
        from learning.execute import run
    decisions = run(rows, skill)
    output = [dict(r) for r in rows]
    seen = set()
    for d in decisions:
        coordinate = d['row'], d['field']
        if coordinate in seen or d['field'] not in info['declaration']['targets']:
            raise ValueError('Duplicate or unapproved repair coordinate')
        seen.add(coordinate)
        if d['before'] != rows[d['row']][d['field']]:
            raise ValueError('Decision does not belong to this observation')
        output[d['row']][d['field']] = d['after']
    return output, decisions
