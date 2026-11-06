"""Inventory v6 skills and fingerprinted, read-only historical references."""
from pathlib import Path
from .common import PROJECT, load, sha256
from .save_skill import list_skills

def inventory(registry):
    result = []
    for path in sorted(Path(registry).glob('*/REFERENCE.json')):
        reference = load(path)
        root = (PROJECT / reference['package_path']).resolve()
        expected = (PROJECT / 'DataForge_SkillPlatform_v2' / 'skills').resolve()
        if expected not in root.parents:
            raise ValueError('历史技能引用必须指向冻结v2技能目录')
        checked = all(((root / name).is_file() and sha256(root / name) == digest for name, digest in reference['files'].items()))
        result.append({'id': reference['id'], 'kind': 'frozen_legacy_reference', 'available': checked, 'entry': 'legacy_entry.py', 'historical_result_replay': 'legacy-replay', 'scope': 'original historical experiment; discovery requires original v2 scope checks'})
    for skill in list_skills(registry):
        result.append({'id': skill['skill_id'], 'kind': 'learned_v6_skill', 'available': True, 'status': skill['status'], 'learning_mode': skill['learning_summary']['mode'], 'quality_gate_passed': skill['quality_gate']['passed'], 'entry': 'python -m unseen_table run'})
    return result
