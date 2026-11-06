"""Versioned component skills; every file and reconstructed model is integrity checked."""
import re
from pathlib import Path
from .common import fresh, dump, fingerprint, load, sha256, validate_declaration
from .features import FEATURE_SCHEMA_VERSION
COMPONENTS = {'transforms.json': ['rules', 'edit_patterns', 'input_conditions', 'state_folds'], 'relations.json': ['relations', 'source_consensus'], 'value_modes.json': ['value_modes', 'profiles'], 'models.json': ['selector', 'stage1_selector', 'detector', 'numeric_context', 'row_state', 'stage1_decision_model', 'decision_model'], 'policy.json': ['policy', 'stage1_policy'], 'quality_report.json': ['quality_gate', 'validation_summary', 'validation_before_qualification', 'qualification', 'learning_cross_check', 'training_summary', 'folds'], 'provenance.json': ['audit', 'method_usage', 'learning_summary', 'learning_bundle_sha256', 'strategy_sha256_before_validation']}

def validate(skill):
    data = dict(skill)
    expected = data.pop('artifact_sha256', None)
    if data.get('format_version') != 6:
        raise ValueError('v6只加载v6技能；旧技能使用冻结旧入口')
    if data.get('feature_schema') != FEATURE_SCHEMA_VERSION:
        raise ValueError('特征版本不一致，不能载入旧选择器')
    if fingerprint(data) != expected:
        raise ValueError('技能文件发生变化')
    validate_declaration(data['declaration'], list(data['declaration']['fields']))
    for name in ('policy', 'stage1_policy'):
        if any(not isinstance(key, str) or key.count(':') != 1 for key in data[name].get('by_region', {})):
            raise ValueError('技能使用早期开发版的修改条件；请用当前v6重新学习，不能混用')
    return skill

def save(path, skill):
    data = {k: v for k, v in skill.items() if k != 'artifact_sha256'}
    if not re.fullmatch('[a-zA-Z][a-zA-Z0-9_-]{0,63}', data['skill_id']):
        raise ValueError('技能名称格式不正确')
    if data.get('format_version') != 6 or data.get('feature_schema') != FEATURE_SCHEMA_VERSION:
        raise ValueError('只能保存当前v6特征版本')
    validate_declaration(data['declaration'], list(data['declaration']['fields']))
    full = validate({**data, 'artifact_sha256': fingerprint(data)})
    root = fresh(path)
    header = dict(full)
    for name, keys in COMPONENTS.items():
        component = {k: header.pop(k) for k in keys if k in header}
        dump(root / name, component)
    header['components'] = list(COMPONENTS)
    dump(root / 'skill.json', header)
    text = [f'技能：{data['skill_id']}，版本6.0.0', '', f'状态：{data['status']}', '学习方式：' + data['learning_summary']['mode'], '适用范围：' + str(data['scope']), '允许修改：' + '、'.join(data['declaration']['targets']), '第一轮建议作为临时上下文；第二轮重新比较并可撤回，最终通过学习内检查。', '字段关系、常见值、变换和KEEP共同比较；同分放弃。估计不计准确恢复。', f'开发修复R：{data['validation_summary']['recall']:.2%}；75%目标通过：{data['quality_gate']['passed']}', '开发数据已重用，不能作为独立测试；此包不代表正式9B实验达标。', '范围或字段角色变化时重新学习，不能覆盖旧版本。']
    (root / 'SKILL.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    dump(root / 'manifest.json', {'artifact_sha256': full['artifact_sha256'], 'files': {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}})
    return validate(full)

def read(path):
    path = Path(path)
    root = path if path.is_dir() else path.parent
    manifest = load(root / 'manifest.json')
    expected = set(COMPONENTS) | {'skill.json', 'SKILL.md'}
    if set(manifest['files']) != expected:
        raise ValueError('技能组件清单不完整')
    if any((sha256(root / name) != digest for name, digest in manifest['files'].items())):
        raise ValueError('技能组件发生变化')
    full = load(root / 'skill.json')
    if full.pop('components') != list(COMPONENTS):
        raise ValueError('技能组件版本不一致')
    for name, keys in COMPONENTS.items():
        values = load(root / name)
        if not set(values) <= set(keys) or set(values) & set(full):
            raise ValueError('组件字段不合法')
        full.update(values)
    if full['artifact_sha256'] != manifest['artifact_sha256']:
        raise ValueError('技能整体指纹变化')
    return validate(full)

def list_skills(registry):
    return [read(p.parent) for p in sorted(Path(registry).glob('*/skill.json'))]
