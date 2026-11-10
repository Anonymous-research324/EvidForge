"""Literal-preserving input, atomic output and immutable fingerprints."""
import csv
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)

def guard_output(path):
    path = Path(path).resolve()
    for root in (PROJECT / 'dataforge-ext', PROJECT / 'DataForge_SkillPlatform', PROJECT / 'DataForge_SkillPlatform_v2', PROJECT / 'DataForge_UnseenTable_v3', PROJECT / 'DataForge_UnseenTable_v3_1', PROJECT / 'DataForge_UnseenTable_v4', PROJECT / 'DataForge_UnseenTable_v5'):
        if path == root or root in path.parents:
            raise ValueError('不能写入旧代码或旧技能目录')
    archive = PROJECT / 'experiments'
    if path == archive or archive in path.parents:
        raise ValueError('实验归档区只读；新结果请放在新代码目录的work/中')
    for parent in (path, *path.parents):
        if parent == PACKAGE:
            continue
        if (parent / 'online_output_freeze.json').exists() or (parent / 'delivery_manifest.json').exists():
            raise ValueError('不能写入已经封存的实验或交付目录')
    return path

def fresh(path):
    path = guard_output(path)
    if path.exists():
        raise ValueError('输出目录已存在；请使用独立的新目录')
    path.mkdir(parents=True)
    return path

def atomic_bytes(path, data):
    path = guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def dump(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())

def read_table(path):
    path = Path(path)
    delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        fields = next(reader, [])
        if not fields or any((not c.strip() for c in fields)) or len(set(fields)) != len(fields):
            raise ValueError('表头为空或有重复字段')
        rows = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(fields):
                raise ValueError(f'第{number}行字段数不一致')
            rows.append(dict(zip(fields, row)))
    if not rows:
        raise ValueError('表格没有数据')
    return (fields, rows)

def write_table(path, fields, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())

def number(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        return None
    try:
        result = Decimal(value)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None

def equal(a, b):
    x, y = (number(a), number(b))
    return x == y if x is not None and y is not None else a == b

def literal(value):
    return format(value.normalize(), 'f')

def validate_declaration(declaration, fields):
    targets = declaration.get('targets', [])
    identity = declaration.get('entity_fields', [])
    if set(declaration.get('fields', {})) != set(fields):
        raise ValueError('声明必须覆盖所有字段')
    if not targets or not set(targets) <= set(fields) or len(set(targets)) != len(targets):
        raise ValueError('需要明确不重复的可修改字段')
    if not identity or not set(identity) <= set(fields) or set(identity) & set(targets):
        raise ValueError('需要明确受保护的记录标识字段')
    if not declaration.get('provenance'):
        raise ValueError('需要确认字段声明的来源')
    protected = set(declaration.get('protected_fields', [])) | {c for c, info in declaration['fields'].items() if info.get('identity') or info.get('role') in ('label', 'identity', 'protected')}
    protected |= {c for c in fields if c.casefold() in ('class', 'label', 'target', 'secret_label')}
    if protected & set(targets):
        raise ValueError('分类标签与保护字段不能作为修复目标')
    if not set(declaration.get('visible_context_fields', [])) <= set(fields) - set(targets):
        raise ValueError('额外可见字段声明有误')
    if protected & set(declaration.get('visible_context_fields', [])) - set(identity):
        raise ValueError('保护字段不能作为可见修复依据')
    source = declaration.get('source_field')
    if source and (source in targets or source in identity or source not in declaration.get('visible_context_fields', [])):
        raise ValueError('来源字段需保护且单独登记为可见上下文')

def canonical(value):
    n = number(value)
    return literal(n) if n is not None else value

def clean_row(record, declaration):
    """Only confirmed targets and explicitly visible, non-protected context."""
    result = dict(record['truth'])
    for col in declaration.get('visible_context_fields', []):
        if col not in declaration['entity_fields']:
            result[col] = record['dirty'][col]
    return result

def signature(row, identity):
    return fingerprint({c: v for c, v in row.items() if c not in identity})

def observation_signatures(rows, identity, source=None):
    if not source:
        return {signature(row, identity) for row in rows}
    packets = {}
    for row in rows:
        entity = fingerprint([row[c] for c in identity])
        packets.setdefault(entity, []).append(signature(row, identity))
    return {fingerprint(sorted(packet)) for packet in packets.values()}

def normalize_bundle(bundle, declaration, min_groups=10):
    if set(bundle) != {'learning', 'validation'}:
        raise ValueError('只接受学习和校验组，不接受测试答案')
    fields = set(declaration['fields'])
    targets = set(declaration['targets'])
    identity = declaration['entity_fields']
    output = {}
    audits = {}
    for zone in ('learning', 'validation'):
        groups = bundle[zone]
        if len(groups) < min_groups:
            raise ValueError(f'{zone}至少需要{min_groups}组')
        keys = set()
        entities = set()
        signatures = set()
        result = []
        coordinates = set()
        group_ids = set()
        for group in groups:
            if not isinstance(group.get('group'), str) or not group['group'] or group['group'] in group_ids:
                raise ValueError('组标识为空或重复')
            group_ids.add(group['group'])
            if not isinstance(group.get('provenance'), str) or not group['provenance'].strip() or '待填写' in group['provenance'] or (not group.get('records')):
                raise ValueError('每组需要记录和确认的答案来源，不能使用待填写内容')
            if not any((record.get('truth') for record in group['records'])):
                raise ValueError('每组至少需要一个真实填写的目标答案，未标注组不能充当学习或校验支持')
            local_entities = set()
            for record in group['records']:
                dirty, truth = (record.get('dirty'), record.get('truth'))
                if not isinstance(dirty, dict) or set(dirty) != fields:
                    raise ValueError('学习记录必须保留完整原始字段')
                if not isinstance(truth, dict) or not set(truth) <= targets:
                    raise ValueError('答案只能包含允许修改的字段')
                if any((not isinstance(v, str) for v in [*dirty.values(), *truth.values()])):
                    raise ValueError('原值与答案必须保留为字符串')
                key = record.get('row_key')
                if not isinstance(key, str) or not key or key in keys:
                    raise ValueError('记录标识为空或重复')
                if any((not dirty[c].strip() for c in identity)):
                    raise ValueError('记录标识字段为空')
                entity = fingerprint([dirty[c] for c in identity])
                local_entities.add(entity)
                source = declaration.get('source_field')
                if source:
                    coordinate = fingerprint([entity, dirty[source]])
                    if not dirty[source].strip() or coordinate in coordinates:
                        raise ValueError('对象与来源组合为空或重复')
                    coordinates.add(coordinate)
                keys.add(key)
                signatures.add(signature(dirty, identity))
                result.append({'group': group['group'], 'row_key': key, 'dirty': dirty, 'truth': truth})
            if len(local_entities) != 1 or entities & local_entities:
                raise ValueError('同一实体必须完整放在一个组中')
            entities.update(local_entities)
        signatures = observation_signatures([r['dirty'] for r in result], identity, declaration.get('source_field'))
        output[zone] = result
        audits[zone] = {'groups': len(groups), 'records': len(result), 'row_keys': sorted(keys), 'entities': sorted(entities), 'signatures': sorted(signatures)}
    for key in ('row_keys', 'entities', 'signatures'):
        if set(audits['learning'][key]) & set(audits['validation'][key]):
            raise ValueError('学习和校验存在重复记录')
    if any((not any((r['truth'] for r in output[zone])) for zone in output)):
        raise ValueError('学习和校验需要真实填写的目标答案')
    return (output, audits)
