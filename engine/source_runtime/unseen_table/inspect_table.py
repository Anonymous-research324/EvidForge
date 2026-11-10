"""Profile, scope match and entity-preserving unlabeled example preparation."""
from collections import Counter, defaultdict
from .common import read_table, number, sha256, fresh, dump, write_table, signature, fingerprint, validate_declaration, observation_signatures

def inspect(path):
    fields, rows = read_table(path)
    columns = {}
    for col in fields:
        values = [r[col] for r in rows]
        nums = [number(v) for v in values]
        columns[col] = {'blank_count': sum((not v.strip() for v in values)), 'numeric_count': sum((n is not None for n in nums)), 'distinct_count': len(set(values)), 'unique_nonblank': len(set(values)) == len(rows) and all((v.strip() for v in values)), 'common_examples': Counter(values).most_common(5), 'suggested_role': 'protected_label' if col.casefold() in ('class', 'label', 'target', 'secret_label') else 'needs_confirmation'}
    return {'source_sha256': sha256(path), 'rows': len(rows), 'fields': fields, 'columns': columns, 'method_templates': ['value-transform', 'field-relation', 'value-mode', 'source-consensus'], 'message': '先提供字段角色线索，再按确认的来源范围匹配；不能从列名或统计保证业务含义。'}

def match(profile, skills, context):
    matches = []
    partial = []
    for skill in skills:
        if skill['scope'] != context:
            continue
        item = {'skill_id': skill['skill_id'], 'version': skill['version'], 'artifact_sha256': skill['artifact_sha256'], 'quality_gate_passed': skill['quality_gate']['passed'], 'skill_status': skill['status']}
        (matches if set(skill['declaration']['fields']) == set(profile['fields']) else partial).append(item)
    return {'status': 'matched' if len(matches) == 1 else 'needs_selection' if matches else 'needs_learning', 'auto_run_ready': len(matches) == 1 and matches[0]['quality_gate_passed'], 'matches': matches, 'partial': partial, 'note': '新来源或字段关系变化需重新学习、校验；匹配后仍检查记录与开发组是否重复。'}

def prepare_learning(path, output, targets, identity, context, skill_id, *, count=10, seed=20260927, visible=(), source=None):
    import random
    fields, rows = read_table(path)
    declaration = {'id': skill_id, 'provenance': '用户确认目标字段、对象标识、来源和适用范围', 'fields': {c: {'meaning': '原字段' + c, **({'identity': True} if c in identity else {})} for c in fields}, 'targets': targets, 'entity_fields': identity, 'visible_context_fields': list(visible)}
    if source:
        declaration['source_field'] = source
    validate_declaration(declaration, fields)
    if count < 10:
        raise ValueError('学习和校验至少各10个独立对象组')
    record_fields = identity + ([source] if source else [])
    entities = defaultdict(list)
    record_keys = []
    for i, row in enumerate(rows):
        if any((not row[c].strip() for c in record_fields)):
            raise ValueError('对象标识或来源为空')
        entities[fingerprint([row[c] for c in identity])].append(i)
        record_keys.append(fingerprint([row[c] for c in record_fields]))
    if len(set(record_keys)) != len(record_keys):
        raise ValueError('对象标识与来源组合重复，不能拆开同一观察')
    signatures = defaultdict(set)
    for entity, ids in entities.items():
        for s in observation_signatures([rows[i] for i in ids], identity, source):
            signatures[s].add(entity)
    repeated = {entity for owners in signatures.values() if len(owners) > 1 for entity in owners}
    choices = [k for k in entities if k not in repeated]
    if len(choices) < count * 2 + 1:
        raise ValueError('独立对象不足，不能安排学习、校验和正式检查')
    random.Random(seed).shuffle(choices)
    roles = {'learning': choices[:count], 'validation': choices[count:count * 2]}
    chosen = {i for keys in roles.values() for k in keys for i in entities[k]}
    root = fresh(output)
    labels = {role: [{'group': entity, 'provenance': '待填写：正确答案的确认来源', 'records': [{'row_key': record_keys[i], 'dirty': rows[i], 'truth': {}} for i in entities[entity]]} for entity in keys] for role, keys in roles.items()}
    dump(root / 'development_labels_to_fill.json', labels)
    dump(root / 'declaration.json', declaration)
    dump(root / 'context.json', context)
    write_table(root / 'remaining.csv', fields, [r for i, r in enumerate(rows) if i not in chosen])
    report = {'source_sha256': sha256(path), 'seed': seed, 'learning_groups': count, 'validation_groups': count, 'remaining_rows': len(rows) - len(chosen), 'duplicate_entities_kept_remaining': len(repeated), 'source_peers_never_split': True, 'answers_invented': False, 'next_step': '填写已确认的目标答案和来源；不把正式剩余记录的答案交给学习阶段。'}
    dump(root / 'preparation.json', report)
    return report
