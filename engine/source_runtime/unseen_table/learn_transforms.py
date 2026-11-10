"""Infer reversible operations from labeled learning pairs, never the pollution generator."""
from collections import Counter, defaultdict
from decimal import Decimal, DecimalException
import difflib
import statistics
from .common import equal, number, literal, fingerprint
KINDS = ['keep', 'affix', 'scale', 'shift', 'constant', 'mapping', 'strip', 'casefold', 'edit', 'domain', 'relation', 'source']

def apply_rule(rule, row, col):
    old = row[col]
    kind = rule['kind']
    numeric = number(old)
    try:
        if kind == 'scale' and numeric is not None:
            return literal(numeric / Decimal(rule['factor']))
        if kind == 'shift' and numeric is not None:
            return literal(numeric - Decimal(rule['offset']))
    except DecimalException:
        return None
    if kind == 'constant':
        return rule['value']
    if kind == 'mapping':
        return rule['mapping'].get(row.get(rule['source'], ''))
    if kind == 'strip':
        return old.strip()
    if kind == 'casefold':
        return old.casefold()
    if kind == 'affix':
        pre, post = (rule['prefix'], rule['suffix'])
        if (pre or post) and old.startswith(pre) and old.endswith(post) and (len(old) > len(pre) + len(post)):
            return old[len(pre):len(old) - len(post) if post else None]
    return None

def describe(old, truth):
    if equal(old, truth) or not truth:
        return []
    result = []
    a, b = (number(old), number(truth))
    if a is not None and b is not None:
        if a and b and (a != b):
            ratio = a / b
            if ratio.is_finite() and abs(ratio) <= 1000000:
                result.append({'kind': 'scale', 'factor': literal(ratio)})
        delta = a - b
        if delta and abs(delta) <= 1000000:
            result.append({'kind': 'shift', 'offset': literal(delta)})
    if old.endswith(truth) and len(old) > len(truth):
        result.append({'kind': 'affix', 'prefix': old[:-len(truth)], 'suffix': ''})
    if old.startswith(truth) and len(old) > len(truth):
        result.append({'kind': 'affix', 'prefix': '', 'suffix': old[len(truth):]})
    if old.strip() == truth:
        result.append({'kind': 'strip'})
    if old.casefold() == truth:
        result.append({'kind': 'casefold'})
    d = edit_descriptor(old, truth)
    if d:
        before, after = d['before'], d['after']
        numeric_edit = a is not None and b is not None and all(c.isdigit() for c in before + after)
        result.append({'kind': 'edit', 'operation': d['op'], 'character_class': 'digit' if numeric_edit else 'literal', **({} if numeric_edit else {'before': before, 'after': after})})
    return result

def infer(records, declaration, *, extra_rules=()):
    targets = declaration['targets']
    profiles = {}
    observed = defaultdict(list)
    for col in targets:
        pairs = [(r, r['truth'][col]) for r in records if col in r['truth']]
        values = [v for _, v in pairs if v.strip()]
        counts = Counter(values)
        nums = [number(v) for v in values]
        numeric = [v for v in nums if v is not None]
        profiles[col] = {'values': sorted(counts), 'frequencies': dict(counts), 'numeric': bool(values) and len(numeric) / len(values) >= 0.9, 'min': literal(min(numeric)) if numeric else '', 'max': literal(max(numeric)) if numeric else '', 'precision': dict(Counter((max(0, -v.as_tuple().exponent) for v in numeric))), 'nonblank': len(values), 'labeled': len(pairs), 'allow_blank': any((not v.strip() for _, v in pairs))}
        profiles[col]['canonical_precision'] = dict(Counter((max(0, -v.normalize().as_tuple().exponent) for v in numeric)))
        profiles[col]['numeric_mean'] = statistics.mean((float(v) for v in numeric)) if numeric else 0
        profiles[col]['numeric_sd'] = max(0.01, statistics.pstdev((float(v) for v in numeric))) if numeric else 1
        for r, truth in pairs:
            for definition in describe(r['dirty'][col], truth):
                single_edit = any((d['kind'] == 'edit' for d in describe(r['dirty'][col], truth)))
                observed[fingerprint(definition)].append((definition, col, r['group'], truth, single_edit))
        if len(pairs) >= 3 and values and (len(values) == len(pairs)) and all((equal(v, values[0]) for v in values)):
            definition = {'kind': 'constant', 'value': values[0], 'target': col}
            observed[fingerprint(definition)] += [(definition, col, r['group'], truth, False) for r, truth in pairs]
    rules = []
    for key, points in observed.items():
        groups = {p[2] for p in points}
        if len(groups) < 3:
            continue
        definition = points[0][0]
        support = Counter((p[1] for p in points))
        fields = list(support)
        if definition['kind'] in ('scale', 'shift'):
            diverse = {p[3] for p in points}
            nonedit_groups = {p[2] for p in points if not p[4]}
            if len(diverse) < 3 or len(nonedit_groups) < 3:
                continue
        if definition['kind'] in ('scale', 'shift') or definition.get('character_class') == 'digit':
            fields = [c for c in targets if profiles[c]['numeric']]
        rules.append({**definition, 'id': key[:20], 'fields': fields, 'support': dict(support), 'groups': len(groups), 'total_support': len(points)})
    for rule in extra_rules:
        support = defaultdict(set)
        for r in records:
            for col in rule['fields']:
                if col in r['truth'] and (not equal(r['dirty'][col], r['truth'][col])) and supports(rule, r['dirty'], col, r['truth'][col]):
                    support[col].add(r['group'])
        fields = [c for c, groups in support.items() if len(groups) >= 3]
        if fields:
            definition = {k: v for k, v in rule.items() if k not in ('fields', 'id', 'support', 'groups', 'total_support', 'origin')}
            key = fingerprint(definition)[:20]
            if key not in {r['id'] for r in rules}:
                rules.append({**definition, 'fields': fields, 'id': key, 'origin': 'model', 'support': {c: len(support[c]) for c in fields}, 'groups': len(set().union(*(support[c] for c in fields))), 'total_support': sum((len(support[c]) for c in fields))})
    result = {'rules': rules, 'profiles': profiles, 'declaration': declaration}
    result['edit_patterns'] = learn_edits(records, result)
    conditions = defaultdict(Counter)
    normal = defaultdict(lambda: [0, 0, set()])
    for r in records:
        for col, truth in r['truth'].items():
            old = r['dirty'][col]
            shape = input_pattern(old)
            normal[shape][0] += equal(old, truth)
            normal[shape][1] += 1
            normal[shape][2].add(r['group'])
            for rule in rules:
                if col in rule['fields'] and rule['kind'] != 'edit' and equal(apply_rule(rule, r['dirty'], col), truth):
                    conditions[shape][rule['kind']] += 1
    result['input_conditions'] = {k: {'normal': p[0] / p[1], 'count': p[1], 'groups': len(p[2]), 'channels': dict(conditions[k])} for k, p in normal.items()}
    return result

def supports(rule, row, col, truth):
    if rule['kind'] == 'edit':
        from .candidate_pool import edit_values
        return any((equal(v, truth) for v in edit_values(rule, row[col])))
    return equal(apply_rule(rule, row, col), truth)

def input_pattern(value):
    n = number(value)
    if n is None:
        return 'blank' if not value.strip() else 'text:' + ''.join(('D' if c.isdigit() else c for c in value))[:48]
    precision = max(0, -n.as_tuple().exponent)
    return f'number:{min(12, precision)}:{int(value.endswith('0'))}:{int(n < 0)}:{min(12, len(value.split('.')[0]))}'

def edit_descriptor(old, new):
    # Direct aligned replacement avoids repeated digits being misread as insert/delete.
    mismatches = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    if len(old) == len(new) and len(mismatches) == 1:
        i = mismatches[0]
        return {'op': 'replace', 'right': len(old) - 1 - i, 'left': i, 'decimal': i - old.find('.') if '.' in old else i, 'before': old[i], 'after': new[i]}
    ops = [p for p in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if p[0] != 'equal']
    if len(ops) != 1:
        return None
    op, i, j, k, l = ops[0]
    if j - i > 1 or l - k > 1:
        return None
    return {'op': op, 'right': len(old) - j, 'left': i, 'decimal': i - old.find('.') if '.' in old else i, 'before': old[i:j], 'after': new[k:l]}

def learn_edits(records, fitted):
    by_shape = defaultdict(Counter)
    global_counts = Counter()
    examples = []
    for r in records:
        for col, truth in r['truth'].items():
            old = r['dirty'][col]
            if equal(old, truth):
                continue
            if any((rule['kind'] in ('scale', 'shift', 'affix') and col in rule['fields'] and equal(apply_rule(rule, r['dirty'], col), truth) for rule in fitted['rules'])):
                continue
            d = edit_descriptor(old, truth)
            if not d:
                continue
            key = f'{d['op']}:{d['right']}:{d['before']}>{d['after']}'
            shape = input_pattern(old)
            by_shape[shape][key] += 1
            global_counts[key] += 1
            examples.append({'field': col, 'group': r['group'], 'pattern': shape, **d})
    return {'by_pattern': {s: dict(c) for s, c in by_shape.items()}, 'global': dict(global_counts), 'examples': examples}

def edit_likelihood(old, new, fitted):
    d = edit_descriptor(old, new)
    if not d:
        return (0.0, -1.0, -1.0)
    key = f'{d['op']}:{d['right']}:{d['before']}>{d['after']}'
    model = fitted.get('edit_patterns', {})
    local = model.get('by_pattern', {}).get(input_pattern(old), {})
    global_ = model.get('global', {})
    counts = local if sum(local.values()) >= 5 else global_
    exact = counts.get(key, 0)
    total = max(1, sum(counts.values()))
    return ((exact + 0.2) / (total + 2), float(d['right']), float(d['decimal']))
