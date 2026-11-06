"""Literal and exact affine relations fitted on confirmed clean learning fields."""
from collections import defaultdict
from itertools import combinations
from decimal import DecimalException
from .common import canonical, clean_row, number, literal, equal, fingerprint

def fit(records, declaration, hypotheses=()):
    targets = declaration['targets']
    sources = targets + [c for c in declaration.get('visible_context_fields', []) if c not in declaration['entity_fields']]
    clean = [(clean_row(r, declaration), r['group']) for r in records]
    relations = []
    eligible = []
    for c in sources:
        repeated = defaultdict(set)
        for row, g in clean:
            if c in row:
                repeated[canonical(row[c])].add(g)
        if len(repeated) > 1 and any((len(groups) >= 3 for groups in repeated.values())):
            eligible.append(c)

    def mapping(det, target):
        buckets = defaultdict(list)
        for row, group in clean:
            if target in row and all((c in row and row[c].strip() for c in det)):
                buckets[fingerprint([canonical(row[c]) for c in det])].append((row[target], group))
        entries = {k: {'value': points[0][0], 'groups': len({g for _, g in points})} for k, points in buckets.items() if len({g for _, g in points}) >= 3 and all((equal(v, points[0][0]) for v, _ in points))}
        if entries:
            rule = {'kind': 'lookup', 'sources': list(det), 'target': target, 'mapping': entries, 'support_groups': len({g for pts in buckets.values() for _, g in pts})}
            rule['id'] = fingerprint(rule)[:20]
            return rule
    for target in targets:
        singles = []
        for source in eligible:
            if source != target:
                rule = mapping([source], target)
                if rule:
                    relations.append(rule)
                    singles.append(source)
        if not singles:
            for det in list(combinations([c for c in eligible if c != target], 2))[:64]:
                rule = mapping(det, target)
                if rule:
                    relations.append(rule)
        for source in sources:
            if source == target:
                continue
            points = [(number(row.get(source, '')), number(row.get(target, '')), group) for row, group in clean]
            points = [p for p in points if p[0] is not None and p[1] is not None]
            unique = {x for x, _, _ in points}
            if len(unique) < 4 or len({g for _, _, g in points}) < 4:
                continue
            first = points[0]
            second = next((p for p in points if p[0] != first[0]))
            try:
                factor = (second[1] - first[1]) / (second[0] - first[0])
                offset = first[1] - factor * first[0]
                if not factor or abs(factor) > 1000000 or abs(offset) > 1000000:
                    continue
                if not all((factor * x + offset == y for x, y, _ in points)):
                    continue
            except DecimalException:
                continue
            rule = {'kind': 'affine', 'sources': [source], 'target': target, 'factor': literal(factor), 'offset': literal(offset), 'support_groups': len({g for _, _, g in points})}
            rule['id'] = fingerprint(rule)[:20]
            relations.append(rule)
    known = {r['id'] for r in relations}
    for h in hypotheses:
        det = h.get('sources', [])
        target = h.get('target')
        if target not in targets or not 1 <= len(det) <= 2 or len(set(det)) != len(det) or (target in det) or (not set(det) <= set(sources)):
            continue
        rule = mapping(det, target)
        if rule and rule['id'] not in known:
            rule['origin'] = 'model'
            relations.append(rule)
            known.add(rule['id'])
    return relations[:2048]

def proposals(row, col, fitted, context):
    values = context.get('values', {})
    trust = context.get('trust', {})
    out = []
    for rule in fitted.get('relations', []):
        if rule['target'] != col or not all((trust.get(c, 0) >= 0.98 and c in values for c in rule['sources'])):
            continue
        if rule['kind'] == 'lookup':
            entry = rule['mapping'].get(fingerprint([canonical(values[c]) for c in rule['sources']]))
            if not entry:
                continue
            value = entry['value']
            support = entry['groups']
        else:
            n = number(values[rule['sources'][0]])
            if n is None:
                continue
            try:
                value = literal(number(rule['factor']) * n + number(rule['offset']))
            except DecimalException:
                continue
            support = rule['support_groups']
        out.append({'value': value, 'kind': 'relation', 'id': rule['id'], 'support': support, 'trust': min((trust[c] for c in rule['sources'])), 'sources': rule['sources']})
    return out
