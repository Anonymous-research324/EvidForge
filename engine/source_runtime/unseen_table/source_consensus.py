"""Field-specific source reliability; records stay grouped by entity."""
from collections import defaultdict
from .common import canonical, equal, fingerprint

def fit(records, declaration):
    source = declaration.get('source_field')
    if not source:
        return {}
    if source not in declaration.get('visible_context_fields', []):
        raise ValueError('来源字段需明确登记为可见上下文')
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, set()]))
    for r in records:
        for col, truth in r['truth'].items():
            item = stats[col][r['dirty'][source]]
            item[0] += equal(r['dirty'][col], truth)
            item[1] += 1
            item[2].add(r['group'])
    return {'source_field': source, 'entity_fields': declaration['entity_fields'], 'accuracy': {col: {s: {'score': count / max(1, total), 'groups': len(groups)} for s, (count, total, groups) in by.items()} for col, by in stats.items()}}

def contexts(rows, fitted):
    model = fitted.get('source_consensus', {})
    out = [{} for _ in rows]
    if not model:
        return out
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[fingerprint([row[c] for c in model['entity_fields']])].append(i)
    for ids in grouped.values():
        if len(ids) < 2:
            continue
        for i in ids:
            for col, accuracy in model['accuracy'].items():
                votes = {}
                for j in ids:
                    if i == j:
                        continue
                    source = rows[j][model['source_field']]
                    s = accuracy.get(source, {})
                    value = rows[j][col]
                    if value.strip() and s.get('groups', 0) >= 5 and (s.get('score', 0) >= 0.98):
                        key = canonical(value)
                        votes[key] = {'value': value, 'support': s['groups'], 'trust': s['score']}
                if len(votes) == 1:
                    out[i][col] = next(iter(votes.values()))
    return out

def proposals(rows, fitted):
    """Method-template entry for grouped, reliable peer candidates."""
    return contexts(rows, fitted)
