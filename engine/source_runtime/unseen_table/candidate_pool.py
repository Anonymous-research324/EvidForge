"""Exact candidates share evidence; KEEP competes with every source."""
from .common import equal, number, canonical, fingerprint
from .learn_transforms import apply_rule, edit_likelihood
from .learn_relations import proposals
MAX_CANDIDATES = 128

def edit_values(rule, old):
    if len(old) > 48:
        return
    op = rule['operation']
    digit = rule['character_class'] == 'digit'
    alphabet = rule.get('after') or ('0123456789' if digit else '')
    if op == 'replace':
        for i, ch in enumerate(old):
            if 'right' in rule and len(old) - 1 - i != rule['right']:
                continue
            if ch.isdigit() if digit else ch == rule.get('before'):
                for after in alphabet:
                    if ch != after:
                        yield (old[:i] + after + old[i + 1:])
    elif op == 'delete':
        for i, ch in enumerate(old):
            if 'right' in rule and len(old) - 1 - i != rule['right']:
                continue
            if ch.isdigit() if digit else ch == rule.get('before'):
                yield (old[:i] + old[i + 1:])
    elif op == 'insert':
        for i in range(len(old) + 1):
            if 'right' in rule and len(old) - i != rule['right']:
                continue
            for ch in alphabet:
                yield (old[:i] + ch + old[i:])

def candidates(row, col, fitted, context=None, source=None):
    old = row[col]
    pool = {}

    def add(value, kind, rule_id='', support=0, trust=0.0, sources=()):
        if value is None:
            return
        key = canonical(value)
        if key not in pool:
            pool[key] = {'value': old if equal(value, old) else value, 'kind': 'keep' if equal(value, old) else kind, 'kinds': [], 'rule_ids': [], 'support': 0, 'evidence': [], 'exact': True}
        item = pool[key]
        if kind not in item['kinds']:
            item['kinds'].append(kind)
        if rule_id and rule_id not in item['rule_ids']:
            item['rule_ids'].append(rule_id)
        item['support'] = max(item['support'], support)
        proof = {'kind': kind, 'id': rule_id, 'support': support, 'trust': trust, 'sources': list(sources)}
        if proof not in item['evidence']:
            item['evidence'].append(proof)
    add(old, 'keep')
    for rule in fitted['rules']:
        if col not in rule['fields']:
            continue
        values = edit_values(rule, old) if rule['kind'] == 'edit' else [apply_rule(rule, row, col)]
        for value in values:
            if value is None or (rule['kind'] == 'edit' and rule['character_class'] == 'digit' and (number(value) is None)):
                continue
            add(value, rule['kind'], rule['id'], rule['total_support'])
    for entry in fitted.get('value_modes', {}).get(col, {}).get('frequent', {}).values():
        add(entry['value'], 'domain', support=entry['groups'])
    if fitted['profiles'][col]['numeric']:
        for value in fitted.get('row_state', {}).get('points', []):
            add(value, 'domain', support=0)
    for proof in proposals(row, col, fitted, context or {}):
        add(proof['value'], 'relation', proof['id'], proof['support'], proof['trust'], proof['sources'])
    if source:
        add(source['value'], 'source', support=source['support'], trust=source['trust'])
    ordered = list(pool.values())
    ordered = ordered[:1] + sorted(ordered[1:], key=lambda p: (all((k == 'edit' for k in p['kinds'])), -max([e['trust'] for e in p['evidence']], default=0), -p['support'], -edit_likelihood(old, p['value'], fitted)[0], p['value']))
    for p in ordered[:MAX_CANDIDATES]:
        p['candidate_id'] = fingerprint([col, canonical(p['value'])])[:16]
    return ordered[:MAX_CANDIDATES]
