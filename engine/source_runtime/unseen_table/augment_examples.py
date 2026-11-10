"""Synthetic practice from transformations inferred inside the training fold only."""
import random
from decimal import Decimal
from .common import number, literal, equal

def forward(rule, truth, rng):
    n = number(truth)
    kind = rule['kind']
    if kind == 'scale' and n is not None:
        return format(n * Decimal(rule['factor']), 'f')
    if kind == 'shift' and n is not None:
        return format(n + Decimal(rule['offset']), 'f')
    if kind == 'affix':
        return rule['prefix'] + truth + rule['suffix']
    if kind == 'edit':
        op = rule['operation']
        digit = rule['character_class'] == 'digit'
        if digit:
            positions = [i for i, c in enumerate(truth) if c.isdigit()]
            if not positions:
                return None
            i = rng.choice(positions)
            if op == 'replace':
                ch = rng.choice([c for c in '0123456789' if c != truth[i]])
                value = truth[:i] + ch + truth[i + 1:]
            elif op == 'delete':
                value = truth[:i] + rng.choice('0123456789') + truth[i:]
            else:
                value = truth[:i] + truth[i + 1:]
            return value if number(value) is not None else None
        before, after = (rule.get('before', ''), rule.get('after', ''))
        if after and after in truth:
            return truth.replace(after, before, 1)
    return None

def augment(records, fitted, *, seed=20260927):
    rng = random.Random(seed)
    result = list(records)
    missing = sum((not r['dirty'][c].strip() and bool(v.strip()) for r in records for c, v in r['truth'].items()))
    total = sum((len(r['truth']) for r in records))
    errors = sum(not equal(r['dirty'][c], v) for r in records for c, v in r['truth'].items())
    corruption_rate = errors / max(1, total)
    for r in records:
        normal = {**r, 'dirty': {**r['dirty'], **r['truth']}, 'synthetic': True, 'row_key': r['row_key'] + ':normal', 'source_group': r['group']}
        result.append(normal)
        row = dict(r['dirty'])
        for col, truth in r['truth'].items():
            row[col] = truth
            if rng.random() >= corruption_rate:
                continue
            available = [rule for rule in fitted['rules'] if col in rule['fields'] and rule['kind'] in ('scale', 'shift', 'affix', 'edit')]
            if missing and rng.random() < missing / max(1, errors):
                row[col] = ''
                continue
            if available:
                rule = rng.choices(available, weights=[p['total_support'] for p in available])[0]
                value = forward(rule, truth, rng)
                if value is not None:
                    row[col] = value
        result.append({**r, 'dirty': row, 'synthetic': True, 'row_key': r['row_key'] + ':practice', 'source_group': r['group']})
    return result
