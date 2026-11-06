"""Bounded repair hypotheses, keeping the immutable observed value as anchor.

These generate proposals; none may bypass source-held clause qualification.
Column intervals and literal precision are empirical evidence, not physical
constraints. No missing continuous value is fabricated by a regression mean.
"""
from decimal import Decimal
from unseen_table.common import number, canonical, equal
from .codec import possible


def supported_precision(value, profile):
    n = number(value)
    if n is None:
        return False
    precision = max(0, -n.normalize().as_tuple().exponent)
    observed = [int(p) for p, count in profile['precision'].items() if count >= 2]
    return bool(observed and precision <= max(observed))


def bounded_hypotheses(row, cell, base, codec):
    col = cell['column']; old = row[col]; profile = base['profiles'][col]
    if not profile['numeric']:
        return {}
    lo, hi = number(profile['min']), number(profile['max'])
    n = number(old)
    result = {}
    if n is None and old.strip():
        literals = {}
        for rule in codec['rules']:
            if rule['kind'] != 'affix' or col not in rule['fields']:
                continue
            prefix, suffix = rule['prefix'], rule['suffix']
            if not (prefix or suffix) or not old.startswith(prefix) or (suffix and not old.endswith(suffix)):
                continue
            literal = old[len(prefix):len(old)-len(suffix) if suffix else len(old)]
            if number(literal) is not None:
                literals[canonical(literal)] = literal
        if len(literals) == 1:
            value = next(iter(literals.values()))
            result['exact_affix_literal'] = value
            if lo is not None and lo <= number(value) <= hi:
                result['bounded_affix_literal'] = value
    if n is not None and lo is not None and not lo <= n <= hi:
        feasible = {}
        for candidate in cell['options']:
            value = candidate['value']; target = number(value)
            if target is not None and lo <= target <= hi and not equal(old, value) and possible(old, value, col, codec):
                if supported_precision(value, profile):
                    feasible[canonical(value)] = value
        if len(feasible) == 1:
            result['column_precision_unique'] = next(iter(feasible.values()))
        top = cell['top']
        if (number(top['value']) is not None and lo <= number(top['value']) <= hi
                and not equal(old, top['value']) and supported_precision(top['value'], profile)
                and possible(old, top['value'], col, codec) and top['score'] >= .97 and cell['margin'] >= .05):
            result['column_supported_top'] = top['value']
    # Context is built only from already trusted other fields. A nearby mean
    # is evidence for ranking an exact candidate; it is never itself returned.
    context = cell.get('context_estimate')
    if context and n is not None and context.get('observed_context', 0) >= 3:
        options = []
        for candidate in cell['options']:
            target = number(candidate['value'])
            if target is not None and possible(old, candidate['value'], col, codec) and supported_precision(candidate['value'], profile):
                residual = abs(float(target) - context['estimate']) / context['uncertainty']
                options.append((residual, candidate['value']))
        options.sort(key=lambda p: (p[0],p[1]))
        if len(options) >= 2 and not equal(old, options[0][1]):
            best, runner = options[0][0], options[1][0]
            for gap in [2, 4, 8]:
                if best <= 1 and runner-best >= gap:
                    result['context_exact_gap_' + str(gap)] = options[0][1]
    return result
