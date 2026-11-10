"""Group-supported frequent values in mixed columns; no automatic mode filling."""
from collections import defaultdict
from .common import canonical, number

def fit(records, fitted):
    result = {}
    for col, profile in fitted['profiles'].items():
        groups = defaultdict(set)
        literals = {}
        for r in records:
            if col in r['truth'] and r['truth'][col].strip():
                key = canonical(r['truth'][col])
                groups[key].add(r['group'])
                literals[key] = r['truth'][col]
        common = {k: {'value': literals[k], 'groups': len(g), 'group_ids': sorted(g)} for k, g in groups.items() if len(g) >= 3}
        kind = 'constant' if len(groups) == 1 else 'discrete' if len(groups) <= 8 else 'mixed' if common else 'continuous' if profile['numeric'] else 'categorical'
        result[col] = {'kind': kind, 'frequent': common, 'groups': len({r['group'] for r in records if col in r['truth']})}
        profile['semantic_frequencies'] = {k: len(g) for k, g in groups.items()}
    return result
