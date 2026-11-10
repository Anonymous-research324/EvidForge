"""Joint exact-answer evidence, with held-group frequency and neighborhood features."""
import math, difflib
from .common import number, equal, canonical
from .learn_transforms import KINDS, edit_likelihood, input_pattern
FEATURE_SCHEMA_VERSION = 4
VALUE_FEATURE_COUNT = 22

def scalar(value):
    n = number(value)
    return float(max(-1000000, min(1000000, n))) if n is not None else 0.0

def value_features(value, profile, context, exclude_group=None):
    n = number(value)
    low, high = (number(profile['min']), number(profile['max']))
    numeric = n is not None
    span = max(float(high - low), 0.01) if low is not None else 1.0
    precision = max(0, -n.as_tuple().exponent) if numeric else -1
    canonical_precision = max(0, -n.normalize().as_tuple().exponent) if numeric else -1
    freq = profile['precision'].get(str(precision), profile['precision'].get(precision, 0)) / max(1, profile['nonblank'])
    inside = numeric and low is not None and (low <= n <= high)
    distance = 0 if inside or not numeric or low is None else min(20, min(abs(float(n - low)), abs(float(n - high))) / span)
    groups = profile.get('value_groups', {}).get(canonical(value), [])
    count = sum((g != exclude_group for g in groups))
    frequency = count / max(1, profile['labeled'] - int(exclude_group is not None))
    residual = min(20, abs(scalar(value) - context['estimate']) / context['uncertainty']) if numeric and context else 0.0
    neighbors = context.get('neighbors', [])
    mode_support = sum((v['weight'] for v in neighbors if numeric and abs(float(n) - v['value']) < 1e-10))
    nearest = min([abs(scalar(value) - v['value']) / max(0.01, profile['numeric_sd']) for v in neighbors], default=0.0)
    return [float(numeric), float(not value.strip()), float(inside), distance, min(20, abs(scalar(value)) / span), min(12, precision), freq, float(value[-1:] == '0'), min(60, len(value)), frequency, residual, float(bool(context)), min(64, context.get('observed_context', 0)), float(profile['numeric']), float(profile['allow_blank']), max(-20, min(20, (scalar(value) - profile['numeric_mean']) / profile['numeric_sd'])) if numeric else 0.0, max(-20, min(20, (scalar(value) - float(low)) / span)) if numeric and low is not None else 0.0, min(12, canonical_precision), mode_support, min(20, nearest), min(20, context.get('nearest_distance', 0)), float(numeric and n == n.to_integral())]

def candidate_features(row, col, candidate, fitted, contexts, exclude_group=None):
    profile = fitted['profiles'][col]
    ctx = contexts.get('columns', {}).get(col, {})
    old, new = (row[col], candidate['value'])
    before = value_features(old, profile, ctx, exclude_group)
    after = value_features(new, profile, ctx, exclude_group)
    evidence = candidate['evidence']
    kinds = candidate['kinds']
    edit, position, decimal = edit_likelihood(old, new, fitted)
    condition = fitted.get('input_conditions', {}).get(input_pattern(old), {})
    channels = condition.get('channels', {})
    count = max(1, condition.get('count', 0))
    states = contexts.get('states', {}).get(col, {})
    state_probability = states.get(canonical(new), states.get('__continuous_or_unseen__', 0.0))
    state_old = states.get(canonical(old), states.get('__continuous_or_unseen__', 0.0))
    identity = [float(col == c) for c in fitted['declaration']['targets']]
    extra = identity + [state_probability, state_old, state_probability - state_old,
                        states.get('__continuous_or_unseen__', 0.0), float(bool(states)),
                        float(edit > 0 and position >= 0), float(old != new and len(old) == len(new))]
    return extra + before + after + [float(k in kinds) for k in KINDS] + [min(1, candidate['support'] / max(1, fitted['training_cells'])), min(8, len(candidate['rule_ids'])), difflib.SequenceMatcher(None, old, new, autojunk=False).ratio(), min(20, abs(scalar(old) - scalar(new))), before[10] - after[10], float(equal(old, new)), edit, min(48, position), max(-48, min(48, decimal)), max([p['trust'] for p in evidence], default=0), sum((p['kind'] == 'relation' for p in evidence)), min(8, len(set((p['kind'] for p in evidence)))), condition.get('normal', 0), max([channels.get(k, 0) / count for k in kinds], default=0), after[18] - before[18]]

def detection_features(row, col, fitted, contexts, exclude_group=None):
    return value_features(row[col], fitted['profiles'][col], contexts.get('columns', {}).get(col, {}), exclude_group) + [float(col in contexts.get('trusted', {}).get('values', {})), fitted['profiles'][col].get('learning_error_fraction', 0)]
