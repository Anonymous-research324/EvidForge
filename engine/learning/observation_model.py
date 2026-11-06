"""Source-learned observation mixtures; target values never enter context.

This model ranks existing exact alternatives. It never writes an estimated
mean, never treats an uncertain KEEP as confirmed, and reads truth only in fit.
The inference API receives a dirty row, candidates and a fitted source model.
"""
from collections import Counter
from decimal import Decimal
import math
import numpy as np
from .environment import RUNTIME
from unseen_table.common import number, equal
from unseen_table.candidate_pool import candidates
from skill_runtime.codec import fit as fit_codec, aliases

CHANNELS = ('keep', 'affix', 'scale', 'shift', 'digit_edit', 'unknown')
UNKNOWN_FLOOR = .02


def form(value):
    return 'blank' if not value.strip() else 'numeric' if number(value) is not None else 'text'


def channels(old, new, column, codec):
    if equal(old, new):
        return ['keep']
    result = set()
    variants = list(dict.fromkeys([new] + aliases(new, codec)))
    for rule in codec['rules']:
        if column not in rule['fields']:
            continue
        for v in variants:
            n = number(v)
            observed = (rule['prefix'] + v + rule['suffix'] if rule['kind'] == 'affix'
                        else format(n * Decimal(rule['factor']), 'f') if rule['kind'] == 'scale' and n is not None
                        else format(n + Decimal(rule['offset']), 'f') if rule['kind'] == 'shift' and n is not None else None)
            if observed == old:
                result.add(rule['kind'])
    for v in variants:
        different = [(a, b) for a, b in zip(old, v) if a != b]
        if len(old) == len(v) and len(different) == 1 and all(c.isdigit() for c in different[0]):
            result.add('digit_edit')
    return sorted(result) or ['unknown']


def fit(records, base, mode='raw'):
    context = base['numeric_context']
    if not context:
        return None
    columns = context['columns']
    groups = [r['group'] for r in records]
    if set(groups) != set(context['training_groups']):
        raise ValueError('Noisy-context source does not match the fitted base')
    raw = np.array([[float(number(r['truth'][c])) if number(r['truth'].get(c, '')) is not None else np.nan
                     for c in columns] for r in records])
    transformed = np.sign(raw) * np.log1p(np.abs(raw)) if mode == 'signed_log' else raw
    center = np.nanmedian(transformed, axis=0)
    scale = np.maximum(np.nanstd(transformed, axis=0), .01)
    z = (transformed - center) / scale
    # A target's source bandwidth is fitted without that target in distances.
    sigma = []
    for j in range(len(columns)):
        residuals = []
        for i in range(len(records)):
            use = [k for k in range(len(columns)) if k != j]
            errors = (z[:, use] - z[i, use]) ** 2
            distance = np.nanmean(np.minimum(errors, 25), axis=1)
            excluded = np.array([g == groups[i] for g in groups]) | ~np.isfinite(z[:, j])
            distance[excluded] = np.inf
            nearest = np.argsort(distance)[:3]
            nearest = nearest[np.isfinite(distance[nearest])]
            if len(nearest) and np.isfinite(z[i, j]):
                residuals.append(abs(z[i, j] - np.mean(z[nearest, j])))
        sigma.append(float(np.clip(np.quantile(residuals, .7) if residuals else 1, .2, 1.5)))
    codec = fit_codec(records, base['declaration'])
    counts = {f: Counter({c: 1.0 for c in CHANNELS}) for f in ['blank', 'text', 'numeric']}
    for r in records:
        for c in columns:
            old, truth = r['dirty'][c], r['truth'][c]
            matched = channels(old, truth, c, codec)
            for channel in matched:
                counts[form(old)][channel] += 1 / len(matched)
    priors = {f: {c: float(v / sum(counts[f].values())) for c, v in counts[f].items()} for f in counts}
    multiplicity = Counter(groups)
    group_weights = np.array([1 / multiplicity[g] for g in groups]); group_weights /= group_weights.sum()
    return {'columns': columns, 'z': z, 'center': center, 'scale': scale, 'sigma': np.array(sigma),
            'priors': priors, 'codec': codec, 'training_groups': groups, 'group_weights': group_weights,
            'mode': mode, 'base': base}


def transform(value, column_index, fitted):
    n = number(value)
    if n is None:
        return None
    x = float(n)
    if fitted['mode'] == 'signed_log':
        x = math.copysign(math.log1p(abs(x)), x)
    return (x - fitted['center'][column_index]) / fitted['scale'][column_index]


def option_prior(old, options, column, fitted):
    priors = fitted['priors'][form(old)]
    matches = [channels(old, q['value'], column, fitted['codec']) for q in options]
    denominator = Counter(c for cs in matches for c in cs)
    weights = np.array([sum(priors[c] / denominator[c] for c in cs if c != 'unknown') for cs in matches])
    # Unrepresented mechanisms remain explicit uncertainty rather than evidence
    # redistributed to the surviving alternatives.
    unknown = max(UNKNOWN_FLOOR, 1 - float(weights.sum()))
    return weights, unknown, matches


def row_context(row, fitted):
    """Observation-only candidate pools exclude all relations and hard trust.

    Therefore changing a target cannot alter another field's context pool.
    Original values and all original observation alternatives remain present.
    """
    base = {**fitted['base'], 'relations': []}
    likelihoods = []; counts = []
    for k, column in enumerate(fitted['columns']):
        options = candidates(row, column, base, {})
        weights, unknown, _ = option_prior(row[column], options, column, fitted)
        likelihood = np.full(len(fitted['z']), unknown)
        useful = 0
        if row[column].strip():
            for q, weight in zip(options, weights):
                value = transform(q['value'], k, fitted)
                if value is None or weight <= 0:
                    continue
                useful += 1
                residual = (fitted['z'][:, k] - value) / fitted['sigma'][k]
                likelihood += weight * np.exp(-.5 * np.minimum(residual ** 2, 1400))
        else:
            likelihood[:] = 1
        likelihood[~np.isfinite(likelihood)] = 1
        likelihoods.append(np.log(np.maximum(likelihood, 1e-300)))
        counts.append(useful)
    return np.array(likelihoods).T, counts


def rank_row(row, cells, fitted):
    if fitted is None:
        return []
    contexts, counts = row_context(row, fitted)
    output = []
    for cell in cells:
        column = cell['column']
        if column not in fitted['columns']:
            continue
        j = fitted['columns'].index(column)
        others = [k for k in range(len(counts)) if k != j]
        context_log = contexts[:, others].sum(axis=1) + np.log(fitted['group_weights'])
        context_log -= context_log.max()
        prototype_weights = np.exp(context_log); prototype_weights /= prototype_weights.sum()
        weights, unknown, matches = option_prior(row[column], cell['options'], column, fitted)
        scored = []
        for q, prior, matched in zip(cell['options'], weights, matches):
            value = transform(q['value'], j, fitted)
            if value is None or prior <= 0:
                score = -1e6
            else:
                residual = (fitted['z'][:, j] - value) / fitted['sigma'][j]
                density = np.exp(-.5 * np.minimum(residual ** 2, 1400)) / fitted['sigma'][j]
                density[~np.isfinite(density)] = 0
                score = math.log(max(1e-300, prior * float(prototype_weights @ density)))
            scored.append({'value': q['value'], 'log_score': score, 'channels': matched,
                           'explainable': any(c != 'unknown' for c in matched)})
        ranked = sorted(scored, key=lambda q: (-q['log_score'], not equal(q['value'], row[column]), q['value']))
        top = ranked[0]; gap = top['log_score'] - ranked[1]['log_score'] if len(ranked) > 1 else 0
        output.append({'column': column, 'value': top['value'], 'gap': gap,
                       'unknown_log_odds': top['log_score'] - math.log(unknown / 6),
                       'context_fields': sum(counts[k] > 0 for k in others),
                       'explainable': top['explainable'], 'options': scored,
                       'prototype_weights': prototype_weights.tolist(), 'target_excluded': True})
    return output


_unconditional_fit = fit
_unconditional_prior = option_prior


def context_key(old, bounds):
    f = form(old)
    n = number(old)
    if f != 'numeric' or not bounds:
        return f
    return 'numeric_inside' if bounds[0] <= float(n) <= bounds[1] else 'numeric_outside'


def fit(records, base, mode='raw'):
    fitted = _unconditional_fit(records, base, mode)
    if fitted is None:
        return None
    counts = {f: Counter({c: 1.0 for c in CHANNELS})
              for f in ['blank', 'text', 'numeric_inside', 'numeric_outside']}
    bounds = {}
    for col in fitted['columns']:
        values = [float(number(r['truth'][col])) for r in records if number(r['truth'][col]) is not None]
        bounds[col] = [min(values), max(values)] if values else None
        for r in records:
            # In-sample min/max would put every normal teaching cell inside.
            # Leave its whole source out when learning observation conditions.
            other = [float(number(s['truth'][col])) for s in records
                     if s['group'] != r['group'] and number(s['truth'][col]) is not None]
            limits = [min(other), max(other)] if other else None
            old = r['dirty'][col]
            key = context_key(old, limits)
            matched = channels(old, r['truth'][col], col, fitted['codec'])
            for channel in matched:
                counts[key][channel] += 1 / len(matched)
    fitted['observation_bounds'] = bounds
    fitted['conditional_priors'] = {f: {c: float(v / sum(counts[f].values())) for c, v in counts[f].items()}
                                    for f in counts}
    fitted['prior_conditions_source_leave_group_out'] = True
    return fitted


def option_prior(old, options, column, fitted):
    key = context_key(old, fitted['observation_bounds'].get(column))
    prior = fitted['conditional_priors'][key]
    return _unconditional_prior(old, options, column, {**fitted, 'priors': {form(old): prior}})
