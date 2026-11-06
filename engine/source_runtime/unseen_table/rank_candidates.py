"""One value selector; candidate sources are features, never a type probability gate."""
import random
from .common import equal, number
from .candidate_pool import candidates
from .features import candidate_features, detection_features
from .numeric_context import evidence
from .models import predict
from .pairwise_ranker import fit as fit_pairwise
from .row_state import probabilities
from .trusted_context import overlay
from .source_consensus import contexts as source_contexts

def fit(records, fitted, seed, contexts=None, states=None):
    rng = random.Random(seed)
    x, pairs = [], []
    peers = [{} for _ in records]
    if fitted.get('source_consensus'):
        blocks = {}
        for index, record in enumerate(records):
            variant = record['row_key'].rsplit(':', 1)[-1] if record.get('synthetic') else 'original'
            blocks.setdefault((record['group'], variant), []).append(index)
        for ids in blocks.values():
            for index, peer in zip(ids, source_contexts([records[i]['dirty'] for i in ids], fitted)):
                peers[index] = peer
    for index, r in enumerate(records):
        ctx = dict(contexts[index]) if contexts is not None else evidence(r['dirty'], fitted, exclude_group=r['group'])
        ctx['states'] = states[index] if states is not None else {}
        for col, truth in r['truth'].items():
            options = candidates(r['dirty'], col, fitted, ctx['trusted'], peers[index].get(col))
            positives = [p for p in options if equal(p['value'], truth)]
            if not positives:
                continue  # Absent exact answers are not fabricated training positives.
            negatives = [p for p in options if not equal(p['value'], truth)]
            mandatory = [p for p in negatives if p['kind'] == 'keep' or any(k not in ('edit', 'domain') for k in p['kinds'])]
            rest = [p for p in negatives if p not in mandatory]
            rng.shuffle(rest)
            hard = sorted(rest, key=lambda p: abs(float(number(p['value']) or 0) - float(number(truth) or 0)))[:4]
            pool = mandatory + hard + [p for p in rest if p not in hard][:max(0, 14 - len(mandatory) - len(hard))]
            at = len(x)
            x.append(candidate_features(r['dirty'], col, positives[0], fitted, ctx, r['group']))
            for option in pool:
                negative = len(x)
                x.append(candidate_features(r['dirty'], col, option, fitted, ctx, r['group']))
                pairs.append((at, negative, (0.25 if r.get('synthetic') else 1.0) / max(1, len(pool))))
    return fit_pairwise(x, pairs, seed=seed)

def predict_stage(rows, fitted, selector=None, trusted=None, stage=1):
    x = []
    dx = []
    cells = []
    lengths = []
    source = source_contexts(rows, fitted)
    states = probabilities(rows, fitted, trusted=trusted)
    for i, row in enumerate(rows):
        ctx = evidence(row, fitted, None if trusted is None else trusted[i])
        ctx['states'] = states[i]
        for col in fitted['declaration']['targets']:
            options = candidates(row, col, fitted, ctx['trusted'], source[i].get(col))
            x.extend((candidate_features(row, col, p, fitted, ctx) for p in options))
            dx.append(detection_features(row, col, fitted, ctx))
            lengths.append(len(options))
            cells.append({'row': i, 'column': col, 'original': row[col], 'options': options, 'context_estimate': ctx['columns'].get(col), 'context_proofs': ctx['trusted']['proofs'], 'profile_kind': fitted['value_modes'][col]['kind'], 'stage': stage})
    scores = predict(selector or fitted['selector'], x)
    errors = predict(fitted['detector'], dx)
    at = 0
    for c, length, error in zip(cells, lengths, errors):
        for p, s in zip(c['options'], scores[at:at + length]):
            p['score'] = float(s)
        ranked = sorted(c['options'], key=lambda p: (-p['score'], p['kind'] != 'keep', p['value']))
        best = ranked[0]
        c['decision_vector'] = x[at + c['options'].index(best)]
        c['state_probabilities'] = states[c['row']].get(c['column'], {})
        second = ranked[1]['score'] if length > 1 else 0.0
        priority = ['source', 'relation', 'constant', 'affix', 'strip', 'casefold', 'scale', 'shift', 'mapping', 'domain', 'edit']
        if best['kind'] != 'keep':
            best['kind'] = next((k for k in priority if k in best['kinds']))
        old = number(c['original'])
        low = number(fitted['profiles'][c['column']]['min'])
        high = number(fitted['profiles'][c['column']]['max'])
        c.update(top=best, margin=best['score'] - second, tied=length > 1 and best['score'] - second <= 1e-10, error_probability=float(error), candidate_count=length, precision_known=True, old_domain='inside' if old is not None and low is not None and (low <= old <= high) else 'outside' if old is not None else 'text_or_blank', old_form='integer' if old is not None and old == old.to_integral() else 'decimal' if old is not None else 'blank' if not c['original'].strip() else 'text')
        at += length
    return cells

def predict_rows(rows, fitted):
    from .validate_skill import permitted
    from .decision_check import apply
    first = predict_stage(rows, fitted, selector=fitted.get('stage1_selector', fitted['selector']))
    apply(first, fitted.get('stage1_decision_model'))
    if 'stage1_policy' not in fitted:
        return first
    grouped = [[] for _ in rows]
    for c in first:
        grouped[c['row']].append(c)
    trust = [overlay(row, fitted, grouped[i], fitted['stage1_policy']) for i, row in enumerate(rows)]
    second = predict_stage(rows, fitted, trusted=trust, stage=2)
    for a, b in zip(first, second):
        b['stage1_was_permitted'] = permitted(a, fitted['stage1_policy'])
        b['stage1_agrees'] = equal(a['top']['value'], b['top']['value'])
        b['stage1_disagrees'] = not b['stage1_agrees']
        b['stage1_proposal'] = a['top']['value']
        b['revision'] = 'withdraw' if b['stage1_was_permitted'] and b['top']['kind'] == 'keep' else 'replace' if b['stage1_was_permitted'] and b['stage1_disagrees'] else 'agree_or_new'
        # Original row is immutable. No stage1 flag bypasses final qualification.
    return apply(second, fitted.get('decision_model'))
