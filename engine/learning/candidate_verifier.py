"""Measure generated programs against an explicit source-held baseline.

Incremental repairs and reversals need their own permission: easy baseline
repairs cannot hide mistakes among new proposals. Outer sources are not used.
"""
from collections import defaultdict
import numpy as np
from .expressions import Expression
from .environment import RUNTIME
from unseen_table.common import equal


def rank(frames, program):
    if 'score_gap' in program['score_expression']:
        raise ValueError('score_gap cannot be a scoring input')
    score = Expression(program['score_expression']); accept = Expression(program['accept_expression'])
    output = []
    for frame in frames:
        choices = sorted([(float(score(q['features'])), q) for q in frame['options']],
                         key=lambda x: (-x[0], -x[1]['features']['is_baseline'], -x[1]['features']['is_keep'], x[1]['value']))
        s, q = choices[0]; gap = s - choices[1][0] if len(choices) > 1 else 0
        f = {**q['features'], 'score_gap': gap}
        region = 'blank' if not frame['original'].strip() else 'numeric' if f['old_numeric'] else 'text'
        action = 'withdraw' if f['is_keep'] else 'repair'
        output.append({'frame': frame, 'value': q['value'], 'score': s, 'gap': gap,
                       'accept': bool(accept(f)) and gap > 1e-10 and bool(f['forward_explainable']),
                       'channel': region + '/' + action,
                       'differs_from_baseline': not equal(q['value'], frame['baseline_after_local'])})
    return output


def permitted(q, policy):
    gate = policy.get(q['channel'])
    return bool(q['accept'] and q['differs_from_baseline'] and gate and
                q['score'] >= gate['score_threshold'] and q['gap'] >= gate['margin_threshold'])


def decisions(ranked, policy):
    return [{'row': q['frame']['row'], 'field': q['frame']['column'], 'group': q['frame']['group'],
             'before': q['frame']['original'], 'after': q['value'] if permitted(q, policy) else q['frame']['baseline_after_local'],
             'baseline': q['frame']['baseline_after_local'], 'expected_local_only': q['frame']['expected_local_only'],
             'generated_change': permitted(q, policy), 'channel': q['channel']} for q in ranked]


def measure(ds):
    counts = defaultdict(int); groups = set()
    for d in ds:
        error = not equal(d['before'], d['expected_local_only']); changed = not equal(d['before'], d['after'])
        counts['errors' if error else 'normal'] += 1
        counts['changes'] += changed
        counts['exact'] += changed and error and equal(d['after'], d['expected_local_only'])
        counts['normal_damage'] += changed and not error
        if d['generated_change']:
            counts['incremental_actions'] += 1; groups.add(d['group'])
            counts['incremental_exact_value'] += equal(d['after'], d['expected_local_only'])
            counts['added_exact'] += error and equal(d['after'], d['expected_local_only']) and not equal(d['baseline'], d['expected_local_only'])
            counts['lost_exact'] += error and equal(d['baseline'], d['expected_local_only']) and not equal(d['after'], d['expected_local_only'])
    p = counts['exact'] / max(1, counts['changes']); r = counts['exact'] / max(1, counts['errors'])
    return {**counts, 'P': p, 'R': r, 'F1': 2*p*r/(p+r) if p+r else 0,
            'damage': counts['normal_damage']/max(1, counts['normal']), 'incremental_groups': len(groups)}


def fit_permissions(ranked):
    policy = {}; total_normal = sum(equal(q['frame']['original'], q['frame']['expected_local_only']) for q in ranked)
    buckets = defaultdict(list)
    for q in ranked:
        if q['accept'] and q['differs_from_baseline']: buckets[q['channel']].append(q)
    for channel, candidates in buckets.items():
        best = None
        thresholds = sorted(set(float(x) for x in np.quantile([q['score'] for q in candidates], [0,.1,.25,.5,.75,.9,.95,1])))
        for threshold in thresholds:
            for gap in [0,.01,.05,.1,.2,.5,1,2,4,8]:
                qs = [q for q in candidates if q['score'] >= threshold and q['gap'] >= gap]
                groups = {q['frame']['group'] for q in qs}
                if len(qs) < 8 or len(groups) < 4: continue
                good = sum(equal(q['value'], q['frame']['expected_local_only']) for q in qs)
                added = sum(not equal(q['frame']['baseline_after_local'], q['frame']['expected_local_only'])
                            and equal(q['value'], q['frame']['expected_local_only']) for q in qs)
                lost = sum(equal(q['frame']['baseline_after_local'], q['frame']['expected_local_only'])
                           and not equal(q['value'], q['frame']['expected_local_only']) for q in qs)
                new_damage = sum(equal(q['frame']['original'], q['frame']['expected_local_only'])
                                 and equal(q['frame']['baseline_after_local'], q['frame']['original'])
                                 and not equal(q['value'], q['frame']['original']) for q in qs)
                if good/len(qs) < .98 or new_damage or added <= lost: continue
                key = (added-lost, good, threshold, gap)
                if best is None or key > best[0]:
                    best = (key, {'score_threshold':threshold, 'margin_threshold':gap,
                                  'checked_incremental':len(qs), 'groups':len(groups), 'incremental_value_P':good/len(qs)})
        if best: policy[channel] = best[1]
    return policy


def evaluate(frames, program):
    ranked = rank(frames, program); policy = fit_permissions(ranked)
    result = measure(decisions(ranked, policy)); baseline = measure(decisions(ranked, {}))
    eligible = (result.get('incremental_actions',0) >= 8 and result['incremental_groups'] >= 4
                and result.get('incremental_exact_value',0)/max(1,result.get('incremental_actions',0)) >= .98
                and result['normal_damage'] <= baseline['normal_damage'] and result['F1'] > baseline['F1'] + 1e-12)
    diagnostics = {}
    for channel in sorted({q['channel'] for q in ranked}):
        qs = [q for q in ranked if q['channel'] == channel and q['accept'] and q['differs_from_baseline']]
        diagnostics[channel] = {'raw_incremental_actions':len(qs),
            'source_groups':len({q['frame']['group'] for q in qs}),
            'raw_exact_value_P':sum(equal(q['value'],q['frame']['expected_local_only']) for q in qs)/len(qs) if qs else None,
            'raw_added_exact':sum(not equal(q['frame']['baseline_after_local'],q['frame']['expected_local_only']) and
                equal(q['value'],q['frame']['expected_local_only']) for q in qs),
            'raw_lost_exact':sum(equal(q['frame']['baseline_after_local'],q['frame']['expected_local_only']) and
                not equal(q['value'],q['frame']['expected_local_only']) for q in qs),
            'raw_new_normal_damage':sum(equal(q['frame']['original'],q['frame']['expected_local_only']) and
                equal(q['frame']['baseline_after_local'],q['frame']['original']) and
                not equal(q['value'],q['frame']['original']) for q in qs),
            'source_permission_fitted':channel in policy,
            'requires': '>=8 incremental actions, >=4 groups, >=98% exact values, zero new normal damage, positive net gain'}
    return {'program':program, 'policy':policy, 'metrics':result, 'baseline_metrics':baseline,
            'before_permission_diagnostics':diagnostics,
            'eligible_for_research_compile':eligible,
            'scope':'inner source-held selection; outer sources still excluded',
            'parameters_and_permission_require_independent_check':True}
