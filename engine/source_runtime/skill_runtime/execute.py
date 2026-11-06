"""Single-pass repair with absolute evidence and native-point preservation.

V6 models stay intact. New point proposals cannot acquire confidence merely by
discarding the unknown/continuous class. Qualified base decisions keep their
own learned checker, rather than being checked against an unrelated state score.
"""
from collections import defaultdict
from unseen_table.common import equal, number, canonical
from unseen_table.rank_candidates import predict_rows
from unseen_table.validate_skill import permitted
from .codec import possible
from .row_family import prove


POLICY = {'integer_q': .9, 'blank_q': .95, 'blank_mass': .95,
          'grid_support': 6, 'grid_fraction': .7, 'grid_distinct': 6,
          'grid_mode': 'residual', 'absolute_point_q': .95,
          'native_guard': 'learned_points', 'native_grid_support': 14,
          'native_grid_fraction': .85, 'native_grid_distinct': 10,
          'enabled_new_clauses': {}}


def propose_from_cells(rows, cells, base, codec, family, policy):
    blocks = defaultdict(list)
    for c in cells:
        blocks[c['row']].append(c)
    proofs = {i: prove(row, base['declaration']['targets'], blocks[i], codec, family)
              for i, row in enumerate(rows)}
    native_points = set(base.get('row_state', {}).get('points', []))
    result = []
    for c in cells:
        i, col = c['row'], c['column']; old = rows[i][col]; n = number(old)
        after = c['top']['value'] if permitted(c, base['policy']) else old
        baseline = after; stage = 'qualified_base'; before_guard = after
        numeric = [p for p in c['options'] if not equal(old, p['value'])
                   and number(p['value']) is not None and possible(old, p['value'], col, codec)]
        outside = n is not None and codec['low'] is not None and not codec['low'] <= n <= codec['high']
        state = c.get('state_probabilities', {})
        if equal(after, old):
            if outside and len(numeric) == 1:
                after = numeric[0]['value']; stage = 'literal_unique'
            elif not old.strip() or outside and '.' not in old:
                points = {v: float(q) for v, q in state.items()
                          if v != '__continuous_or_unseen__' and number(v) is not None
                          and possible(old, v, col, codec)}
                mass = sum(points.values())
                if mass:
                    value = max(points, key=lambda v: (points[v], v))
                    conditional = points[value] / mass
                    integer = bool(old.strip() and conditional >= policy['integer_q'])
                    blank = bool(not old.strip() and conditional >= policy['blank_q']
                          and mass >= policy['blank_mass'] and equal(c['top']['value'], value)
                          and c['top']['score'] >= .95)
                    if integer or blank:
                        after = value; stage = 'typed_integer' if integer else 'typed_blank'
        grid = proofs[i][col]
        if grid['value'] is not None and grid['support'] >= policy['grid_support'] and grid['fraction'] >= policy['grid_fraction'] and grid['distinct'] >= policy['grid_distinct'] and equal(after, old):
            after = grid['value']; stage = 'row_grid'
        before_guard = after
        constant = len(base['profiles'][col]['values']) == 1
        absolute = max([float(q) for v, q in state.items() if equal(v, after)], default=0)
        strong_grid = (grid['value'] is not None and grid['support'] >= policy['native_grid_support']
              and grid['fraction'] >= policy['native_grid_fraction'] and grid['distinct'] >= policy['native_grid_distinct'])
        native = (n is not None and codec['low'] is not None and codec['low'] <= n <= codec['high']) if policy['native_guard'] == 'all_numeric' else canonical(old) in native_points
        reason = 'own_source_calibrated_base_or_supported_new_rule'
        if native and not constant:
            after = grid['value'] if strong_grid else old
            reason = 'independent_row_proof' if strong_grid else 'preserve_supported_native_point'
        elif not constant and stage in ('typed_integer', 'typed_blank') and len({canonical(p['value']) for p in numeric}) != 1:
            if absolute < policy['absolute_point_q']:
                after = old; reason = 'insufficient_absolute_point_evidence'
        # A discovered pattern is only a proposal. Source-held checks must prove
        # this particular clause useful on the NEW table before it can execute.
        if stage != 'qualified_base' and not policy.get('enabled_new_clauses', {}).get(stage, False) and not equal(after,baseline):
            after = baseline; reason = 'new_clause_failed_source_cross_check'
        result.append({'row': i, 'field': col, 'before': old, 'after': after,
              'base': baseline, 'v6': baseline, 'stage': stage, 'before_guard': before_guard,
              'guard_reason': reason, 'point_probability_absolute': absolute,
              'grid_proof': grid, 'detection_score': c['error_probability']})
    return result


def propose(rows, base, codec, family, policy):
    return propose_from_cells(rows, predict_rows(rows, base), base, codec, family, policy)
