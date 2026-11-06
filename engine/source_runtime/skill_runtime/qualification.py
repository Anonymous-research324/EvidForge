"""Qualify each supplemental executable clause on source-held teaching cases."""
from collections import defaultdict
from unseen_table.common import equal

NEW_CLAUSES = ('literal_unique', 'typed_integer', 'typed_blank', 'row_grid')


def qualify(held_decisions):
    """Input pairs are (decisions, held records); they are authorized development.

    A candidate-count or numerical-pattern argument is insufficient on its own.
    Failed clauses are disabled even if the aggregate old/new programme looks
    useful. Small untested clauses remain closed instead of inheriting trust.
    """
    stages = defaultdict(lambda: {'changes':0, 'exact':0, 'normal_damage':0, 'groups':set()})
    normal = 0
    for decisions, records in held_decisions:
        for p in decisions:
            record = records[p['row']]; truth = record['truth'][p['field']]
            error = not equal(p['before'], truth); normal += not error
            if p['stage'] not in NEW_CLAUSES or equal(p['after'],p['before']) or equal(p['after'],p['base']):
                continue
            s = stages[p['stage']]; s['changes'] += 1
            s['exact'] += error and equal(p['after'],truth)
            s['normal_damage'] += not error; s['groups'].add(record['group'])
    report = {}; enabled = {}
    for stage in NEW_CLAUSES:
        s = stages[stage]; precision = s['exact']/s['changes'] if s['changes'] else None
        damage = s['normal_damage']/normal if normal else None
        passed = bool(s['changes']>=5 and len(s['groups'])>=3 and normal>=10
                      and precision is not None and precision>=.98 and damage<=.005)
        enabled[stage] = passed
        report[stage] = {'enabled':passed, 'checked_proposals':s['changes'],
             'precision':precision, 'normal_damage':damage, 'checked_source_groups':len(s['groups']),
             'normal_controls':normal, 'reason':'source-held clause qualified' if passed else 'insufficient or inaccurate clause evidence',
             'scope':'learning calibration; final checkpoint still needs independent evaluation'}
    return enabled, report
