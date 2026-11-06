"""Explicit learned rule directions shared with both comparison arms."""
from .package import model_view, open_skill


def expanded_summary(path):
    _, base, _, _, _ = open_skill(path)
    summary = model_view(path)
    for col, view in summary['columns'].items():
        p = base['profiles'][col]
        view['observed_constant'] = p['values'][0] if len(p['values']) == 1 and p['nonblank'] >= 10 else None
        view['domain_kind'] = 'continuous_with_repeated_points' if any(int(k) > 0 for k in p['precision']) else 'observed_discrete'
        view['repeated_points_are_exhaustive_domain'] = False
        view['nonblank_learning_count'] = p['nonblank']
    for rule in summary['learned_transform_families']:
        rule['direction'] = ('observed = original * factor; inverse = observed / factor' if rule['kind'] == 'scale'
             else 'observed = original + offset; inverse = observed - offset' if rule['kind'] == 'shift'
             else 'observed = prefix + original_literal + suffix; inverse removes precisely those affixes')
    summary['other_observed_mechanisms'] = [{'kind': r['kind'], 'operation': r.get('operation'),
         'character_class': r.get('character_class'), 'fields': r['fields']}
         for r in base['rules'] if r['kind'] == 'edit' and r.get('operation') == 'replace']
    summary['instructions'] += ['Numeric fractions in continuous columns are valid; repeated integer points do not exhaust the value domain.',
         'Literal precision describes valid representations, not a request to round or quantize.',
         'Any empirical constant is a scoped learned fact; do not extend it to other columns.',
         'Restoring a guessed typical value is not exact repair. When alternatives cannot be justified, preserve the observed value.']
    return summary
