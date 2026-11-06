"""Conservative bootstrap and cross-checked overlays; range alone grants no trust."""
from .common import number, equal
from .learn_transforms import apply_rule, input_pattern

def bootstrap(row, fitted):
    values = {}
    trust = {}
    proofs = {}
    for col, p in fitted['profiles'].items():
        options = []
        for rule in fitted['rules']:
            if col in rule['fields'] and rule['kind'] in ('affix', 'strip', 'casefold', 'constant') and (rule['groups'] >= 3):
                if rule['kind'] != 'constant' and number(row[col]) is not None:
                    continue
                v = apply_rule(rule, row, col)
                if v is not None and v.strip():
                    options.append((v, rule['id']))
        if options and all((equal(v, options[0][0]) for v, _ in options)):
            v = options[0][0]
            if not p['numeric'] or number(v) is not None:
                values[col] = v
                trust[col] = 0.99
                proofs[col] = {'kind': 'unambiguous_learned_rule', 'rules': [i for _, i in options]}
                continue
        condition = fitted.get('input_conditions', {}).get(input_pattern(row[col]), {})
        if condition.get('normal', 0) >= 0.98 and condition.get('groups', 0) >= 3 and (condition.get('count', 0) >= 5):
            values[col] = row[col]
            trust[col] = condition['normal']
            proofs[col] = {'kind': 'learning_supported_keep'}
    for col in fitted['declaration'].get('visible_context_fields', []):
        if col not in fitted['declaration']['entity_fields']:
            values[col] = row[col]
            trust[col] = 1.0
            proofs[col] = {'kind': 'declared_visible_context'}
    return {'values': values, 'trust': trust, 'proofs': proofs}

def overlay(row, fitted, cells, policy):
    from .validate_skill import permitted
    context = bootstrap(row, fitted)
    for c in cells:
        if permitted(c, policy):
            col = c['column']
            context['values'][col] = c['top']['value']
            from .validate_skill import region
            regional = policy.get('by_region', {}).get(region(c), {})
            context['trust'][col] = regional.get('empirical_precision', c.get('repair_confidence', c['top']['score']))
            context['proofs'][col] = {'kind': 'held_group_qualified_stage1', 'candidate_id': c['top']['candidate_id']}
    return context
