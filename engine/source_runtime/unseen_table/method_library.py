"""Reusable strategies, never dataset-specific answers or historical mutation."""
from pathlib import Path
from .common import PACKAGE, load, fingerprint

def inventory():
    return [load(p) for p in sorted((PACKAGE / 'methods').glob('*.json'))]

def applicable(declaration, profiles):
    methods = ['value-transform', 'field-relation', 'value-mode']
    if declaration.get('source_field') and declaration.get('entity_fields'):
        methods.append('source-consensus')
    templates = inventory()
    return {'methods': methods, 'template_sha256': fingerprint(templates), 'parameters_source': 'current learning groups only', 'numeric_targets': [c for c, p in profiles.items() if p['numeric']]}
