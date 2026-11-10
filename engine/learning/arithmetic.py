"""Interpret model-proposed arithmetic/Boolean rules without arbitrary code.

Rules can combine measured candidate evidence in new ways. They cannot read
files, source rows, identifiers, labels, references, attributes or subscripts.
The interpreter never uses eval/exec, and rejects unbounded syntax/numbers.
"""
import ast
import math
import operator

FIELDS = {'is_shift', 'noisy_raw_unknown_odds', 'forward_explainable', 'is_affix', 'noisy_raw_log_score', 'score_gap', 'noisy_signed_unknown_odds', 'old_numeric', 'old_blank', 'is_baseline', 'is_keep', 'rank_score', 'baseline_changed', 'is_scale', 'is_edit', 'noisy_raw_context_fields', 'noisy_signed_log_score', 'noisy_signed_context_fields'}
CONTEXT_FIELDS = {'context_nb_probability','context_tree_probability','context_model_agrees','context_domain_cardinality','context_observed_fraction'}
FIELDS |= CONTEXT_FIELDS
FUNCTIONS = {'abs': abs, 'min': min, 'max': max, 'log1p': math.log1p, 'sqrt': math.sqrt}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv}
COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class Expression:
    def __init__(self, source):
        if not isinstance(source, str) or not 1 <= len(source) <= 1024:
            raise ValueError('Expression must be 1..1024 characters')
        self.source = source
        self.tree = ast.parse(source, mode='eval').body
        nodes = list(ast.walk(self.tree))
        if len(nodes) > 160:
            raise ValueError('Expression exceeds bounded complexity')
        self._check(self.tree, 0)

    def _check(self, node, depth):
        if depth > 16:
            raise ValueError('Expression exceeds bounded depth')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, bool) or not math.isfinite(float(node.value)) or abs(node.value) > 1000:
                raise ValueError('Only bounded finite numeric/Boolean constants are allowed')
        elif isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise ValueError('Unknown measured candidate feature: ' + node.id)
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            self._check(node.left, depth + 1); self._check(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub, ast.Not):
            self._check(node.operand, depth + 1)
        elif isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
            for value in node.values:
                self._check(value, depth + 1)
        elif isinstance(node, ast.Compare) and all(type(op) in COMPARE for op in node.ops):
            self._check(node.left, depth + 1)
            for value in node.comparators:
                self._check(value, depth + 1)
        elif isinstance(node, ast.IfExp):
            for value in (node.test, node.body, node.orelse):
                self._check(value, depth + 1)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            if node.keywords or not 1 <= len(node.args) <= 4:
                raise ValueError('Bounded positional numeric functions only')
            if node.func.id in ('abs', 'log1p', 'sqrt') and len(node.args) != 1:
                raise ValueError('Unary numerical function requires one argument')
            for value in node.args:
                self._check(value, depth + 1)
        else:
            raise ValueError('Unsupported executable syntax: ' + type(node).__name__)

    def _value(self, node, features):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return features[node.id]
        if isinstance(node, ast.BinOp):
            return BINARY[type(node.op)](self._value(node.left, features), self._value(node.right, features))
        if isinstance(node, ast.UnaryOp):
            value = self._value(node.operand, features)
            return not value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._value(v, features) for v in node.values)
            return any(self._value(v, features) for v in node.values)
        if isinstance(node, ast.Compare):
            left = self._value(node.left, features)
            for op, right in zip(node.ops, node.comparators):
                right = self._value(right, features)
                if not COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._value(node.body if self._value(node.test, features) else node.orelse, features)
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*(self._value(v, features) for v in node.args))
        raise AssertionError('Expression escaped validation')

    def __call__(self, features):
        if set(features) != FIELDS or any(not math.isfinite(float(v)) for v in features.values()):
            raise ValueError('Expected exact finite feature interface')
        try:
            value = self._value(self.tree, features)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            raise ValueError('Invalid numeric hypothesis on source-held input') from exc
        if not math.isfinite(float(value)) or abs(float(value)) > 1e6:
            raise ValueError('Hypothesis produced an invalid or unbounded result')
        return value
