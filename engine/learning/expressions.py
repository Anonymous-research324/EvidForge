"""Case-spelling compatibility for logical operators; original v1 is retained.

Tokenize NAME tokens only. No new executable operations or external access are
added. The original model expression and its normalized spelling are auditable.
"""
import io
import tokenize
from .arithmetic import Expression as OriginalExpression


def normalize(source):
    aliases = {'AND': 'and', 'OR': 'or', 'NOT': 'not', 'TRUE': 'True', 'FALSE': 'False', 'true': 'True', 'false': 'False'}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in aliases:
            token = token._replace(string=aliases[token.string])
        tokens.append(token)
    return tokenize.untokenize(tokens)


class Expression(OriginalExpression):
    def __init__(self, source):
        if not isinstance(source, str):
            raise ValueError('Expected a source expression string')
        self.original_source = source
        try:
            self.normalized_source = normalize(source)
        except tokenize.TokenError as exc:
            raise ValueError('Incomplete candidate expression syntax') from exc
        super().__init__(self.normalized_source)
