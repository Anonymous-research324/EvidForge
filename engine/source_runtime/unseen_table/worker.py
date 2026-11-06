"""Whitelisted CPU actions executed under the parent workflow's hard deadline."""
import sys, time
from .common import load, dump, validate_declaration

def main():
    request, result = sys.argv[1:]
    data = load(request)
    started = time.monotonic()
    if data['action'] == 'fit':
        from .learn_skill import fit
        validate_declaration(data['declaration'], list(data['declaration']['fields']))
        output = fit(data['records'], data['declaration'], seed=data['seed'], extra_rules=data['extra_rules'])
    elif data['action'] == 'predict':
        from .rank_candidates import predict_rows
        from .save_skill import validate
        output = predict_rows(data['rows'], validate(data['skill']))
    elif data['action'] == 'check':
        from .rank_candidates import predict_rows
        fitted = data['fitted']
        validate_declaration(fitted['declaration'], list(fitted['declaration']['fields']))
        output = predict_rows(data['rows'], fitted)
    else:
        raise ValueError('不支持的本地任务')
    dump(result, output)
    print({'action': data['action'], 'seconds': time.monotonic() - started}, flush=True)
if __name__ == '__main__':
    main()
