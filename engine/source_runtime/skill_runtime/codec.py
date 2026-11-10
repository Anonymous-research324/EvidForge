"""Learn literal damage mechanisms and supported numerical representations."""
from collections import defaultdict
from decimal import Decimal
from unseen_table.common import number,canonical,equal
from unseen_table.learn_transforms import infer,apply_rule

def fit(records,schema):
    base=infer(records,schema);groups=defaultdict(set);point_aliases=defaultdict(set);values=[]
    for r in records:
        for col,v in r['truth'].items():
            n=number(v)
            if n is None:continue
            values.append(n);groups[max(0,-n.as_tuple().exponent)].add(r['group'])
            if n==n.to_integral():point_aliases[canonical(v)].add(v)
    if not values:return {'rules':[],'precisions':[],'point_aliases':{},'low':None,'high':None}
    rules=[r for r in base['rules'] if r['kind'] in ('scale','shift') or r['kind']=='affix' and any(c.isalpha() for c in r['prefix']+r['suffix'])];grammar=[]
    for rule in rules:
        support=[]
        for r in records:
            for col,v in r['truth'].items():
                if col not in rule['fields'] or equal(r['dirty'][col],v) or not equal(apply_rule(rule,r['dirty'],col),v):continue
                n=number(v);observed=format(n*Decimal(rule['factor']),'f') if rule['kind']=='scale' and n is not None else format(n+Decimal(rule['offset']),'f') if rule['kind']=='shift' and n is not None else rule['prefix']+v+rule['suffix'] if rule['kind']=='affix' else None
                support.append(observed==r['dirty'][col])
        if support and sum(support)/len(support)>=.98:grammar.append(rule)
    return {'rules':grammar,'precisions':[p for p,g in groups.items() if len(g)>=3],'point_aliases':{k:sorted(v) for k,v in point_aliases.items()},'low':min(values),'high':max(values)}

def encode(codec):return {**codec,'low':str(codec['low']) if codec['low'] is not None else None,'high':str(codec['high']) if codec['high'] is not None else None}
def decode(codec):return {**codec,'low':number(codec['low']) if codec['low'] is not None else None,'high':number(codec['high']) if codec['high'] is not None else None}
def aliases(value,codec):
    n=number(value)
    if n is None:return [value]
    if n==n.to_integral():return codec['point_aliases'].get(canonical(value),[value])
    return [v for p in codec['precisions'] if number(v:=format(n,f'.{p}f'))==n]

def possible(old,new,col,codec):
    if equal(old,new) or not old.strip():return True
    n=number(new)
    if n is not None and codec['low'] is not None and not codec['low']<=n<=codec['high']:return False
    for rule in codec['rules']:
        if col not in rule['fields']:continue
        for v in aliases(new,codec):
            n=number(v);observed=format(n*Decimal(rule['factor']),'f') if rule['kind']=='scale' and n is not None else format(n+Decimal(rule['offset']),'f') if rule['kind']=='shift' and n is not None else rule['prefix']+v+rule['suffix'] if rule['kind']=='affix' else None
            if observed==old:return True
    for v in aliases(new,codec):
        differences=[(a,b) for a,b in zip(old,v) if a!=b]
        if len(old)==len(v) and len(differences)==1 and all(c.isdigit() for c in differences[0]):return True
    return False
