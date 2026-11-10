"""Learn bounded exact rounded multi-field relationships from teaching values.

An interval contains ALL coefficients consistent with every source record's
literal precision. Runtime emits a value only when the complete interval gives
the SAME rounded literal. A fitted average or regression mean is insufficient.
No field names, scientific constants, dataset metadata or test values are fixed.
"""
from decimal import Decimal,ROUND_HALF_EVEN,localcontext
from collections import Counter,defaultdict
from itertools import permutations
from unseen_table.common import number,equal,canonical

OPS=['ratio','ratio_square','product','sum']


def expression(a,b,op):
    if op=='ratio':return a/b if b else None
    if op=='ratio_square':return a/(b*b) if b else None
    if op=='product':return a*b
    if op=='sum':return a+b
    raise ValueError('Unknown bounded arithmetic operation')


def fit(records,declaration):
    groups={r['group'] for r in records}
    if len(groups)<20:return []
    columns=declaration['targets'];values={c:[number(r['truth'][c]) for r in records] for c in columns}
    columns=[c for c in columns if all(n is not None for n in values[c]) and len(set(values[c]))>=8]
    rules=[]
    with localcontext() as context:
        context.prec=40
        for target in columns:
            precision=Counter(max(0,-n.normalize().as_tuple().exponent) for n in values[target])
            supported=[p for p,n in precision.items() if n>=2]
            if not supported or max(supported)>8:continue
            q=max(supported);unit=Decimal(1).scaleb(-q);half=unit/2
            if max(values[target])-min(values[target])<100*unit:continue
            for left,right in permutations([c for c in columns if c!=target],2):
                for op in OPS:
                    if op in ['product','sum'] and left>right:continue
                    lo=hi=None;valid=True
                    for y,a,b in zip(values[target],values[left],values[right]):
                        x=expression(a,b,op)
                        if not x:valid=False;break
                        lower,upper=sorted([(y-half)/x,(y+half)/x])
                        lo=lower if lo is None else max(lo,lower);hi=upper if hi is None else min(hi,upper)
                        if lo>=hi:valid=False;break
                    if not valid:continue
                    center=(lo+hi)/2
                    if abs(center)<Decimal('0.000001') or abs(center)>Decimal('1000000') or (hi-lo)/abs(center)>Decimal('.005'):continue
                    # Equality must be possible under one shared interval, and
                    # enough training targets must be invariant throughout it.
                    spec={'target':target,'left':left,'right':right,'operation':op,'precision':q,
                          'coefficient_interval':[str(lo),str(hi)],'training_groups':sorted(groups),
                          'coefficient_is_source_learned':True,'target_self_excluded':True}
                    stable=sum(predict(r['truth'],spec) is not None for r in records)
                    if stable>=max(10,int(.8*len(records))):rules.append(spec)
    return rules


def predict(trusted,spec):
    a=number(trusted.get(spec['left'],''));b=number(trusted.get(spec['right'],''))
    if a is None or b is None:return None
    with localcontext() as context:
        context.prec=40;x=expression(a,b,spec['operation'])
        if x is None:return None
        lo,hi=[Decimal(v) for v in spec['coefficient_interval']]
        # Both fitting endpoints are open: they are quantization boundaries.
        epsilon=max(abs(lo),abs(hi),Decimal(1))*Decimal('1e-30');lo+=epsilon;hi-=epsilon
        unit=Decimal(1).scaleb(-spec['precision'])
        lower=(lo*x).quantize(unit,rounding=ROUND_HALF_EVEN);upper=(hi*x).quantize(unit,rounding=ROUND_HALF_EVEN)
        return str(lower) if lower==upper else None


def hypotheses(row,target,trusted,rules):
    choices=defaultdict(list)
    for spec in rules:
        if spec['target']!=target or target in [spec['left'],spec['right']]:continue
        value=predict(trusted,spec)
        if value is not None:choices[canonical(value)].append({'value':value,'rule':spec})
    if len(choices)!=1:return None
    proof=next(iter(choices.values()))
    if equal(row[target],proof[0]['value']):return None
    return {'value':proof[0]['value'],'supporting_relations':len(proof),'rules':proof,
            'whole_coefficient_interval_agrees':True,'predicted_mean_not_used':True,'target_self_excluded':True}
