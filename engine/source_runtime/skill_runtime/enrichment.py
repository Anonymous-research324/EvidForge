"""Execute an additive, finite skill; the sealed baseline is read-only.

The model selects source-checked operator IDs during learning. Numerical
parameters come from the authorized learning pool, not from a dataset name.
Original observed literals remain the anchor and disagreements abstain.
"""
from pathlib import Path
from collections import defaultdict
from unseen_table.common import load,dump,sha256,equal,canonical
from unseen_table.rank_candidates import predict_rows
from unseen_table.trusted_context import bootstrap
from .package import open_skill
from .execute import propose_from_cells
from .refinement import bounded_hypotheses
from .conditional_context import fit as fit_conditional,hypotheses as conditional_hypotheses
from .approximate_relations import fit as fit_relations,hypotheses as relation_hypotheses
from .uncertain_context import compare
from .mixture_context import fit as fit_mixture,hypotheses as mixture_hypotheses
from .learning_tools import DESCRIPTIONS
from .exact_relations import fit as fit_exact,hypotheses as exact_hypothesis


def fit(records,declaration,operators):
    if not operators or len(set(operators))!=len(operators) or not set(operators)<=set(DESCRIPTIONS):
        raise ValueError('Only unique source-checked bounded operator IDs can be fitted')
    conditional=any(x.startswith(('conditional_','uncertain_')) for x in operators)
    mixtures={}
    for name in operators:
        if name.startswith('mixture_'):
            mode='signed_log' if 'signed_log' in name else 'raw';clusters=int(name.split('_k')[1][0])
            key=f'{mode}_k{clusters}'
            if key not in mixtures:mixtures[key]=fit_mixture(records,declaration,mode,clusters)
    return {'operators':operators,'training_groups':sorted({r['group'] for r in records}),
            'conditional_models':{m:fit_conditional(records,declaration,m) for m in ['raw','signed_log']} if conditional else {},
            'soft_relations':fit_relations(records,declaration) if any(x.startswith('soft_relation_') for x in operators) else [],
            'mixture_models':mixtures,
            'source_exact_rules':fit_exact(records,declaration) if 'source_exact_arithmetic' in operators else [],
            'estimated_means_not_written':True,'immutable_original_anchors':True}


def propose_from_existing(rows,cells,decisions,base,codec,learner):
    operators=set(learner['operators']);blocks=defaultdict(list)
    for cell in cells:blocks[cell['row']].append(cell)
    lookup={(d['row'],d['field']):d for d in decisions};offers=defaultdict(list)
    for i,row in enumerate(rows):
        trusted=dict(bootstrap(row,base)['values'])
        for cell in blocks[i]:
            d=lookup[(i,cell['column'])]
            if not equal(d['before'],d['after']):trusted[d['field']]=d['after']
        uncertain={}
        for name in sorted(operators):
            if name.startswith('uncertain_'):
                rounds=3 if '_r3_' in name else 1
                uncertain[name]=compare(row,blocks[i],trusted,learner['conditional_models']['signed_log'],codec,rounds=rounds)
        for cell in blocks[i]:
            col=cell['column'];d=lookup[(i,col)]
            if not equal(d['before'],d['after']):continue
            proposals={name:{'value':value,'method':name} for name,value in bounded_hypotheses(row,cell,base,codec).items()}
            if learner['soft_relations']:
                proposals.update(relation_hypotheses(row,cell,trusted,learner['soft_relations'],codec))
            if learner.get('source_exact_rules'):
                exact=exact_hypothesis(row,col,trusted,learner['source_exact_rules'])
                if exact:proposals['source_exact_arithmetic']=exact
            for model in learner['conditional_models'].values():
                if model:proposals.update(conditional_hypotheses(row,cell,trusted,model,codec))
            for model in learner.get('mixture_models',{}).values():
                if model:proposals.update(mixture_hypotheses(row,cell,trusted,model,codec))
            for name,entries in uncertain.items():
                e=entries.get(col)
                if e and e['best_standardized_residual']<=1 and e['log_score_gap']>=5 and (e['rounds']==1 or e['stable_last_two']):
                    proposals[name]=e
            for name,proof in proposals.items():
                if name in operators and not equal(row[col],proof['value']):
                    offers[(i,col)].append({'operator':name,'value':proof['value'],'proof':proof})
    result=[]
    for d in decisions:
        proposals=offers[(d['row'],d['field'])];values={canonical(q['value']):q['value'] for q in proposals}
        after=d['after'];reason='sealed_baseline'
        if len(values)==1:
            after=next(iter(values.values()));reason='source_checked_additive_skill'
        elif len(values)>1:reason='conflicting_new_proposals_preserved'
        result.append({**d,'after':after,'v10':d['after'],'enrichment_reason':reason,
                       'enrichment_proposals':proposals,'original_anchor_unchanged':True})
    return result


def run(rows,base_path,learner):
    _,base,codec,family,policy=open_skill(base_path)
    cells=predict_rows(rows,base);decisions=propose_from_cells(rows,cells,base,codec,family,policy)
    return propose_from_existing(rows,cells,decisions,base,codec,learner)


def open_enrichment(path):
    root=Path(path);manifest=load(root/'manifest.json')
    if any(sha256(root/n)!=expected for n,expected in manifest['files'].items()):
        raise ValueError('Additive skill content changed')
    configuration=load(root/'configuration.json');base=Path(configuration['base_skill_path'])
    if sha256(base/'manifest.json')!=configuration['base_manifest_sha256']:
        raise ValueError('Sealed baseline dependency changed')
    if configuration['seed']!=20260927 or configuration['original_reference_cases_visible']:
        raise ValueError('Seed/visibility contract changed')
    open_skill(base)
    return configuration,load(root/'learner.json')
