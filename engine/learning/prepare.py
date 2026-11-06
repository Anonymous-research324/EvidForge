"""Learn once from the authorized bundle and prepare actual native9B tools."""
from pathlib import Path
from collections import defaultdict
import argparse,sys,time
import numpy as np
from sklearn.model_selection import GroupKFold
from .environment import RUNTIME,ROOT
from .evidence import labelled
from .review_selection import choose as fit_review,execute as review
from .candidate_verifier import rank,decisions,measure
from unseen_table.common import load,dump,fresh,normalize_bundle,sha256,equal,fingerprint
from skill_runtime.learning import learn
from skill_runtime.package import open_skill
from skill_runtime.codec import fit as fit_codec
from skill_runtime.row_family import fit as fit_family
from skill_runtime.execute import propose_from_cells
from skill_runtime.refinement import bounded_hypotheses
from unseen_table.rank_candidates import predict_rows
from unseen_table.learn_skill import pack_observations
from unseen_table.decision_check import crossfit


def affix_apply(rows,cells,ds,base,codec,enabled):
    lookup={(c['row'],c['column']):c for c in cells};out=[]
    for d in ds:
        after=d['after']
        if enabled and equal(d['before'],after):
            proposal=bounded_hypotheses(rows[d['row']],lookup[(d['row'],d['field'])],base,codec).get('exact_affix_literal')
            if proposal is not None:after=proposal
        out.append({**d,'after':after})
    return out


def prepare(bundle,schema,context,out):
    started=time.monotonic();root=fresh(out);basepath=root/'base_skill'
    learn(bundle,schema,context,basepath)
    _,base,codec,family,policy=open_skill(basepath)
    normalized,_=normalize_bundle(bundle,schema);records=normalized['learning']+normalized['validation']
    held=[];observations=[]
    for fold,(u,v) in enumerate(GroupKFold(3).split(records,groups=[r['group'] for r in records])):
        source=[records[int(i)] for i in u];query=[records[int(i)] for i in v]
        model=load(basepath/f'fold-{fold}-model.json')
        if set(model['numeric_context']['training_groups'])!={r['group'] for r in source}:raise ValueError('Fitting groups differ')
        cells=predict_rows([r['dirty'] for r in query],model);observations+=pack_observations(query,cells)
        held.append((source,query,model,cells))
    crossfit(observations,20260927)
    preliminary=[];offers=[];all_affix=[]
    for source,query,model,cells in held:
        model['policy']=base['policy'];cc=fit_codec(source,schema)
        ds=propose_from_cells([r['dirty'] for r in query],cells,model,cc,fit_family(source,schema),policy)
        augmented=affix_apply([r['dirty'] for r in query],cells,ds,model,cc,True)
        for cell in cells:
            literal=bounded_hypotheses(query[cell['row']]['dirty'],cell,model,cc).get('exact_affix_literal')
            if literal is not None:
                r=query[cell['row']];col=cell['column']
                all_affix.append({'group':r['group'],'matches':equal(literal,r['truth'][col]),
                                  'normal':equal(r['dirty'][col],r['truth'][col])})
        for a,b in zip(ds,augmented):
            if not equal(a['after'],b['after']):offers.append({'group':query[a['row']]['group'],
                'matches':equal(b['after'],query[a['row']]['truth'][a['field']]),
                'normal':equal(a['before'],query[a['row']]['truth'][a['field']])})
        preliminary.append((source,query,model,cells,cc,ds))
    # This exact character-preserving inverse is the SAME operation for base
    # and deferred cases. Check its whole held-source application domain and
    # separately reject any observed residual mistake. It never estimates a
    # value, clips the numeric literal, or borrows support from a different edit.
    affix=bool(len(all_affix)>=8 and len({q['group'] for q in all_affix})>=4 and
               sum(q['matches'] for q in all_affix)/len(all_affix)>=.98 and not any(q['normal'] for q in all_affix) and
               all(q['matches'] and not q['normal'] for q in offers))
    dump(root/'literal_inverse_source_receipt.json',{'operation':'learned numeric literal wrapper inverse only',
        'all_held_applications':len(all_affix),'groups':len({q['group'] for q in all_affix}),
        'exact':sum(q['matches'] for q in all_affix),'residual_applications':len(offers),
        'residual_exact':sum(q['matches'] for q in offers),'enabled':affix,
        'arbitrary_generated_or_numeric_edits_share_no_support':True,'formal_reference_read':False})
    fs=[]
    for source,query,model,cells,cc,ds in preliminary:
        ds=affix_apply([r['dirty'] for r in query],cells,ds,model,cc,affix)
        fs+=labelled([r['dirty'] for r in query],source,model,cells,ds,query)
    gate=fit_review(fs)
    dump(root/'source_records_backend.json',{'records':records,'source_reference_rows_visible_to_model':False})
    dump(root/'backend_only.json',{'frames':fs})
    anchor=measure(decisions(rank(fs,{'score_expression':'is_baseline','accept_expression':'True'}),{}))
    distribution={}
    pairs=[(q['features'],equal(q['value'],f['expected_local_only'])) for f in fs for q in f['options']]
    for name in sorted(pairs[0][0]):
        if name=='score_gap':continue
        values=[q[name] for q,_ in pairs];positive=[q[name] for q,match in pairs if match];negative=[q[name] for q,match in pairs if not match]
        distribution[name]={'p10_p50_p90':[float(x) for x in np.quantile(values,[.1,.5,.9])],
            'mean_on_exact_source_alternatives':float(np.mean(positive)) if positive else None,
            'mean_on_other_source_alternatives':float(np.mean(negative)) if negative else None}
    dump(root/'model_profile.json',{'target_fields':len(schema['targets']),'checked_source_groups':len({r['group'] for r in records}),
        'checked_cells':len(fs),'source_baseline_metrics':anchor,'source_feature_distributions':distribution,
        'base_review':'Unsafe source digit-only base proposals revert; other channels preserved.',
        'feature_semantics':{'rank_score':'source-held classifier evidence, uncalibrated',
            'noisy_raw/noisy_signed_log_score':'higher supports a candidate; soft OTHER-field evidence, often negative',
            'unknown_odds':'higher supports relative to unknown outcome','is_baseline':'the reviewed base choice'},
        'raw_reference_rows_visible':False,'no_formal_feedback_available':True})
    dump(root/'configuration.json',{'format_version':14,'base_skill_path':str(basepath.resolve()),
        'exact_affix_enabled':affix,'base_review_policy':gate,'seed':20260927,
        'scope':context,'status':'pending_independent_validation','raw_references_visible_to_model':False})
    dump(root/'source_freeze.json',{'source_bundle_sha256':fingerprint(bundle),'schema_sha256':fingerprint(schema),
        'base_manifest_sha256':sha256(basepath/'manifest.json'),'source_groups':sorted({r['group'] for r in records}),
        'scope':'source-held development selection, never a formal performance score'})
    dump(root/'prepared_manifest.json',{'files':{n:sha256(root/n) for n in ['backend_only.json','model_profile.json','source_freeze.json']}})
    dump(root/'prepare_receipt.json',{'seconds':time.monotonic()-started,'actual9B_calls':0,'source_baseline':anchor,
        'original_experiments_changed':False,'formal_gold_read':False})
    print('SOURCE_PREPARED_FOR_NATIVE9B',anchor,flush=True)
    return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--declaration',required=True)
    p.add_argument('--context',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(load(a.labels),load(a.declaration),load(a.context),a.out)


if __name__=='__main__':main()
