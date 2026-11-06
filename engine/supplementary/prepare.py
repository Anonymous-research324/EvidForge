"""Prepare a new context operator from authorized source20% only.

It preserves the V18 parameter package as a sealed prior stage. Development
frames were already used for source selection and are not formal results.
"""
from pathlib import Path
import argparse,shutil,json,time
import numpy as np
from sklearn.model_selection import GroupKFold
from learning.environment import RUNTIME
from learning.review_selection import execute as review
from learning.candidate_verifier import rank,permitted,measure,decisions
from learning.arithmetic import FIELDS
from unseen_table.common import load,dump,fresh,sha256,equal
from .context import fit,predict
from .frames import enrich


def prepare(prior_path,out):
    started=time.monotonic();root=fresh(out);prior=root/'prior_skill'
    shutil.copytree(prior_path,prior)
    config=load(prior/'configuration.json');config['base_skill_path']=str((prior/'base_skill').resolve())
    dump(prior/'configuration.json',config);manifest=load(prior/'manifest.json')
    manifest['files']['configuration.json']=sha256(prior/'configuration.json');dump(prior/'manifest.json',manifest)
    records=load(prior/'source_records_backend.json')['records'];source_frames=load(prior/'backend_only.json')['frames']
    schema=load(prior/'base_skill/base/skill.json')['declaration'];targets=schema['targets']
    safe=review(source_frames,config['base_review_policy']);cpath=prior/'compiled_candidate.json'
    c=load(cpath) if cpath.exists() else None;ranked=rank(source_frames,c['program']) if c else None
    baseline=[]
    for i,(f,s) in enumerate(zip(source_frames,safe)):
        q=ranked[i] if ranked else None;after=q['value'] if q and permitted(q,c['policy']) else s['baseline_after_local']
        baseline.append({**f,'baseline_after_local':after})
    ordered=[];folds=[];frames=[];at=0
    for fold,(u,v) in enumerate(GroupKFold(3).split(records,groups=[r['group'] for r in records])):
        training=[records[int(i)] for i in u];checking=[records[int(i)] for i in v]
        model=fit(training,schema);predicts=predict([r['dirty'] for r in checking],model)
        block=baseline[at:at+len(checking)*len(targets)];at+=len(block)
        for f in block:
            r=checking[f['row']]
            if f['group']!=r['group'] or not equal(f['original'],r['dirty'][f['column']]):raise ValueError('Source frame order differs')
        frames+=enrich(block,predicts)
        folds.append({'training_groups':sorted({r['group'] for r in training}),
                      'check_groups':sorted({r['group'] for r in checking}),
                      'context_model_fields':list(model['models'])})
    if at!=len(baseline):raise ValueError('Some source frames were omitted')
    dump(root/'context_model.json',fit(records,schema));dump(root/'context_source_folds.json',folds)
    dump(root/'backend_only.json',{'frames':frames});dump(root/'source_records_backend.json',{'records':records})
    anchor=measure(decisions(rank(frames,{'score_expression':'is_baseline','accept_expression':'True'}),{}))
    pairs=[(q['features'],equal(q['value'],f['expected_local_only'])) for f in frames for q in f['options']]
    distributions={}
    for feature in sorted(FIELDS-{'score_gap'}):
        values=[f[feature] for f,_ in pairs];positive=[f[feature] for f,y in pairs if y];negative=[f[feature] for f,y in pairs if not y]
        distributions[feature]={'p10_p50_p90':list(map(float,np.quantile(values,[.1,.5,.9]))),
            'mean_on_exact_source_alternatives':float(np.mean(positive)) if positive else None,
            'mean_on_other_source_alternatives':float(np.mean(negative)) if negative else None}
    dump(root/'model_profile.json',{'target_fields':len(targets),'checked_source_groups':len(records),'checked_cells':len(frames),
        'source_baseline_metrics':anchor,'source_feature_distributions':distributions,
        'new_operator':'Small-domain OTHER-observed-field context. Never use ID or protected class. Existing baseline edits are preserved.',
        'feature_semantics':{'context_nb_probability':'NB conditional probability for this candidate',
            'context_tree_probability':'ExtraTrees conditional probability for this candidate',
            'context_model_agrees':'whether NB and ExtraTrees select the same domain value',
            'context_domain_cardinality':'small learned numeric domain size divided by16',
            'context_observed_fraction':'fraction of other small-domain fields with unambiguous current observations',
            'forward_explainable':'only residual blanks or numeric values outside the learned domain with literal forward support',
            'is_baseline':'previous full V18 output, including any already generated source rule'},
        'suggested_research_hypothesis':'Rank on conditional probabilities, retain baseline on uncertainty, validate probability/agreement gates independently using the backend tool.',
        'source_score_not_formal':True,'raw_reference_rows_visible':False,'no_formal_feedback_available':True})
    dump(root/'configuration.json',{'format_version':19,'base_skill_path':config['base_skill_path'],
        'prior_skill_path':str(prior.resolve()),'scope':config['scope'],'seed':20260927,'status':'pending_independent_validation'})
    dump(root/'source_freeze.json',{'source_bundle_sha256':load(prior/'source_freeze.json')['source_bundle_sha256'],
        'base_manifest_sha256':sha256(prior/'base_skill/manifest.json'),'prior_manifest_sha256':sha256(prior/'manifest.json'),
        'context_model_sha256':sha256(root/'context_model.json'),'source_groups':sorted({r['group'] for r in records}),
        'scope':'Reused authorized source-held development; no formal reference read'})
    dump(root/'prepared_manifest.json',{'files':{n:sha256(root/n) for n in ['backend_only.json','model_profile.json','source_freeze.json']}})
    dump(root/'prepare_receipt.json',{'seconds':time.monotonic()-started,'actual9B_calls':0,'formal_gold_read':False,'source_baseline':anchor})
    print('CONTEXT_SOURCE_PREPARED',anchor,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(Path(a.prior),Path(a.out))
