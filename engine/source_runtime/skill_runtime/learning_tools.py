"""Bounded learning tools callable from an existing harness or a model probe.

Only aggregate source-held learning measurements are returned. No source rows,
reference pairs, test labels, original tables or arbitrary Python execution.
"""
from pathlib import Path
from unseen_table.common import load,sha256

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTIONS={
 'exact_affix_literal':'Remove an exactly matched learned alphabetic prefix/suffix; keep the numerical literal untouched.',
 'column_precision_unique':'Consider an exact inverse/edit only when one supported candidate fits this column range and literal precision.',
 'context_exact_gap_2':'Compare exact candidates using an already trusted nearest-neighbour numerical context, gap2.',
 'context_exact_gap_4':'Same exact-candidate check with stricter gap4.',
 'context_exact_gap_8':'Same exact-candidate check with stricter gap8.',
 'soft_relation_consensus_1':'Use source-cross-checked soft affine/quadratic/log relations from another trusted field to compare exact candidates.',
 'soft_relation_consensus_2':'Require agreement from two other fields; never write predicted means.',
 'conditional_raw_gap_2':'Use trusted-other-field covariance evidence with gap2; target itself is masked.',
 'conditional_raw_gap_4':'Same conditional candidate evidence with stricter gap4.',
 'conditional_raw_gap_8':'Same conditional candidate evidence with stricter gap8.',
 'conditional_signed_log_gap_4':'Use a signed-log conditional covariance check, gap4; exact candidates only.',
 'uncertain_signed_log_r1_gap5':'Compare exact candidates with uncertain other-field candidate mixtures, signed log, one update, log gap5; never treat unapproved guesses as certain.',
 'uncertain_signed_log_r3_gap5':'Same uncertain candidate evidence, three simultaneous updates; require stable last two winners.',
}
FAMILIES={
 'literal_family':['exact_affix_literal','column_precision_unique'],
 'nearest_context_family':['context_exact_gap_2','context_exact_gap_4','context_exact_gap_8'],
 'soft_relation_family':['soft_relation_consensus_1','soft_relation_consensus_2'],
 'conditional_family':['conditional_raw_gap_2','conditional_raw_gap_4','conditional_raw_gap_8','conditional_signed_log_gap_4'],
 'uncertain_context_family':['uncertain_signed_log_r1_gap5','uncertain_signed_log_r3_gap5'],
}
for _mode in ['raw','signed_log']:
    for _clusters in [2,3]:
        for _gap in [3,4,5]:
            _id=f'mixture_{_mode}_k{_clusters}_gap{_gap}'
            DESCRIPTIONS[_id]=f'Learn {_clusters} unlabelled source-only numerical clusters, {_mode} covariance, compare exact candidates with log gap{_gap}; target is excluded from cluster membership evidence.'
FAMILIES['mixture_context_family']=[k for k in DESCRIPTIONS if k.startswith('mixture_')]
DESCRIPTIONS['source_exact_arithmetic']='Learn a bounded two-other-field arithmetic relationship and its complete rounding-consistent coefficient interval from teaching sources; return only a literal invariant throughout that interval, never a fitted mean.'
FAMILIES['source_exact_arithmetic_family']=['source_exact_arithmetic']


def measurements(dataset):
    root=ROOT/'research';result={}
    for suffix in ['refinement_public','relations_public','conditional_public','uncertain_context_public_v2','mixture_public']:
        report=root/(dataset+'_'+suffix)/'result.json'
        if not report.is_file():continue
        for name,m in load(report)['clauses'].items():
            if name in DESCRIPTIONS:
                simple=name in ('exact_affix_literal','column_precision_unique')
                groups=m.get('source_groups',m.get('groups',0));exact=m.get('correct',m.get('exact',0))
                passed=(m['proposals']>=(5 if simple else 10) and groups>=(3 if simple else 5)
                        and m['P']>=.98 and m['normal_damage']==0)
                result[name]={'operator':name,'checked_proposals':m['proposals'],
                    'exact_repairs':exact,'precision':m['P'],'normal_cells_damaged':m['normal_damage'],
                    'checked_source_groups':groups,'source_cross_check_passed':bool(passed),
                    'feedback':'eligible to compile a PARTIAL candidate skill now; deployment awaits independent refit evaluation' if passed else
                       'insufficient source support or inaccurate/normal-damaging proposals; do not compile',
                    'evidence_sha256':sha256(report), 'original_reference_cases_visible':False}
    return result


def trial(dataset,operators):
    if not operators or len(set(operators))!=len(operators) or not set(operators)<=set(DESCRIPTIONS):
        raise ValueError('Choose unique bounded operator IDs')
    stats=measurements(dataset)
    return [stats.get(op,{'operator':op,'checked_proposals':0,'source_cross_check_passed':False,
                         'feedback':'no held-source support; this clause remains closed'}) for op in operators]


def family_trial(dataset,families):
    if not families or len(set(families))!=len(families) or not set(families)<=set(FAMILIES):
        raise ValueError('Evaluate unique available learning FAMILY IDs')
    return trial(dataset,[op for family in families for op in FAMILIES[family]])


def compile_plan(dataset,operators,evaluated):
    if not operators or len(set(operators))!=len(operators):
        return {'accepted':False,'reason':'Compile a nonempty set of unique evaluated operator IDs'}
    if not set(operators)<=set(evaluated):
        return {'accepted':False,'reason':'Every operator must be evaluated before compilation'}
    verdicts=trial(dataset,operators)
    if not all(v['source_cross_check_passed'] for v in verdicts):
        return {'accepted':False,'reason':'An operator failed the source-held quality check','feedback':verdicts}
    return {'accepted':True,'language':'bounded verified repair-template configuration',
            'operators':operators,'status':'pending_independent_validation',
            'raw_reference_cases_visible':False,'precision_target':.98,'normal_damage_target':.005,
            'estimated_continuous_values_not_written':True,'conflicting_proposals':'keep and record disagreement'}


def progress(evaluated,remaining_calls):
    """Expose evidence and scope, without choosing the model's final operators."""
    return {'distinct_methods_evaluated':len(evaluated),'required_distinct_methods':2,
            'eligible_partial_skill_operators':[k for k,v in evaluated.items() if v['source_cross_check_passed']],
            'not_yet_evaluated':[k for k in DESCRIPTIONS if k not in evaluated],
            'untried_method_families':[k for k,v in FAMILIES.items() if not set(v)<=set(evaluated)],
            'remaining_calls':remaining_calls,
            'partial_skill_allowed':True,'compilation_is_not_deployment':True,
            'duplicate_trials_produce_identical_cached_evidence':True}


def finish_allowed(action,operators,evaluated):
    if len(evaluated)<2:
        return False,'Evaluate at least two distinct methods before closing'
    if action=='defer' and operators:
        return False,'A defer action performs no trial; use evaluate for requested methods'
    if action=='defer' and any(v['source_cross_check_passed'] for v in evaluated.values()):
        return False,'Unreviewed passing methods remain: compile a PARTIAL pending-validation skill or continue testing; failing unrelated methods do not invalidate passing methods'
    return True,'Closure contract satisfied'
