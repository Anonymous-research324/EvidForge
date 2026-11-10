"""Choose baseline review using source predictions, never formal scores.

A safety check is not automatically useful on an already reliable channel.
Select it when source safety requires it, or when it improves exact repair F1
without reducing precision/increasing normal damage. Preserve all diagnostics.
"""
from .base_review import fit,apply,point
from .candidate_verifier import rank,decisions,measure
from unseen_table.common import equal


def metrics(frames):
    return measure(decisions(rank(frames,{'score_expression':'is_baseline','accept_expression':'True'}),{}))


def choose(frames):
    policy=fit(frames);before=metrics(frames);after=metrics(apply(frames,policy))
    digit=[f for f in frames if point(f) is not None]
    digit_good=sum(equal(f['baseline_after_local'],f['expected_local_only']) for f in digit)
    digit_normal=sum(equal(f['original'],f['expected_local_only']) for f in digit)
    digit_P=digit_good/len(digit) if digit else 1.0
    # Match the user's whole-output safety targets, then maximize source F1.
    # Channel risks remain visible; they cannot silently redefine the objective
    # as98% for EACH old channel. Novel proposed actions keep separate checks.
    before_ok=before['P']>=.98 and before['damage']<=.005
    after_ok=after['P']>=.98 and after['damage']<=.005
    unsafe=not before_ok
    enabled=bool(after_ok and (not before_ok or after['F1']>before['F1']+1e-12))
    return {**policy,'enabled':enabled,'source_unreviewed':before,'source_reviewed':after,'digit_channel':{'checks':len(digit),'P':digit_P,'normal_damage':digit_normal},
            'reason':'source_safety' if enabled and unsafe else 'source_F1_gain' if enabled else 'retain_source_baseline',
            'independent_deployment_validation_required':True}


def execute(frames,policy):
    return apply(frames,policy) if policy['enabled'] else frames
