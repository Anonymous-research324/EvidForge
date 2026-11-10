"""Add context probabilities while preserving every prior alternative and KEEP."""
from learning.environment import RUNTIME
from unseen_table.common import equal,number
from skill_runtime.codec import possible
from .context import key


def enrich(frames,predictions):
    output=[]
    for frame in frames:
        e=predictions.get((frame['row'],frame['column']));old=frame['original'];baseline=frame['baseline_after_local']
        eligible=bool(e and equal(old,baseline) and
                      (not old.strip() or number(old) is not None and key(old) not in e['domain']))
        options=[]
        for q in frame['options']:
            k=key(q['value']);features={**q['features'],
                'is_baseline':float(equal(q['value'],baseline)),
                'baseline_changed':float(not equal(old,baseline)),
                'context_nb_probability':e['nb'].get(k,0.) if e else 0.,
                'context_tree_probability':e['tree'].get(k,0.) if e else 0.,
                'context_model_agrees':float(e['agrees']) if e else 0.,
                'context_domain_cardinality':len(e['domain'])/16 if e else 0.,
                'context_observed_fraction':e['observed_fraction'] if e else 0.,
                'forward_explainable':float(eligible and k in e['domain'] and possible(old,q['value'],frame['column'],e['codec']))}
            options.append({**q,'features':features})
        output.append({**frame,'options':options})
    return output
