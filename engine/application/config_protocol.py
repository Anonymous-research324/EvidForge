"""Typed model-generated rule configuration avoids pseudocode syntax failures.

Original9B JSON is immutable. A bounded compiler creates the expression as a
separate tool artifact; it never invents scores, reference values or decisions.
"""
from pathlib import Path
import json,math
from .bounded_protocol import Protocol as PreviousProtocol,state,classify
from learning.arithmetic import FIELDS
from learning.candidate_verifier import evaluate
from unseen_table.common import load,dump,fingerprint


def compile_configuration(answer):
    if set(answer)!={'action','terms','gates'} or answer['action']!='propose':raise ValueError('Expected typed new-rule configuration')
    if not isinstance(answer['terms'],list) or not 1<=len(answer['terms'])<=8:raise ValueError('One to eight score terms')
    if not isinstance(answer['gates'],list) or len(answer['gates'])>6:raise ValueError('At most six acceptance predicates')
    terms=[];gates=[]
    import operator
    compares={'>':operator.gt,'>=':operator.ge,'<':operator.lt,'<=':operator.le}
    required_new_values={'is_baseline':0.,'forward_explainable':1.}
    for q in answer['terms']:
        if set(q)!={'feature','weight'} or q['feature'] not in FIELDS-{'score_gap'}:raise ValueError('Unknown scoring feature')
        if type(q['weight']) not in (int,float) or not math.isfinite(q['weight']) or abs(q['weight'])>2:raise ValueError('Bounded finite score coefficient')
        terms.append(f"({q['feature']} * {q['weight']!r})")
    for q in answer['gates']:
        if set(q)!={'feature','op','value'} or q['feature'] not in FIELDS or q['op'] not in ['>=','<=','>','<']:raise ValueError('Unknown bounded predicate')
        if type(q['value']) not in (int,float) or not math.isfinite(q['value']) or abs(q['value'])>100:raise ValueError('Bounded finite predicate')
        if q['feature'] in required_new_values and not compares[q['op']](required_new_values[q['feature']],q['value']):
            raise ValueError('Logically impossible NEW repair gate: '+q['feature']+' must allow '+str(required_new_values[q['feature']])+'. Every new candidate has is_baseline=0; existing baseline is automatically the fallback, not a positive acceptance gate. Do not require a previous answer to be a new answer.')
        gates.append(f"({q['feature']} {q['op']} {q['value']!r})")
    return {'score_expression':' + '.join(terms),'accept_expression':' and '.join(gates) if gates else 'True'}


class Protocol(PreviousProtocol):
    def payload(self):
        request=super().payload()
        term={'type':'object','properties':{'feature':{'type':'string','enum':sorted(FIELDS-{'score_gap'})},
              'weight':{'type':'number','minimum':-2,'maximum':2}},'required':['feature','weight'],'additionalProperties':False}
        gate={'type':'object','properties':{'feature':{'type':'string','enum':sorted(FIELDS)},
              'op':{'type':'string','enum':['>=','<=','>','<']},'value':{'type':'number','minimum':-100,'maximum':100}},
              'required':['feature','op','value'],'additionalProperties':False}
        proposed={'type':'object','properties':{'action':{'type':'string','enum':['propose']},
            'terms':{'type':'array','items':term,'minItems':1,'maxItems':8},
            'gates':{'type':'array','items':gate,'minItems':0,'maxItems':6}},
            'required':['action','terms','gates'],'additionalProperties':False}
        schema=request['structured_outputs']['json']
        if 'anyOf' in schema:
            schema['anyOf']=[proposed if x['properties']['action']['enum']==['propose'] else x for x in schema['anyOf']]
        elif schema['properties']['action']['enum']==['propose']:request['structured_outputs']['json']=proposed
        request['messages'][0]['content']='''Create a NEW weighted candidate-selection rule as constrained JSON, not free-form program strings. For propose, terms define the SUM of feature*weight. gates define acceptance as the AND of numeric predicates; empty gates means True. The tool compiles your configuration into bounded executable code and independently checks it. Score every original candidate including KEEP; no competitor is removed. Unqualified new actions defer to the strong existing baseline. Maximize COMPOSITE exact repair F1, comparing with source_baseline_metrics. Incremental actions need their OWN >=98% exactness, >=8 actions from >=4 sources, zero added normal damage and positive net gain; easy baseline repairs cannot mask them. Use feature distributions and previous raw feedback to revise a rule that was rejected. Higher rank_score, noisy log scores and unknown odds mean more support. Noisy log scores are often negative: additive terms preserve their direction. Higher score_gap means a better separated winner. is_baseline marks the previous approved candidate; EVERY NOVEL candidate has is_baseline=0. Keeping the baseline means DEFAULT FALLBACK when a new rule fails. NEVER require is_baseline>0 or >=a positive number in an acceptance gate: that logically disables every new repair. Use its coefficient only as a preference if needed. Acceptance must allow forward_explainable=1. is_keep marks the observed value. Choose your coefficients and predicates from measured source feedback. Try two distinct configurations, up to four total attempts, then compile the highest measured eligible program or safely defer if none passes. No original teaching rows or outer-check answers are visible. Return only the constrained JSON.'''
        return request

    def process(self,n,answer):
        if answer.get('action')!='propose':return super().process(n,answer)
        budget=load(self.root/'budget.json')
        if budget['pending_call']!=n or answer!=load(self.root/f'actual9b-decision-{n}.json'):
            raise ValueError('Only the exact recorded9B configuration may execute')
        current=state(self.work)
        try:
            if current['closed'] or current.get('attempted_proposals',0)>=4:raise ValueError('Preserve closed/bounded session')
            current['attempted_proposals']=current.get('attempted_proposals',0)+1
            program=compile_configuration(answer)
            dump(self.root/f'actual9b-derived-program-{n}.json',{'original_configuration':answer,'compiled_expression':program,
                'compiler_only_no_model_response_changed':True})
            identifier=fingerprint(program)[:16]
            if any(q['program_id']==identifier for q in current['programs']):raise ValueError('Repeated configuration cannot add support')
            try:
                result={**evaluate(load(self.work/'backend_only.json')['frames'],program),
                    'program_id':identifier,'original_configuration':answer}
            except (ValueError,SyntaxError,KeyError,TypeError) as exc:
                result={'program_id':identifier,'program':program,'original_configuration':answer,
                        'eligible_for_research_compile':False,'rejected_reason':str(exc)}
            current['programs'].append(result)
        except ValueError as exc:result={'accepted':False,'reason':str(exc)}
        dump(self.work/'hypothesis_state.json',current)
        with (self.work/'hypothesis_trace.jsonl').open('a') as f:
            f.write(json.dumps({'action':answer,'result':result,'outer_feedback_visible':False,'reference_rows_visible':False})+'\n')
        budget['last_tool_result']=result;budget['pending_call']=None;dump(self.root/'budget.json',budget)
        return result
