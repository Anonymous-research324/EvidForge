"""Domain hypothesis tools carried by the existing native Codex loop/Gateway."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load, dump, sha256, fingerprint
from skill_runtime.package import verify_request
from skill_runtime.client import strict_json
from learning.arithmetic import FIELDS
from learning.candidate_verifier import evaluate


def classify(path, body):
    if path != '/v1/responses':
        raise ValueError('Unknown existing native control endpoint')
    return 'decision' if body.get('tools') else 'compaction'


def state(work):
    path = Path(work) / 'hypothesis_state.json'
    return load(path) if path.exists() else {'programs': [], 'attempted_proposals': 0, 'closed': False, 'compiled': None, 'deferred': False}


class Protocol:
    def __init__(self, session):
        self.root = Path(session)
        self.spec = load(self.root / 'session.json')
        self.work = Path(self.spec['work'])
        manifest = load(self.work / 'prepared_manifest.json')
        if any(sha256(self.work / n) != h for n, h in manifest['files'].items()):
            raise ValueError('Source-only learning inputs changed')

    def state(self):
        return {'status': 'reviewed' if state(self.work)['closed'] else 'pending'}

    def checkpoint(self):
        return {'hypothesis_state': state(self.work), 'scope': 'inner-source feedback; outer answers hidden'}

    def payload(self):
        current = state(self.work)
        valid = [q for q in current['programs'] if q.get('eligible_for_research_compile')]
        proposed = {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': ['propose']},
            'score_expression': {'type': 'string', 'minLength': 1, 'maxLength': 1024},
            'accept_expression': {'type': 'string', 'minLength': 1, 'maxLength': 1024}},
            'required': ['action', 'score_expression', 'accept_expression'], 'additionalProperties': False}
        compile_ = {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': ['compile']},
            'program_id': {'type': 'string', 'enum': [q['program_id'] for q in valid]}},
            'required': ['action', 'program_id'], 'additionalProperties': False}
        defer = {'type': 'object', 'properties': {'action': {'type': 'string', 'enum': ['defer']}},
                 'required': ['action'], 'additionalProperties': False}
        attempts = current.get('attempted_proposals', len(current['programs']))
        choices = [proposed] if attempts < 4 else []
        if len(current['programs']) >= 2 and valid:
            choices.append(compile_)
        if not choices:
            choices = [defer]
        schema = choices[0] if len(choices) == 1 else {'anyOf': choices}
        budget = load(self.root / 'budget.json')
        summary = load(self.work / 'model_profile.json')
        summary['allowed_feature_variables'] = sorted(FIELDS)
        payload = {'rows': {}, 'learned_summary': summary, 'batch': budget['calls'],
                   'previous_answer': {'tested_programs': current['programs'],
                                       'maximum_attempted_hypotheses': 4, 'last_tool_feedback': budget.get('last_tool_result'), 'attempted_proposals': attempts, 'outer_feedback_visible': False}}
        verify_request(payload, [])
        prompt = '''You propose NEW executable numerical candidate-selection rules that IMPROVE the existing strong source-held baseline. The measured objective is COMPOSITE exact repair F1, not the precision of a tiny isolated rule. Unqualified new actions defer to that baseline. Proposals that add errors cannot hide behind baseline easy repairs: each incremental channel must independently pass P>=98%, zero new normal damage, at least8 actions and4 source groups. Compare your results with source_baseline_metrics and revise using actual feedback. The variable is_baseline denotes the previous approved candidate; rank_score is a classifier prior. noisy_raw_log_score/noisy_signed_log_score include learned observation damage probabilities and soft evidence from OTHER fields, excluding the target. Their context-fields flags and unknown-odds feature indicate support. They are not calibrated probabilities. All noisy log scores and unknown odds are HIGHER when the candidate has MORE support; log scores are commonly NEGATIVE. Do not multiply negative log scores as probability factors (1 + w * log_score can become negative). An additive scoring expression or explicit bounded conditional has predictable orientation. score_gap is HIGHER when the winner is MORE separated; accepting only small gaps targets ambiguous cases. Source feature distributions are provided; use the measured support and feedback, not guessed ranges. Consider combining this evidence with the existing rank_score or a baseline fallback instead of discarding most cells. Do not select a prewritten method ID. Produce score_expression and accept_expression combining the allowed measured feature variables. Syntax is bounded arithmetic +,-,*,/, comparisons, and/or/not, if/else expressions, and abs/min/max/log1p/sqrt calls. No imports, loops, strings, variables outside the interface, or row/reference access. All candidate alternatives including KEEP participate in ranking; unsupported top proposals defer rather than removing competitors. score_gap is available only to accept_expression, never to score_expression. The other features are numbers; use comparisons for Boolean tests. Use Python lowercase and/or/not; logical uppercase spellings are token-normalized without adding executable operations. When noisy context fields are fewer than two, use a measured fallback with is_baseline/rank_score and learned forward mechanisms. The allowed feature list is authoritative; do not refer to old residual features outside it. Candidate scores are not guaranteed probabilities. Source-only tool feedback will measure your program and choose permission thresholds; never hardcode a table value. Try at least two distinct hypotheses and improve from measured feedback, up to four. Then compile the eligible program with the highest measured F1 (prefer simpler expressions for ties), or defer if none qualifies. Compilation is research only; separate outer sources have not been checked. Original teaching rows and outer check answers are hidden. Return only the constrained JSON.'''
        return {'model': self.spec['model'], 'messages': [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(payload, separators=(',', ':'))}],
                'seed': 20260927, 'temperature': 0, 'max_tokens': 4096, 'thinking': False,
                'structured_outputs': {'json': schema, 'disable_any_whitespace': True},
                '_trace_tag': self.spec['tag'] + 'call' + str(budget['calls'] + 1)}

    def begin(self, payload):
        budget = load(self.root / 'budget.json')
        if budget['pending_call'] is not None or budget['calls'] >= 8:
            raise ValueError('Native hypothesis action budget exceeded or previous decision pending')
        budget['calls'] += 1
        budget['pending_call'] = budget['calls']
        dump(self.root / 'budget.json', budget)
        return budget['calls']

    def receive(self, n, raw):
        dump(self.root / f'actual9b-response-{n}.json', raw)
        budget = load(self.root / 'budget.json')
        budget['actual_usage'].append(raw.get('usage', {}))
        if raw['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('Actual9B hypothesis was truncated; no partial code executes')
        answer = strict_json(raw['choices'][0]['message']['content'])
        dump(self.root / f'actual9b-decision-{n}.json', answer)
        dump(self.root / 'budget.json', budget)
        return answer

    def process(self, n, answer):
        budget = load(self.root / 'budget.json')
        if budget['pending_call'] != n or answer != load(self.root / f'actual9b-decision-{n}.json'):
            raise ValueError('Only the original recorded9B decision may execute')
        current = state(self.work)
        try:
            if current['closed']:
                raise ValueError('Preserve the compiled/closed checkpoint')
            if answer.get('action') == 'propose':
                if set(answer) != {'action', 'score_expression', 'accept_expression'} or current.get('attempted_proposals', len(current['programs'])) >= 4:
                    raise ValueError('Invalid bounded hypothesis action')
                current['attempted_proposals'] = current.get('attempted_proposals', len(current['programs'])) + 1
                program = {k: answer[k] for k in ['score_expression', 'accept_expression']}
                identifier = fingerprint(program)[:16]
                if any(q['program_id'] == identifier for q in current['programs']):
                    raise ValueError('Repeating the same program cannot add source support')
                try:
                    frames = load(self.work / 'backend_only.json')['frames']
                    result = {**evaluate(frames, program), 'program_id': identifier}
                except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                    result = {'program_id': identifier, 'program': program,
                              'eligible_for_research_compile': False, 'rejected_reason': str(exc)}
                current['programs'].append(result)
            elif answer.get('action') == 'compile':
                if set(answer) != {'action', 'program_id'} or len(current['programs']) < 2:
                    raise ValueError('Check at least two distinct hypotheses before compilation')
                valid = [q for q in current['programs'] if q.get('eligible_for_research_compile')]
                chosen = next((q for q in valid if q['program_id'] == answer['program_id']), None)
                if not chosen or chosen['metrics']['F1'] < max(q['metrics']['F1'] for q in valid) - 1e-12:
                    raise ValueError('Compile the highest measured eligible program')
                result = {**chosen, 'status': 'pending_independent_outer_source_check',
                          'source_freeze_sha256': sha256(self.work / 'source_freeze.json'),
                          'original_reference_rows_visible': False, 'formal_gold_read': False}
                dump(self.work / 'compiled_candidate.json', result)
                current.update(closed=True, compiled=result)
            elif answer == {'action': 'defer'}:
                valid = [q for q in current['programs'] if q.get('eligible_for_research_compile')]
                if current.get('attempted_proposals', len(current['programs'])) < 4 or (valid and len(current['programs']) >= 2):
                    raise ValueError('Review remaining hypotheses or save a passing candidate')
                result = {'deferred': True, 'no_eligible_generated_program': not bool(valid), 'insufficient_distinct_hypotheses': len(current['programs']) < 2}
                current.update(closed=True, deferred=True)
            else:
                raise ValueError('Unknown generated-rule tool action')
        except ValueError as exc:
            result = {'accepted': False, 'reason': str(exc)}
        dump(self.work / 'hypothesis_state.json', current)
        with (self.work / 'hypothesis_trace.jsonl').open('a') as handle:
            handle.write(json.dumps({'action': answer, 'result': result,
                                     'outer_feedback_visible': False, 'reference_rows_visible': False}) + '\n')
        budget['last_tool_result'] = result
        budget['pending_call'] = None
        dump(self.root / 'budget.json', budget)
        return result
