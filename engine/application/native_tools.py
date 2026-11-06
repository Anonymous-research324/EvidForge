"""Registered native decide tool for original9B-generated hypotheses only."""
from pathlib import Path
import argparse, json, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'application'))
from learning.environment import RUNTIME
from unseen_table.common import load as read, dump as save
from application.native_protocol import Protocol, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['hook', 'mcp']); p.add_argument('session')
    a = p.parse_args(); session = Path(a.session)
    work = Path(read(session / 'session.json')['work'])
    if a.mode == 'hook':
        payload = json.load(sys.stdin)
        event, name = payload.get('hook_event_name'), str(payload.get('tool_name', ''))
        verdict = {}
        if event == 'PreToolUse' and name not in ['mcp__dataforge_learning__decide', 'mcp__dataforge_learning.decide']:
            verdict = {'decision': 'block', 'reason': 'Only recorded hypothesis tool decisions may execute. Teacher/reference file access is closed.'}
        elif event == 'Stop' and not state(work)['closed']:
            verdict = {'decision': 'block', 'reason': 'Check distinct hypotheses and compile the best qualified program, or defer after all checks fail.'}
        with (session / 'native_hooks.jsonl').open('a') as handle:
            handle.write(json.dumps({'event': event, 'tool_name': name, 'verdict': verdict}) + '\n')
        if verdict:
            print(json.dumps(verdict))
        return
    tool = {'name': 'decide', 'description': 'Execute only recorded9B-generated candidate rules and return aggregate inner-source feedback.',
            'inputSchema': {'type': 'object', 'properties': {'decision': {'type': 'object'}},
                            'required': ['decision'], 'additionalProperties': False}}
    for line in sys.stdin:
        try:
            q = json.loads(line); id_ = q.get('id'); method = q.get('method')
            if id_ is None:
                continue
            if method == 'initialize':
                result = {'protocolVersion': q.get('params', {}).get('protocolVersion', '2024-11-05'),
                          'capabilities': {'tools': {}}, 'serverInfo': {'name': 'dataforge-hypotheses', 'version': '13.0'}}
            elif method == 'ping': result = {}
            elif method == 'tools/list': result = {'tools': [tool]}
            elif method == 'tools/call':
                try:
                    if q['params']['name'] != 'decide':
                        raise ValueError('Unknown registered tool')
                    value = Protocol(session).process(read(session / 'budget.json')['calls'], q['params']['arguments']['decision'])
                    result = {'content': [{'type': 'text', 'text': json.dumps(value)}]}
                except Exception as exc:
                    result = {'isError': True, 'content': [{'type': 'text', 'text': f'{type(exc).__name__}: {exc}'}]}
            else:
                raise ValueError('Unknown registered MCP method')
            print(json.dumps({'jsonrpc': '2.0', 'id': id_, 'result': result}), flush=True)
        except Exception as exc:
            print(str(exc), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
