"""Run the locked EXISTING native Codex harness, MCP and hooks with actual9B.

No HOME/CODEX_HOME overrides, framework edits, alternative model or new model
service. The old transport gateway is byte-identical and injects no fake9B.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,subprocess,sys,time,shlex
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'application'))
from learning.environment import RUNTIME
from unseen_table.common import fresh,load,dump,sha256
from application.native_protocol import state, Protocol as GeneratedProtocol
import native_gateway
native_gateway.Protocol = GeneratedProtocol


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--out',required=True)
    p.add_argument('--codex',required=True);p.add_argument('--remote-helper',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    work=Path(a.work).resolve();session=fresh(a.out);model='dataforge-native-learning-9b'
    dump(session/'session.json',{'work':str(work),'model':model,'tag':a.tag,'scope':'actual9B-generated executable expression learning; outer checks hidden, not formal performance'})
    dump(session/'budget.json',{'calls':0,'pending_call':None,'actual_usage':[],'last_tool_result':None})
    def forward(url,**kwargs):
        if url!='http://127.0.0.1:8000/v1/chat/completions':raise ValueError('Unexpected actual model endpoint')
        wire=kwargs['json'];dump(session/f'actual9b-request-{wire["_trace_tag"]}.json',wire)
        proc=subprocess.run([sys.executable,'-B',a.remote_helper,'call'],input=json.dumps(wire),text=True,capture_output=True,timeout=330)
        if proc.returncode:raise RuntimeError(proc.stderr)
        raw=json.loads(proc.stdout)
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:raw)
    # Only the copied adapter's transport dependency is replaced. Original
    # gateway source and all original framework modules remain unchanged.
    native_gateway.requests=SimpleNamespace(post=forward)
    gateway=native_gateway.Gateway(session,model);cwd=fresh(session/'harness_workspace');hooks=fresh(cwd/'.codex')
    command=shlex.join([sys.executable,'-B',str(ROOT/'application/native_tools.py'),'hook',str(session)])
    dump(hooks/'hooks.json',{'hooks':{event:[{'hooks':[{'type':'command','command':command,'timeout':30}]}] for event in ['PreToolUse','PostToolUse','Stop']}})
    instructions=session/'instructions.txt';instructions.write_text('Dispatch the registered decide tool, preserve original recorded9B decisions, and stop only when the source-held generated-rule tool session closes. Do not read teacher files or use shell/web tools.\n')
    overrides={'model':model,'model_provider':'dataforge_native','model_context_window':131072,'model_auto_compact_token_limit':10000000,
        'model_instructions_file':str(instructions),'project_doc_max_bytes':0,'tool_output_token_limit':6000,'web_search':'disabled',
        'features.shell_tool':False,'features.unified_exec':False,'features.apply_patch_freeform':False,'features.multi_agent':False,
        'features.multi_agent_v2':False,'features.tool_search':False,'features.plugins':False,'features.recommended_plugins':False,
        'model_providers.dataforge_native.name':'Audited original9B transport','model_providers.dataforge_native.base_url':f'http://127.0.0.1:{gateway.port}/v1',
        'model_providers.dataforge_native.wire_api':'responses','model_providers.dataforge_native.requires_openai_auth':False,
        'model_providers.dataforge_native.supports_websockets':False,'model_providers.dataforge_native.request_max_retries':0,'model_providers.dataforge_native.stream_max_retries':0,
        'mcp_servers.dataforge_learning.command':sys.executable,'mcp_servers.dataforge_learning.args':['-B',str(ROOT/'application/native_tools.py'),'mcp',str(session)],
        'mcp_servers.dataforge_learning.enabled_tools':['decide'],'mcp_servers.dataforge_learning.tools.decide.approval_mode':'approve',
        'mcp_servers.dataforge_learning.tool_timeout_sec':120,'mcp_servers.dataforge_learning.startup_timeout_sec':60,'mcp_servers.dataforge_learning.required':True}
    args=[a.codex,'exec','--json','--strict-config','--skip-git-repo-check','--ephemeral','--dangerously-bypass-hook-trust','-s','workspace-write','-C',str(cwd)]
    for key,value in overrides.items():args.extend(['-c',key+'='+json.dumps(value)])
    args.append('-');dump(session/'command_freeze.json',{'args':args,'code':{str(q):sha256(q) for folder in ['application','learning'] for q in (ROOT/folder).glob('*.py')},
        'raw_teacher_rows_visible_to_model':False,'test_gold_read':False,'actual_model':'Qwen3.5-9B','seed':20260927})
    started=time.monotonic()
    try:
        with (session/'native_events.jsonl').open('x') as out,(session/'native_stderr.log').open('x') as err:
            result=subprocess.run(args,input='Generate and evaluate new bounded candidate-selection expressions through the registered DataForge tool.\n',text=True,stdout=out,stderr=err,timeout=360)
        status=state(work);budget=load(session/'budget.json')
        receipt={'native_exit_code':result.returncode,'closed':status['closed'],'candidate':status['compiled'],'deferred':status['deferred'],
            'real9B_calls':budget['calls'],'actual_usage':budget['actual_usage'],'synthetic_control_count':gateway.controls,
            'raw_reference_cases_visible':False,'test_gold_read':False,'seconds':time.monotonic()-started,'gateway_failure':gateway.failed,
            'native_framework':'codex-cli0.153.1','scope':'native learning component; not a formal repair-performance score'}
        dump(session/'result.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
        if result.returncode or gateway.failed or not status['closed']:raise RuntimeError('Native learning failed; preserve receipts and logs')
    finally:gateway.close()

if __name__=='__main__':main()
