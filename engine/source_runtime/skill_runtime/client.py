"""One audited9B connection; seed/visibility fixed, no raw teaching data."""
import json,subprocess,time
from pathlib import Path
from unseen_table.common import dump
from .package import verify_request
REMOTE='/root/fsas/home/anonymous/experiments/dataforge_redo_learning_9b_20261012/remote_probe.py'

def strict_json(text):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    return json.loads(text,object_pairs_hook=pairs)

def call(system,payload,schema,path,tag,columns):
    verify_request(payload,columns);path=Path(path)
    body={'_trace_tag':tag,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],
          'seed':20260927,'temperature':0,'max_tokens':4096,'structured_outputs':{'json':schema,'disable_any_whitespace':True},'chat_template_kwargs':{'enable_thinking':False}}
    dump(path.with_name(path.name+'_request.json'),body);start=time.monotonic()
    proc=subprocess.run(['ssh','-o','BatchMode=yes','a100','/root/fsas/home/anonymous/envs/dataforge-infer/bin/python -B '+REMOTE+' call'],input=json.dumps(body),capture_output=True,text=True,timeout=330)
    if proc.returncode:dump(path.with_name(path.name+'_failed.json'),{'stderr':proc.stderr});raise RuntimeError(proc.stderr)
    response=json.loads(proc.stdout);dump(path.with_name(path.name+'_response.json'),response);dump(path.with_name(path.name+'_timing.json'),{'seconds':time.monotonic()-start,'usage':response.get('usage'),'original_reference_cases_visible':False})
    if response['choices'][0]['finish_reason']!='stop':raise ValueError('Model answer truncated; no partial patch is accepted')
    return strict_json(response['choices'][0]['message']['content'])
