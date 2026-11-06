"""Separate synthetic Codex control traffic from actual 9B decisions."""
import json,time,threading,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import requests
from runtime import read,save
from protocol import Protocol,classify

class Gateway:
    def __init__(self,session,model):
        self.session=session;self.model=model;self.failed=None;self.lock=threading.Lock();self.controls=0
        protocol=Protocol(session);outer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def emit(self,output,usage=None):
                rid='resp_'+uuid.uuid4().hex
                response={'id':rid,'object':'response','created_at':int(time.time()),'status':'completed','model':model,'output':output,'usage':usage or {'input_tokens':0,'output_tokens':0,'total_tokens':0}}
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                events=[{'type':'response.created','response':{**response,'status':'in_progress','output':[]}}]
                for i,item in enumerate(output):
                    events.append({'type':'response.output_item.added','output_index':i,'item':{**item,'status':'in_progress'}})
                    events.append({'type':'response.output_item.done','output_index':i,'item':item})
                events.append({'type':'response.completed','response':response})
                try:
                    for seq,event in enumerate(events):
                        event['sequence_number']=seq
                        self.wfile.write(('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
                self.close_connection=True
                return response
            def text_item(self,text):
                return {'id':'msg_'+uuid.uuid4().hex,'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':text,'annotations':[]}]}
            def fail(self,code,message):
                self.send_response(code);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps({'error':{'message':message,'type':'invalid_request_error','code':'control_or_budget_error'}}).encode());self.close_connection=True
            def do_POST(self):
                codex=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                try:
                    purpose=classify(self.path,codex)
                    b=read(session/'budget.json')
                    if purpose=='compaction':
                        if b['pending_call'] is not None:raise RuntimeError('compaction requested before previous decision was processed')
                        outer.controls+=1;checkpoint=protocol.checkpoint()
                        save(session/f'control-checkpoint-{outer.controls}.json',{'request':codex,'checkpoint':checkpoint,'actual_budget_before':b})
                        self.emit([self.text_item(json.dumps(checkpoint,ensure_ascii=False))]);return
                    if protocol.state()['status']!='pending':
                        save(session/'control_closed_ack.json',{'provenance':'deterministic harness control, no model call','group_status':protocol.state()['status'],'codex_request':codex})
                        self.emit([self.text_item('The registered entity group is closed. Execution complete.')]);return
                    payload=protocol.payload();n=protocol.begin(payload)
                    if n is None:self.fail(400,read(session/'budget.json')['denied']);return
                    save(session/f'codex-request-{n}.json',codex)
                    wire={k:v for k,v in payload.items() if k!='thinking'};wire['chat_template_kwargs']={'enable_thinking':False}
                    save(session/f'wire-request-{n}.json',wire)
                    reply=requests.post('http://127.0.0.1:8000/v1/chat/completions',json=wire,timeout=(20,600));reply.raise_for_status();raw=reply.json()
                    decision=protocol.receive(n,raw)
                    if decision is None:output=[self.text_item(raw['choices'][0]['message'].get('content') or 'Invalid empty model output.')]
                    else:
                        namespace=next((t['name'] for t in codex['tools'] if t.get('type')=='namespace' and any(x.get('name')=='decide' for x in t.get('tools',[]))),'mcp__dataforge')
                        output=[{'id':'fc_'+uuid.uuid4().hex,'type':'function_call','call_id':'call_'+uuid.uuid4().hex,'namespace':namespace,'name':'decide','arguments':json.dumps({'decision':decision},ensure_ascii=False),'status':'completed'}]
                    # These are upper-bound estimates of synthetic control traffic,
                    # not actual model usage. Raw 9B usage remains in response/budget.
                    accounting={'input_tokens':len(json.dumps(codex,ensure_ascii=False).encode())+1024,'output_tokens':len(json.dumps(output,ensure_ascii=False).encode())+1024}
                    accounting['total_tokens']=sum(accounting.values())
                    emitted=self.emit(output,accounting)
                    save(session/f'codex-response-{n}.json',{'provenance':'JSON transport adapter; original 9B decision','control_usage_provenance':'UTF8 byte upper bound of synthetic control envelope; no Codex model inference','actual_model_usage':raw.get('usage',{}),'response':emitted})
                except Exception as exc:
                    outer.failed=f'{type(exc).__name__}: {exc}'
                    save(session/f'control_failure-{time.time_ns()}.json',{'error':outer.failed,'request':codex,'budget':read(session/'budget.json')})
                    self.fail(502,outer.failed)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    @property
    def port(self):return self.server.server_port
    def close(self):self.server.shutdown();self.server.server_close()
