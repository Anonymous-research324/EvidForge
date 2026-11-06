"""Operational seed override only; frozen V23 source files stay byte-identical."""
from pathlib import Path
import ast,json,hashlib,importlib.machinery,sys,os,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'frozen_workspace/DataForge_UnseenTable_v23'
OLD=20260927;NEW=1014
FIXED=json.loads((ROOT/'audit/frozen_v23_files.json').read_text())
KNOWN=set(FIXED['files'].values())
SOURCE_TO_CODE=importlib.machinery.SourceFileLoader.source_to_code
def checked_transform(data,path):
 blob=data.encode() if isinstance(data,str) else data
 h=hashlib.sha256(blob).hexdigest()
 if h not in KNOWN or not str(Path(path).resolve()).startswith(str(ROOT)+'/'):return None
 tree=ast.parse(blob,filename=str(path));count=0
 class Seeds(ast.NodeTransformer):
  def visit_Constant(self,node):
   nonlocal count
   if type(node.value) is int and node.value==OLD:count+=1;return ast.copy_location(ast.Constant(value=NEW),node)
   return node
 tree=Seeds().visit(tree);ast.fix_missing_locations(tree)
 with (ROOT/'audit/seed_imports.jsonl').open('a') as f:f.write(json.dumps({'path':str(path),'source_sha256':h,'integer_seed_overrides':count,'pid':os.getpid()})+'\n')
 return compile(tree,str(path),'exec',dont_inherit=True)
def source_to_code(self,data,path,*,_optimize=-1):
 x=checked_transform(data,path)
 return x if x is not None else SOURCE_TO_CODE(self,data,path,_optimize=_optimize)
def install():
 sys.dont_write_bytecode=True
 import random,numpy
 random.seed(NEW);numpy.random.seed(NEW)
 sys.path[:0]=[str(ENGINE),str(ENGINE/'source_runtime'),str(ENGINE/'application')]
 importlib.machinery.SourceFileLoader.source_to_code=source_to_code
 sys.executable=str(ROOT/'code/v23-python')
 import skill_runtime.client as client
 from unseen_table.common import dump
 def local_call(system,payload,schema,path,tag,columns):
  client.verify_request(payload,columns);path=Path(path)
  body={'_trace_tag':tag,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],
      'seed':NEW,'temperature':0,'max_tokens':int(os.environ.get('DF1014_TOOL_MAX_TOKENS','4096')),'structured_outputs':{'json':schema,'disable_any_whitespace':True},'chat_template_kwargs':{'enable_thinking':False}}
  dump(path.with_name(path.name+'_request.json'),body);t=time.monotonic()
  proc=subprocess.run(['@RUN_PYTHON@','-B',str(ROOT/'code/model_service.py'),'call'],
      input=json.dumps(body),capture_output=True,text=True,timeout=330)
  if proc.returncode:dump(path.with_name(path.name+'_failed.json'),{'stderr':proc.stderr});raise RuntimeError(proc.stderr)
  response=json.loads(proc.stdout);dump(path.with_name(path.name+'_response.json'),response)
  dump(path.with_name(path.name+'_timing.json'),{'seconds':time.monotonic()-t,'usage':response.get('usage'),'effective_seed':NEW,'original_reference_cases_visible':False})
  if response['choices'][0]['finish_reason']!='stop':raise ValueError('Truncated model response')
  return client.strict_json(response['choices'][0]['message']['content'])
 client.call=local_call
def main():
 args=sys.argv[1:]
 while args and args[0] in ['-B','-u']:args.pop(0)
 if not args:raise ValueError('An explicit V23 script is required')
 script=Path(args.pop(0)).resolve();blob=script.read_bytes()
 code=checked_transform(blob,script)
 if code is None:
  if script==ROOT/'code/model_service.py':code=compile(blob,str(script),'exec')
  else:raise ValueError('Only manifest-verified V23 entries and the audited local transport are permitted')
 sys.argv=[str(script),*args];install()
 exec(code,{'__name__':'__main__','__file__':str(script),'__package__':None})
if __name__=='__main__':main()
