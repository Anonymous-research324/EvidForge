from pathlib import Path
import json,os,sys,tempfile,unittest,subprocess
from unittest.mock import patch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
import supplement_runtime as runtime
import owned_model_service as owned

class ReuseBatchTests(unittest.TestCase):
 def test_owned_service_name_matches_archived_reuse_transport(self):
  import ast,tarfile
  with tarfile.open(R/'experiments/paper_runs/sealed_reuse/execution_code.tar.gz') as tf:
   source=tf.extractfile('code/model_service.py').read().decode()
  expression=next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME' for t in n.targets))
  for gpu in [0,1]:
   expected=eval(compile(ast.Expression(expression),'transport-name','eval'),{'__builtins__':{'str':str},'GPU':gpu})
   self.assertEqual(owned.served_name('sealed_reuse',gpu),expected)
 def test_all_learning_and_shutdown_precede_query_execution_and_scoring(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);events=[];names=[]
   def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
   for folder in ['audit','logs','status','services','reports','code']:(root/folder).mkdir()
   for seed in [1013,1014,1015]:
    for dataset,gpu in [('retinopathy',0),('airfoil',1)]:
     name=dataset+'_s'+str(seed);names.append(name);case=root/'cases'/name
     for folder in ['work','outputs','status','sealed']:(case/folder).mkdir(parents=True)
     save(case/'case.json',{'dataset':dataset,'seed':seed,'gpu':gpu})
   save(root/'audit/portable_materialization.json',{'study':'sealed_reuse','case_names':names,'machine':{'run_python':sys.executable}})
   def start(path,gpu):
    events.append(('start',gpu));save(path/'services'/('gpu'+str(gpu))/'owner.json',{})
   def stop(path,gpu):
    events.append(('stop',gpu));save(path/'services'/('gpu'+str(gpu))/'stopped.json',{})
   def learn(path,name,gpu):
    events.append(('learn',name));case=path/'cases'/name
    save(case/'status/learning_closed.json',{});save(case/'sealed/capsule.json',{})
    for stage in ['prior','context']:
     tag=name+stage;save(case/'work/full'/(stage+'_stage/native_v3/session.json'),{'tag':tag})
     for i in range(1,6):
      request=path/'services'/('gpu'+str(gpu))/'requests'/(tag+'call'+str(i))
      save(request/'request.json',{});save(request/'response.json',{'usage':{}});save(request/'timing.json',{'seconds':1.0})
   def run(cmd,**kwargs):
    entry=Path(cmd[2]).name
    if entry=='execute_reuse.py':
     self.assertEqual(sum(e[0]=='learn' for e in events),6)
     self.assertEqual(sum(e[0]=='stop' for e in events),2)
     self.assertTrue((root/'status/inference_disabled.json').exists())
     events.append(('execute',Path(cmd[-1]).name))
    else:
     self.assertEqual(sum(e[0]=='execute' for e in events),6)
     events.append((entry,None));save(root/'reports/all_scores.json',[{}]*18)
    return subprocess.CompletedProcess(cmd,0)
   before=Path.cwd()
   try:
    with patch.object(runtime,'doctor',return_value={'missing':[],'unresolved_absolute_paths':[]}),patch.object(runtime,'run_case',side_effect=learn),patch.object(runtime,'audit_case',side_effect=lambda path,name:{'case':name,'real_calls':10}),patch.object(owned,'start',side_effect=start),patch.object(owned,'stop',side_effect=stop),patch.object(runtime.subprocess,'run',side_effect=run),patch.dict(os.environ):
     with self.assertRaisesRegex(ValueError,'call limit'):runtime.run_reuse_batch(root,59)
     self.assertEqual(events,[])
     result=runtime.run_reuse_batch(root,60)
   finally:os.chdir(before)
   self.assertEqual(result['batches'],18);self.assertEqual(result['real_model_calls'],60)
   self.assertEqual(events[-2:],[('score_reuse.py',None),('report_reuse.py',None)])

if __name__=='__main__':unittest.main()
