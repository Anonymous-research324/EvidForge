from pathlib import Path
import importlib.util,json,tempfile,unittest,sys,tarfile,io,hashlib,os
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplement_runtime',REPO/'scripts/supplement_runtime.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
class PortableRuntimeTests(unittest.TestCase):
 def test_only_declared_string_and_port_bindings(self):
  source="SEED=20260927\nTHRESHOLD=.98\nCALLS=5\nPORT=8181+GPU\nMODEL='/old/model'\nGOLD='/other/private/truth.csv'\n"
  new,changes=runtime.strings_rebind(source,'fixture.py',{'/old':'/new'},8431)
  self.assertIn('SEED = 20260927',new);self.assertIn('THRESHOLD = 0.98',new);self.assertIn('CALLS = 5',new)
  self.assertIn('PORT = 8431 + GPU',new);self.assertIn("MODEL = '/new/model'",new);self.assertIn("GOLD = '/other/private/truth.csv'",new)
  self.assertEqual(len(changes),2)
 def test_cold_start_wait_does_not_change_request_budget(self):
  source="def start():\n for _ in range(300):pass\ndef call():\n attempts=5\n max_tokens=8192\n"
  new,changes=runtime.strings_rebind(source,"service.py",{},8431,900)
  self.assertIn("range(900)",new);self.assertIn("attempts = 5",new);self.assertIn("max_tokens = 8192",new)
  self.assertEqual(len(changes),1)
 def test_path_binding_longest_prefix(self):
  new,_=runtime.strings_rebind("P='/a/b/c.csv'",'fixture.py',{'/a':'/x','/a/b':'/y'})
  self.assertIn("'/y/c.csv'",new)
 def test_no_formal_truth_read_in_audit_parent(self):
  # Audit loads private hashes, never references. Only unchanged scorer reads truth.
  import inspect
  text=inspect.getsource(runtime.audit_case)
  self.assertNotIn('repair_reference.csv',text);self.assertNotIn('read_csv',text)
  self.assertIn("budget['pending_call'] is None",text);self.assertIn("read(service/'owner.json')",text)
 def test_input_hash_and_history_rejection(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'configs').mkdir();p=root/'configs/cases.json';p.write_text('["c"]');h=hashlib.sha256(p.read_bytes()).hexdigest()
   (root/'input_export.json').write_text(json.dumps({'study':'llm_digit70','files':{'configs/cases.json':h}}))
   self.assertEqual(runtime.input_check(root)['study'],'llm_digit70')
   p.write_text('[]')
   with self.assertRaisesRegex(ValueError,'mismatch'):runtime.input_check(root)
   (root/'input_export.json').write_text(json.dumps({'files':{'cases/c/outputs/full/output.csv':'a'*64}}))
   with self.assertRaisesRegex(ValueError,'Historical execution'):runtime.input_check(root)
 def test_archive_path_traversal_denied(self):
  payload=io.BytesIO()
  with tarfile.open(fileobj=payload,mode='w:gz') as tf:
   item=tarfile.TarInfo('../escape.py');item.size=1;tf.addfile(item,io.BytesIO(b'x'))
  payload.seek(0)
  with tarfile.open(fileobj=payload,mode='r:gz') as tf:
   with self.assertRaisesRegex(ValueError,'Unsafe source'):list(runtime.safe_members(tf))
 def test_missing_frozen_raw_archive_blocks_before_materialization(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'configs').mkdir();case=root/'cases/x';case.mkdir(parents=True)
   (root/'configs/cases.json').write_text('["x"]')
   (case/'input_freeze.json').write_text(json.dumps({'private':{'source.zip':'a'*64}}))
   (root/'input_export.json').write_text(json.dumps({'study':'llm_digit70','files':{'configs/cases.json':hashlib.sha256((root/'configs/cases.json').read_bytes()).hexdigest()}}))
   with self.assertRaisesRegex(ValueError,'frozen input differs'):runtime.input_check(root)
 def test_original_shell_wrapper_is_replaced_with_one_valid_hash_record(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);repo=base/'repo';(repo/'engine').mkdir(parents=True);(repo/'provenance').mkdir();archive=repo/'experiments/paper_runs/llm_digit70/execution_code.tar.gz';archive.parent.mkdir(parents=True)
   source={'code/runtime_v4.py':b'print("fixture")\n','code/ef-python-v4':b'#!/bin/sh\nexec /old/python "$@"\n','code/model_service.py':b'GPU=0\nPORT=8181+GPU\ndef start():\n for _ in range(300):pass\n'}
   with tarfile.open(archive,'w:gz') as tf:
    for name,blob in source.items():
     member=tarfile.TarInfo(name);member.size=len(blob);tf.addfile(member,io.BytesIO(blob))
   spec={'id':'llm_digit70','server_root':'/old/run','execution_code_files':[{'path':n,'sha256':hashlib.sha256(b).hexdigest()} for n,b in source.items()]}
   (repo/'provenance/paper_experiments.json').write_text(json.dumps({'studies':[spec]}))
   inputs=base/'inputs';(inputs/'configs').mkdir(parents=True);files={}
   for rel,text in {'configs/cases.json':'["x"]',**{'cases/x/'+n:'{}' for n in ['case.json','input_freeze.json','public/development_bundle.json','public/declaration.json','public/context.json','public/repair_dirty.csv']}}.items():
    f=inputs/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text);files[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
   (inputs/'input_export.json').write_text(json.dumps({'study':'llm_digit70','files':files}));model=base/'model';model.mkdir();machine=base/'machine.json';machine.write_text(json.dumps({'run_python':sys.executable,'infer_python':sys.executable,'codex':sys.executable,'model_dir':str(model)}))
   with patch.object(runtime,'REPO',repo),patch.object(runtime,'core_check',return_value={}):result=runtime.materialize('llm_digit70',base/'run',machine,inputs)
   self.assertFalse(result['missing']);self.assertFalse(result['missing_display_fonts']);meta=json.loads((base/'run/audit/portable_materialization.json').read_text());records=[x for x in meta['adapted_files'] if x['path']=='code/ef-python-v4']
   self.assertEqual(len(records),1);self.assertEqual(records[0]['original_sha256'],hashlib.sha256(source['code/ef-python-v4']).hexdigest())
 def test_unknown_study_fails_without_partial_directory(self):
  with tempfile.TemporaryDirectory() as td:
   run=Path(td)/'run'
   with self.assertRaises(NotImplementedError):runtime.materialize('not-implemented',run,'machine','inputs')
   self.assertFalse(run.exists())
 def test_batch_budget_and_partial_state_block_before_service(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'audit').mkdir();case=root/'cases/x'
   for d in ['work','outputs','status']:(case/d).mkdir(parents=True)
   (case/'case.json').write_text(json.dumps({'suite':'main'}))
   (root/'audit/portable_materialization.json').write_text(json.dumps({'study':'advisor_supplement','case_names':['x']}))
   with patch.object(runtime,'doctor',return_value={'missing':[],'unresolved_absolute_paths':[]}):
    with self.assertRaisesRegex(ValueError,'exceed explicit limit'):runtime.run_batch(root,0,9)
    (case/'work/partial').write_text('preserve')
    with self.assertRaisesRegex(ValueError,'cannot be replayed'):runtime.run_batch(root,0,100)
   self.assertFalse((root/'status/portable_batch_started.json').exists())
   self.assertEqual((case/'work/partial').read_text(),'preserve')
 def test_original_case_arguments_do_not_use_retired_controller(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'audit').mkdir();(root/'audit/portable_materialization.json').write_text(json.dumps({'study':'llm_digit70','machine':{'run_python':sys.executable},'case_names':['x']}))
   cmd=runtime.case_command(root,'x',0);self.assertEqual(cmd[2],str(root.resolve()/'code/worker.py'));self.assertIn('case',cmd)
   with self.assertRaisesRegex(ValueError,'Unregistered'):runtime.case_command(root,'old-pilot',0)
if __name__=='__main__':unittest.main()
