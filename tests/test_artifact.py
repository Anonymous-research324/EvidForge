from pathlib import Path
import json,sys,tempfile,subprocess,unittest,hashlib,statistics
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'experiments'),str(R/'scripts')]
from tables import stats,build
from release_runtime import initialize,settings,check_engine
class ArtifactTests(unittest.TestCase):
 def test_core_and_environment_unchanged(self):
  self.assertEqual(check_engine(),124)
  for rel,h in json.loads((R/'provenance/environment_sha256.json').read_text())['files'].items():self.assertEqual(hashlib.sha256((R/rel).read_bytes()).hexdigest(),h,rel)
 def test_sample_sd_missing_not_zero(self):
  self.assertEqual(stats([1.,2.,3.])['sample_sd'],1.)
  self.assertIsNone(stats([1.,None,3.])['mean'])
 def test_materialized_worker_denies_private_and_y(self):
  with tempfile.TemporaryDirectory() as td:
   root=initialize(Path(td)/'runs','repair',1014,settings(None))
   paths=[root/'datasets/example/private/probe.txt',root/'curator_cache/private/probe.txt',root/'downstream_training/labels.csv',root/'datasets/example/public/holdout_dirty.csv']
   for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
   code='import sys;sys.path.insert(0,'+repr(str(root/'code'))+');from role_guard import install;install()\n'
   code+='for name in '+repr(list(map(str,paths)))+":\n try:open(name).read();raise AssertionError('Private read permitted')\n except PermissionError:pass\n"
   subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True,text=True)
 def test_all_selected_seeds_and_missing_qsar(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);build(R,out);data=json.loads((out/'aggregate.json').read_text())['records']
   by={(x['study'],x['dataset'],x['method']):x for x in data}
   q=by['repair','qsar','cdc']['metrics']['F1'];self.assertEqual(q['n'],2);self.assertIsNone(q['mean'])
   ab=[x for x in data if x['study']=='ablation'];self.assertEqual(len(ab),6)
   self.assertEqual({x['dataset'] for x in ab},{'retinopathy','airfoil'});self.assertEqual({x['method'] for x in ab},{'A','B','D'})
   for x in data:
    for s in x['metrics'].values():
     vals=list(s['per_seed'].values())
     if all(v is not None for v in vals):self.assertAlmostEqual(s['sample_sd'],statistics.stdev(vals),places=12)
 def test_scope_adapter_keeps_original_metric_code(self):
  import tarfile
  from paper_scope import runtime_source
  for study in ['advisor_supplement','robustness_retinopathy']:
   with tarfile.open(R/'experiments/paper_runs'/study/'execution_code.tar.gz') as tf:
    s=tf.extractfile('code/score.py').read().decode()
    self.assertEqual(runtime_source(study,'score.py',s),(s,[]))
    s=tf.extractfile('code/worker_v4.py').read().decode();bound,changes=runtime_source(study,'worker_v4.py',s)
    self.assertTrue(changes);self.assertIn('"arms":["full"]',bound)
 def test_threshold_metadata_uses_selected_output_count(self):
  import ast,tarfile
  from paper_scope import runtime_source
  with tarfile.open(R/'experiments/paper_runs/threshold/execution_code.tar.gz') as tf:
   original=tf.extractfile('code/score_once.py').read().decode()
  scoped,changes=runtime_source('threshold','score_once.py',original)
  self.assertTrue(changes);self.assertNotIn('all90_outputs_closed',scoped)
  metric=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='metrics'))
  self.assertEqual(metric(original),metric(scoped))
  self.assertIn('"outputs":len(cases)*len(OFFSETS)',scoped)
if __name__=='__main__':unittest.main()
