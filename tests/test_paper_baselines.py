from pathlib import Path
import gzip,hashlib,importlib.util,json,socket,sys,tempfile,unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import paper_baselines as baselines
class FinalBaselineFilesTests(unittest.TestCase):
 def test_missing_source_prevents_preparation_or_model_start(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cfg=root/'machine.json';cfg.write_text(json.dumps({'baseline_sources':str(root/'absent')}))
   with patch.object(baselines.rt,'materialize') as materialize:
    with self.assertRaises(FileNotFoundError):baselines.prepare('tabclean_same9b',root/'run',cfg,root/'inputs')
    materialize.assert_not_called();self.assertFalse((root/'run').exists())
 def test_unknown_failure_blocks_before_reference_access(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'cases/x/status/worker_failure.json';p.parent.mkdir(parents=True);p.write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'Unreviewed failure'):baselines.verify(root)
 def test_four_method_scope_blocks_unused_garf(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'audit').mkdir();(root/'configs').mkdir()
   (root/'audit/portable_materialization.json').write_text(json.dumps({'study':'real_dirty'}));(root/'configs/cases.json').write_text('["x"]')
   with patch.object(baselines.subprocess,'run') as launch:
    with self.assertRaisesRegex(ValueError,'four final paper'):baselines.run_case(root,'x','garf',0)
    launch.assert_not_called()
 def test_live_model_blocks_scoring_before_reference_read(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for d in ['audit','configs','services/gpu0']:(root/d).mkdir(parents=True,exist_ok=True)
   (root/'audit/portable_materialization.json').write_text(json.dumps({'study':'real_dirty'}));(root/'configs/cases.json').write_text('["x"]')
   (root/'services/gpu0/owner.json').write_text(json.dumps({'host':socket.gethostname(),'pid':123,'name':'fixture','port':8911}))
   with self.assertRaisesRegex(RuntimeError,'Stop every owned service'):baselines.score(root)
   self.assertFalse((root/'audit/paper_fresh_scoring_started.json').exists())
 def test_published_row_configuration_contains_no_cell_values(self):
  p=R/'experiments/paper_runs/real_dirty/row_partition_config.json.gz';cfg=json.loads(gzip.decompress(p.read_bytes()));self.assertEqual(len(cfg['cases']),15)
  for name,c in cfg['cases'].items():
   for groups in c['groups'].values():
    for g in groups:self.assertLessEqual(set(g),{'group','provenance','source_indices'});self.assertTrue(all(type(i)is int for i in g['source_indices']))
   for part in ['public','private']:
    for filename,digest in c['expected_input_freeze'][part].items():self.assertRegex(digest,r'^[0-9a-f]{64}$')
   for rows in c['zone_rows'].values():
    for index,group in rows:self.assertIs(type(index),int);self.assertRegex(group,r'^g\d+$')
 def test_final_tc_fingerprints_and_catalog_use_the_same21_cases(self):
  folder=R/'experiments/paper_runs/tabclean_same9b';fps=json.loads((folder/'input_fingerprints.json').read_text())['cases'];scope=json.loads((folder/'config.json').read_text())
  self.assertEqual(set(fps),{x['id'] for x in scope['paper_case_specs']});self.assertEqual(len(fps),21)
if __name__=='__main__':unittest.main()
