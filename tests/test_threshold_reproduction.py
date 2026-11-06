from pathlib import Path
import importlib.util,json,sys,tempfile,unittest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
spec=importlib.util.spec_from_file_location('reproduce_threshold',R/'scripts/reproduce_threshold.py');threshold=importlib.util.module_from_spec(spec);spec.loader.exec_module(threshold)
class ThresholdReproductionTests(unittest.TestCase):
 def test_missing_fresh_main_does_not_launch_or_create_run(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);m=base/'machine.json';m.write_text(json.dumps({'run_python':sys.executable}));out=base/'scan'
   with self.assertRaisesRegex(ValueError,'Fresh main ablation must be complete'):threshold.prepare(base/'main',out,m)
   self.assertFalse(out.exists())
 def test_external_historical_skill_cannot_be_substituted(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);m=base/'machine.json';m.write_text(json.dumps({'run_python':sys.executable}));main=base/'main/ablation_seed1013'
   (main/'status').mkdir(parents=True);(main/'status/scoring_complete.json').write_text('{}')
   ds='tic_tac_toe';p=main/'work'/ds/'A/selection.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'skill':str(base/'historical/skill'),'passed':True}))
   with self.assertRaisesRegex(ValueError,'A skill must belong'):threshold.prepare(base/'main',base/'scan',m)
   self.assertFalse((base/'scan').exists())
if __name__=='__main__':unittest.main()
