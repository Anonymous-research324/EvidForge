"""The final paper's Hook-off arm comes from original recorded F, never copied A."""
from pathlib import Path
import json,unittest,ast
R=Path(__file__).resolve().parents[1]
class HookAblationTests(unittest.TestCase):
 def test_all18_F_records_match_the_original_aggregate(self):
  source=json.loads((R/'results/studies/module_ablation/three_seed_ablation_aggregate.json').read_text())['records'];count=0
  for seed in [1013,1014,1015]:
   rows=json.loads((R/'results/recorded'/f'ablation_seed{seed}.json').read_text());f=[x for x in rows if x['method']=='F'];self.assertEqual(len(f),6)
   for row in f:
    original=next(x for x in source if x['dataset']==row['dataset'] and x['arm']=='F')['per_seed_records'][str(seed)]
    self.assertEqual(row['repair'],original['repair']);self.assertEqual(row['downstream'],original['downstream']);count+=1
  self.assertEqual(count,18)
 def test_F_runs_a_separate_native_variant(self):
  protocol=json.loads((R/'experiments/protocol.json').read_text());self.assertEqual(protocol['ablation_arms'],list('ABCDEF'))
  text=(R/'experiments/templates/ablation/learning_flow.py').read_text();self.assertIn("'F':'no_hook'",text);self.assertIn("variant!='F'",text)
  native=(R/'experiments/templates/ablation/run_native_variant.py').read_text();self.assertIn("EVIDFORGE_NATIVE_VARIANT",native);self.assertIn("!='no_hook'",native)
if __name__=='__main__':unittest.main()
