from pathlib import Path
import importlib.util,json,tempfile,unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('owned_model_service',R/'scripts/owned_model_service.py');service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
class OwnedServiceTests(unittest.TestCase):
 def test_small_memory_with_live_compute_process_is_still_occupied(self):
  results=['0, GPU-fixture, 1\n','GPU-fixture, 123\n']
  with patch.object(service.subprocess,'check_output',side_effect=results):
   with self.assertRaisesRegex(RuntimeError,'occupied'):service.inventory([0])
 def test_wrong_host_never_sends_signal(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'services/gpu0/owner.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'host':'another-machine','pid':123}))
   with patch.object(service.socket,'gethostname',return_value='this-machine'),patch.object(service.os,'killpg') as kill:
    with self.assertRaisesRegex(RuntimeError,'Wrong host'):service.stop(root,0)
    kill.assert_not_called()
if __name__=='__main__':unittest.main()
