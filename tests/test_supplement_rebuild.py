from pathlib import Path
import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
import rebuild_supplement_inputs as rebuild

class SupplementRebuildTests(unittest.TestCase):
 def test_only_fixed_case_parameters_and_hashes_are_published(self):
  spec=json.loads((R/'provenance/supplement_rebuild_specs.json').read_text())
  catalog=json.loads((R/'provenance/reproduction_data_sources.json').read_text())
  self.assertEqual(sum(len(x['cases']) for x in spec['studies'].values()),54)
  for study,info in spec['studies'].items():
   self.assertEqual(info['case_ids'],next(x['case_ids'] for x in catalog['studies'] if x['study']==study))
   for item in info['cases'].values():
    self.assertLessEqual(set(item),{'parameters','expected_original_input_sha256','expected_deterministic_case_sha256'})
    for hashes in item['expected_original_input_sha256'].values():
     for h in hashes.values():self.assertRegex(h,r'^[0-9a-f]{64}$')
 def test_changed_scientific_cells_block_even_with_fresh_cache_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   case=Path(td);(case/'private').mkdir();ref=case/'private/repair_reference.csv';ref.write_text('x\n12\n')
   (case/'private/source_provenance.json').write_text('{"fresh_timestamp":1}')
   expected={'private':{'repair_reference.csv':rebuild.sha(ref),'source_provenance.json':'a'*64}}
   checked,fresh=rebuild.verify_scientific(case,expected)
   self.assertEqual(checked,1);self.assertEqual(fresh,['private/source_provenance.json'])
   ref.write_text('x\n13\n')
   with self.assertRaisesRegex(ValueError,'scientific input bytes differ'):rebuild.verify_scientific(case,expected)
 def test_raw_archive_hash_mismatch_blocks_curator(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'provenance').mkdir();raw=root/'raw';raw.mkdir();(raw/'concrete.zip').write_bytes(b'tampered')
   (root/'provenance/raw_sources.json').write_text(json.dumps({'concrete':{'sha256':hashlib.sha256(b'official raw').hexdigest()}}))
   with patch.object(rebuild,'REPO',root):
    with self.assertRaisesRegex(ValueError,'Official raw source absent/different'):rebuild.official_cache(root/'scratch','concrete',raw)
   self.assertFalse((root/'scratch').exists())

if __name__=='__main__':unittest.main()
