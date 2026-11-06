from pathlib import Path
import hashlib, importlib.util, io, json, tempfile, tarfile, unittest

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduction_data", R / "scripts/reproduction_data.py")
data = importlib.util.module_from_spec(spec); spec.loader.exec_module(data)

class ReproductionDataTests(unittest.TestCase):
    def make_source(self, base):
        root = base / "old"; c = root / "cases/case1"
        (c / "public").mkdir(parents=True); (c / "private").mkdir()
        (c / "public/repair_dirty.csv").write_text("id,x\na,ERR12\n")
        (c / "private/repair_reference.csv").write_text("id,x\na,12\n")
        freeze = {}
        for zone in ["public", "private"]:
            freeze[zone] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (c / zone).iterdir()}
        (c / "input_freeze.json").write_text(json.dumps(freeze)); (c / "case.json").write_text('{"id":"case1","seed":1}')
        (c / "work").mkdir(); (c / "work/leaked_model.json").write_text('{"should_not_export":true}')
        (c / "scores.json").write_text('{"should_not_export":true}')
        cat = base / "catalog.json"; cat.write_text(json.dumps({"studies":[{"study":"s","server_root":str(root),"case_ids":["case1"],"kind":"case_inputs"}]}))
        return root, cat

    def test_exact_inputs_no_scores_or_models_and_fresh_destination(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); root, cat = self.make_source(base)
            result = data.export_server(cat, "s", base / "bundle")
            self.assertEqual(result["model_calls"], 0); self.assertEqual(result["fit_calls"], 0)
            self.assertEqual(result["original_freeze_checks"], 2)
            manifest = data.inspect_bundle(base / "bundle")
            self.assertFalse(any("scores" in x["path"] or "/work/" in x["path"] for x in manifest["files"]))
            prepared = data.prepare(base / "bundle", base / "fresh")
            self.assertTrue(prepared["original_input_bytes_unchanged"])
            for item in manifest["files"]:
                self.assertEqual((root/item["path"]).read_bytes(), (base/"fresh"/item["path"]).read_bytes())
            self.assertFalse((base / "fresh/cases/case1/status/scored.json").exists())
            with self.assertRaises(FileExistsError): data.prepare(base / "bundle", base / "fresh")

    def test_original_tamper_prevents_export(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); root,cat=self.make_source(base)
            (root/"cases/case1/public/repair_dirty.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError,"frozen input differs"): data.export_server(cat,"s",base/"bundle")

    def test_archive_tamper_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);_,cat=self.make_source(base);data.export_server(cat,"s",base/"bundle")
            mp=base/"bundle/input_manifest.json";m=json.loads(mp.read_text());archive=base/"bundle"/m["archive"]
            archive.write_bytes(archive.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"archive SHA"):data.inspect_bundle(base/"bundle")
            with self.assertRaises(ValueError):data.safe_name("../../leak")
            with self.assertRaises(ValueError):data.safe_name("/absolute")

    def test_export_cannot_write_inside_original(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,cat=self.make_source(base)
            with self.assertRaisesRegex(ValueError,"new sibling"):data.export_server(cat,"s",root/"new")

    def test_imported_frozen_input_alias_requires_catalog_allowlist(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,cat=self.make_source(base)
            original=root/"cases/case1/public";parent=base/"original_imported_case";parent.mkdir()
            original.rename(parent/"public");original.symlink_to(parent/"public",target_is_directory=True)
            with self.assertRaisesRegex(ValueError,"Unapproved input alias"):
                data.export_server(cat,"s",base/"unapproved")
            c=json.loads(cat.read_text());c['studies'][0]['allowed_input_alias_roots']=[str(parent)];cat.write_text(json.dumps(c))
            result=data.export_server(cat,"s",base/"approved")
            self.assertEqual(result['original_freeze_checks'],2)
            self.assertEqual(result['fit_calls'],0)

if __name__ == "__main__": unittest.main()
