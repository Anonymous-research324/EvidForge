#!/usr/bin/env python3
"""Prepare immutable paper inputs without fitting, model calls, or reading result metrics.

Two deliberately separate roles:
* export-server: curator-only, reads original frozen inputs from a research server.
* prepare: verifies a curator export, then materializes a fresh independent run root.

This tool never downloads or silently reconstructs a different split. Data bundles are
not uploaded automatically: redistribution rights must be established by the author.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, os, shutil, tarfile, tempfile, urllib.request

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "provenance/reproduction_data_sources.json"
ROLES = REPO / "provenance/reproduction_public_roles_seed1013.json"
UCI = {
    "tic_tac_toe": (101, "tic+tac+toe+endgame"),
    "car_evaluation": (19, "car+evaluation"),
    "retinopathy": (329, "diabetic+retinopathy+debrecen"),
    "real_estate": (477, "real+estate+valuation+data+set"),
    "concrete": (165, "concrete+compressive+strength"),
    "airfoil": (291, "airfoil+self+noise"),
    "auto_mpg": (9, "auto+mpg"),
    "spectf": (96, "spectf+heart"),
}

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")

def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or any(x in {"", "."} for x in p.parts):
        raise ValueError("Unsafe artifact path: " + name)
    return p

def study_spec(catalog, study):
    rows = [x for x in catalog["studies"] if x["study"] == study]
    if len(rows) != 1:
        raise ValueError("Unknown study: " + study)
    return rows[0]

def source_paths(root, spec):
    """Only declared inputs, no work/models/outputs/scores/logs or fitted skills."""
    result = {}
    for case in spec["case_ids"]:
        safe_name(case)
        c = root / "cases" / case
        if not c.is_dir():
            raise FileNotFoundError("Frozen input case absent: " + str(c))
        for name in ["case.json", "input_freeze.json"]:
            p = c / name
            if p.is_file(): result[str(p.relative_to(root))] = p
        directories = ["public", "private"]
        if spec["study"] == "sealed_reuse": directories += ["source", "batches"]
        for folder in directories:
            q = c / folder
            if not q.exists(): continue
            if q.is_symlink():
                # The final real-dirty protocol imported 25 frozen outputs and
                # their unchanged paired inputs from a named prior root.
                # Resolve only catalog-pinned read-only input aliases, never an
                # arbitrary method/model directory or an unverified target.
                resolved = q.resolve(strict=True)
                allowed = [Path(x).resolve() for x in spec.get("allowed_input_alias_roots", [])]
                if not any(resolved == base or base in resolved.parents for base in allowed):
                    raise ValueError("Unapproved input alias: " + str(q))
            for p in sorted(q.rglob("*")):
                if not p.is_file(): continue
                if p.is_symlink(): raise ValueError("Input file symlink rejected: " + str(p))
                if p.suffix not in {".json", ".csv", ".txt"}: continue
                result[str(p.relative_to(root))] = p
        if not any(k.startswith("cases/" + case + "/public/") or k.startswith("cases/" + case + "/source/") for k in result):
            raise ValueError("Public method inputs absent: " + case)
    # Only static scientific configuration; never dispatch/PID/service ownership.
    for name in ["protocol.json", "controlled_protocol.json", "semantic_cases.json", "run_config.json", "model-files-9B.sha256"]:
        p = root / "configs" / name
        if p.is_file(): result[str(p.relative_to(root))] = p
    return result

def verify_original_freeze(root, spec):
    """Read only input-freeze declarations, then hash the declared input bytes."""
    count = 0
    for name in spec["case_ids"]:
        c = root / "cases" / name
        p = c / "input_freeze.json"
        if not p.exists(): continue
        f = load(p)
        for zone in ["public", "private"]:
            entries = f.get(zone, {})
            if not isinstance(entries, dict): continue
            for rel, h in entries.items():
                if not isinstance(h, str) or len(h) != 64: continue
                safe_name(rel); q = c / zone / rel
                if not q.is_file() or digest(q.read_bytes()) != h:
                    raise ValueError("Original frozen input differs: " + str(q))
                count += 1
    return count

def export_server(catalog_path, study, output, source_root=None):
    catalog = load(catalog_path); spec = study_spec(catalog, study)
    if spec["kind"] != "case_inputs":
        raise ValueError(spec["kind"] + ": " + spec["dependency_note"])
    root = Path(source_root or spec["server_root"]).resolve()
    out = Path(output).resolve()
    # Export destination cannot be inside the original research directory.
    if out == root or root in out.parents: raise ValueError("Export must use a new sibling staging directory")
    if out.exists(): raise FileExistsError("Never overwrite an export: " + str(out))
    original_checks = verify_original_freeze(root, spec)
    files = source_paths(root, spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    target = out / (study + ".inputs.tar.gz")
    manifest = {
        "format_version": 1, "study": study, "case_ids": spec["case_ids"],
        "source_root": str(root), "original_input_freeze_checks": original_checks,
        "catalog_sha256": digest(Path(catalog_path).read_bytes()),
        "source_role": "curator_export", "new_fits": 0, "new_model_calls": 0,
        "scores_or_fitted_models_included": False,
        "contains_evaluation_references": True,
        "redistribution": "NOT_AUTO_AUTHORIZED: author must verify source dataset redistribution terms before publishing this bundle",
        "files": [],
    }
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for rel, p in sorted(files.items()):
            safe_name(rel); data = p.read_bytes()
            info = tarfile.TarInfo(rel); info.size = len(data); info.mode = 0o600
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(data))
            manifest["files"].append({"path": rel, "bytes": len(data), "sha256": digest(data)})
    manifest.update(archive=target.name, archive_sha256=digest(target.read_bytes()), archive_bytes=target.stat().st_size)
    write(out / "input_manifest.json", manifest)
    return {"study": study, "cases": len(spec["case_ids"]), "files": len(files),
            "archive_bytes": target.stat().st_size, "original_freeze_checks": original_checks,
            "manifest": str(out / "input_manifest.json"), "model_calls": 0, "fit_calls": 0}

def inspect_bundle(bundle_dir, expected_manifest_sha256=None):
    base = Path(bundle_dir).resolve(); mp = base / "input_manifest.json"
    if expected_manifest_sha256 and digest(mp.read_bytes()) != expected_manifest_sha256:
        raise ValueError("Input manifest is not the pinned manifest")
    m = load(mp); safe_name(m["archive"])
    archive = base / m["archive"]
    if digest(archive.read_bytes()) != m["archive_sha256"]:
        raise ValueError("Input archive SHA-256 mismatch")
    expected = {x["path"]: x for x in m["files"]}
    if len(expected) != len(m["files"]): raise ValueError("Duplicate declared input path")
    seen = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            safe_name(member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected/duplicate archive member: " + member.name)
            data = tf.extractfile(member).read(); item = expected[member.name]
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError("Input member mismatch: " + member.name)
            seen.add(member.name)
    if seen != set(expected): raise ValueError("Input archive member missing")
    return m

def prepare(bundle_dir, work, expected_manifest_sha256=None):
    m = inspect_bundle(bundle_dir, expected_manifest_sha256)
    work = Path(work).resolve()
    if work.exists(): raise FileExistsError("Fresh directory required; never overwrite prior inputs: " + str(work))
    work.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=work.name + ".preparing-", dir=work.parent))
    try:
        with tarfile.open(Path(bundle_dir) / m["archive"], "r:gz") as tf:
            for member in tf:
                rel = safe_name(member.name); target = staging.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f: shutil.copyfileobj(tf.extractfile(member), f)
                target.chmod(0o600)
        for case in m["case_ids"]:
            c = staging / "cases" / case
            for folder in ["work", "outputs", "status", "logs"]: (c / folder).mkdir(exist_ok=True)
        for folder in ["audit", "status", "reports", "logs", "services", "tmp"]: (staging / folder).mkdir(exist_ok=True)
        # Both data and runtime layers share an input-only SHA contract. Narrowing
        # the dispatch catalog changes no original per-case scientific settings.
        (staging / "configs").mkdir(exist_ok=True)
        original_config = REPO / "experiments/paper_runs" / m["study"] / "original_config"
        if original_config.exists():
            for p in original_config.iterdir():
                destination = staging / "configs" / p.name
                if not destination.exists(): shutil.copyfile(p, destination)
        for filename in ["cases.json", "semantic_cases.json"]:
            p = staging / "configs" / filename
            if p.exists(): p.rename(p.with_suffix(".original.json"))
        write(staging / "configs/cases.json", m["case_ids"])
        if m["study"] == "semantic_historical": write(staging / "configs/semantic_cases.json", m["case_ids"])
        model_manifest = staging / "configs/model-files-9B.sha256"
        if not model_manifest.exists(): shutil.copyfile(REPO / "provenance/model-files-9B.sha256", model_manifest)
        files = {x["path"]: x["sha256"] for x in m["files"] if not x["path"].endswith(("/cases.json", "/semantic_cases.json"))}
        for p in (staging / "configs").glob("*"):
            if p.is_file(): files[str(p.relative_to(staging))] = digest(p.read_bytes())
        write(staging / "input_export.json", {"study": m["study"], "files": files,
            "kind": "EXACT_FROZEN_INPUT_REPLAY", "source_manifest_sha256": digest((Path(bundle_dir) / "input_manifest.json").read_bytes()),
            "case_catalog_narrowed_to_paper_scope": True, "fits": 0, "model_calls": 0})
        write(staging / "audit/reproduction_input_binding.json", {
            "study": m["study"], "case_ids": m["case_ids"],
            "input_manifest_sha256": digest((Path(bundle_dir) / "input_manifest.json").read_bytes()),
            "files_verified": len(m["files"]), "original_input_bytes_unchanged": True,
            "model_calls": 0, "fit_calls": 0, "outputs_imported": False,
            "role_boundary": "method workers must deny private/; only the independent once-only scorer may read evaluation references",
        })
        staging.rename(work)
    except BaseException:
        # Keep partial evidence. Never silently delete potentially useful material.
        raise
    return {"work": str(work), "study": m["study"], "cases": len(m["case_ids"]),
            "files_verified": len(m["files"]), "original_input_bytes_unchanged": True,
            "model_calls": 0, "fit_calls": 0}

def download_uci(dataset, out, roles=ROLES):
    """Official raw archive, hash-pinned; never substitute a newer archive."""
    if dataset not in UCI: raise ValueError("Dataset has no pinned official source")
    expected = load(roles)["datasets"][dataset]["raw_zip_sha256"]
    out = Path(out).resolve()
    if out.exists():
        if digest(out.read_bytes()) != expected: raise ValueError("Existing raw source differs; preserve and inspect")
        return {"dataset":dataset,"raw_source":str(out),"sha256":expected,"reused_verified":True}
    ident,slug = UCI[dataset]
    url = f"https://archive.ics.uci.edu/static/public/{ident}/{slug}.zip"
    out.parent.mkdir(parents=True,exist_ok=True)
    partial = out.with_name(out.name + ".download")
    if partial.exists(): raise FileExistsError("Prior partial download preserved: " + str(partial))
    with urllib.request.urlopen(url,timeout=120) as src, partial.open("xb") as dst:
        shutil.copyfileobj(src,dst)
    actual=digest(partial.read_bytes())
    if actual!=expected: raise ValueError("Official raw bytes changed; retain .download for review, do not accept replacement")
    partial.rename(out)
    return {"dataset":dataset,"raw_source":str(out),"sha256":actual,"url":url,"formal_reference_read":False}

def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export-server"); e.add_argument("--catalog", type=Path, default=CATALOG)
    e.add_argument("--study", required=True); e.add_argument("--source-root", type=Path)
    e.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify"); v.add_argument("--bundle-dir", type=Path, required=True)
    v.add_argument("--manifest-sha256")
    a = sub.add_parser("prepare"); a.add_argument("--bundle-dir", type=Path, required=True)
    a.add_argument("--work", type=Path, required=True); a.add_argument("--manifest-sha256")
    raw = sub.add_parser("download-uci"); raw.add_argument("--dataset",choices=sorted(UCI),required=True)
    raw.add_argument("--out",type=Path,required=True);raw.add_argument("--roles",type=Path,default=ROLES)
    x = p.parse_args()
    if x.command == "export-server": result = export_server(x.catalog, x.study, x.output, x.source_root)
    elif x.command == "verify":
        m = inspect_bundle(x.bundle_dir, x.manifest_sha256)
        result = {"passed": True, "study": m["study"], "files_verified": len(m["files"]), "model_calls": 0}
    elif x.command == "download-uci": result = download_uci(x.dataset,x.out,x.roles)
    else: result = prepare(x.bundle_dir, x.work, x.manifest_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
