"""Read-only check of packaged files and archived method integrity."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'delivery_manifest.json').read_text());bad=[]
    for n,h in manifest['files'].items():
        p=ROOT/n
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(n)
    result={'checked_files':len(manifest['files']),'mismatches':bad,'passed':not bad,'read_only':True}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return int(bool(bad))


if __name__=='__main__':raise SystemExit(main())
