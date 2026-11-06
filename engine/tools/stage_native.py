"""Stage code and internal teaching evidence; evaluation references excluded."""
from pathlib import Path
import argparse,sys,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,fresh,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    stage=fresh(a.out);source=Path(a.prepared);manifest=load(source/'prepared_manifest.json')
    if any(sha256(source/n)!=h for n,h in manifest['files'].items()):raise ValueError('Prepared source changed')
    for name in ['application','learning','source_runtime']:
        shutil.copytree(ROOT/name,stage/name,ignore=shutil.ignore_patterns('__pycache__'))
    work=fresh(stage/'work')
    for n in [*manifest['files'],'prepared_manifest.json']:shutil.copyfile(source/n,work/n)
    files={str(q.relative_to(stage)):sha256(q) for q in stage.rglob('*') if q.is_file()}
    if any('held_local_only' in n or 'local_private' in n for n in files):raise ValueError('Reference material cannot enter stage')
    dump(stage/'stage_manifest.json',{'files':files,'scope':'source teaching backend only','formal_gold_included':False,
         'original_reference_rows_visible_to_model':False})
    archive=stage.with_suffix('.tar')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w') as t:
        for q in sorted(stage.rglob('*')):
            if q.is_file():t.add(q,arcname=str(q.relative_to(stage)))
    print(str(archive),flush=True)


if __name__=='__main__':main()
