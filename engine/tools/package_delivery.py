"""Seal a reviewable ZIP, excluding formal references and duplicate stages."""
from pathlib import Path
import argparse,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]


def excluded(relative):
    parts=relative.parts
    return ('local_private' in parts or '__pycache__' in parts or
            any(p.endswith('_native_stage') for p in parts) or relative.suffix=='.tar' or
            relative.name in ['delivery_manifest.json','delivery_zip_receipt.json'] or relative.suffix=='.zip')


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);a=p.parse_args();archive=Path(a.zip)
    if archive.exists() or (ROOT/'delivery_manifest.json').exists():raise FileExistsError('Preserve sealed delivery')
    status=json.loads((ROOT/'research/frozen_suite_status.json').read_text())
    if status['phase']!='complete':raise ValueError('Fresh suite has not completed')
    validation=json.loads((ROOT/'validation/delivery_readiness.json').read_text())
    if not validation['passed']:raise ValueError('Delivery checks did not pass')
    files={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(ROOT.rglob('*'))
           if q.is_file() and not excluded(q.relative_to(ROOT))}
    manifest={'files':files,'formal_references_included':False,'model_weights_included':False,
        'historical_resources':'same workspace DataForge_SkillPlatform_v2; no claim all6.8GB bundled',
        'fixed_seed':20260927,'frozen_method':'validation/fresh_transfer_algorithm_freeze.json',
        'not_all_sources_qualified':'Read actual results and qualification registry; failures retained',
        'delivery_checks_passed':True}
    (ROOT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*files,'delivery_manifest.json']:z.write(ROOT/name,arcname=ROOT.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        for name,h in files.items():
            if hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()!=h:raise ValueError('Archived file changed')
    receipt={'archive':str(archive.resolve()),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
       'bytes':archive.stat().st_size,'files_checked':len(files),'all_archive_SHA_passed':True,'formal_gold_included':False}
    (ROOT/'delivery_zip_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
