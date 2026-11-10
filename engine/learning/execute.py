"""Run a learned package from dirty observations only; no query reference API."""
from pathlib import Path
from .environment import RUNTIME
from .evidence import build
from .review_selection import execute as review
from .candidate_verifier import rank,permitted
from .prepare import affix_apply
from unseen_table.common import load,sha256
from skill_runtime.package import open_skill
from unseen_table.rank_candidates import predict_rows
from skill_runtime.execute import propose_from_cells


def run(rows,path,*,return_evidence=False):
    root=Path(path);configuration=load(root/'configuration.json')
    manifest=load(root/'manifest.json')
    if any(sha256(root/n)!=h for n,h in manifest['files'].items()):raise ValueError('Sealed source package changed')
    source=load(root/'source_freeze.json');basepath=Path(configuration['base_skill_path'])
    if sha256(basepath/'manifest.json')!=source['base_manifest_sha256']:raise ValueError('Base binding changed')
    _,base,codec,family,policy=open_skill(basepath)
    if any(set(row)!=set(base['declaration']['fields']) for row in rows):raise ValueError('Input field roles/schema differ')
    records=load(root/'source_records_backend.json')['records']
    cells=predict_rows(rows,base);original=propose_from_cells(rows,cells,base,codec,family,policy)
    enriched=affix_apply(rows,cells,original,base,codec,configuration['exact_affix_enabled'])
    fs=build(rows,records,base,cells,enriched)
    safe=review(fs,configuration['base_review_policy'])
    candidate=root/'compiled_candidate.json';result=[]
    if candidate.exists():
        c=load(candidate)
        if c['source_freeze_sha256']!=sha256(root/'source_freeze.json'):raise ValueError('Generated rule source changed')
        ranked=rank(fs,c['program'])
    else:ranked=[]
    lookup={(q['frame']['row'],q['frame']['column']):q for q in ranked}
    for f,s,d in zip(fs,safe,enriched):
        q=lookup.get((f['row'],f['column']));after=s['baseline_after_local'];why='source_reviewed_baseline'
        if q and permitted(q,c['policy']):after=q['value'];why='source_checked_generated_configuration'
        result.append({**d,'after':after,'unreviewed_base':original[len(result)]['after'],
                       'v15_reason':why,'no_reference_value_used':True})
    return (result,fs,cells) if return_evidence else result
