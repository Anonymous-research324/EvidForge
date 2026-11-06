"""Apply the preserved prior, then a separately source-qualified context rule."""
from pathlib import Path
from learning.environment import RUNTIME
from learning.execute import run as prior_run
from learning.candidate_verifier import rank,permitted
from unseen_table.common import load,sha256
from .context import predict
from .frames import enrich


def run(rows,path):
    root=Path(path);configuration=load(root/'configuration.json');manifest=load(root/'manifest.json')
    if any(sha256(root/n)!=h for n,h in manifest['files'].items()):raise ValueError('Sealed context skill changed')
    source=load(root/'source_freeze.json');prior=Path(configuration['prior_skill_path'])
    if sha256(prior/'manifest.json')!=source['prior_manifest_sha256']:raise ValueError('Prior binding changed')
    if sha256(root/'context_model.json')!=source['context_model_sha256']:raise ValueError('Context model binding changed')
    previous,prior_frames,cells=prior_run(rows,prior,return_evidence=True);candidate=root/'compiled_candidate.json'
    if not candidate.exists():return [{**d,'prior_after':d['after'],'context_rule_used':False} for d in previous]
    if len(prior_frames)!=len(previous):raise ValueError('Prior evidence order/length differs')
    frames=[]
    for f,d in zip(prior_frames,previous):
        if (f['row'],f['column'])!=(d['row'],d['field']):raise ValueError('Prior evidence coordinates differ')
        frames.append({**f,'baseline_after_local':d['after']})
    frames=enrich(frames,predict(rows,load(root/'context_model.json')))
    c=load(candidate)
    if c['source_freeze_sha256']!=sha256(root/'source_freeze.json'):raise ValueError('Context program source binding changed')
    ranked=rank(frames,c['program']);output=[]
    for d,q in zip(previous,ranked):
        allowed=permitted(q,c['policy']);output.append({**d,'prior_after':d['after'],
            'after':q['value'] if allowed else d['after'],'context_rule_used':allowed})
    return output
