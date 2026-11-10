"""Bind a closed native decision to the local source-fitted skill package."""
from pathlib import Path
import argparse,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--native',required=True);a=p.parse_args()
    root=Path(a.skill);native=Path(a.native);result=load(native/'result.json')
    if not result['closed'] or result['native_exit_code'] or result['gateway_failure']:raise ValueError('Native rule learning has not closed')
    if (root/'manifest.json').exists():raise FileExistsError('Preserve sealed skill')
    if result['candidate']:
        if result['candidate']['source_freeze_sha256']!=sha256(root/'source_freeze.json'):raise ValueError('Native source binding differs')
        dump(root/'compiled_candidate.json',result['candidate'])
    dump(root/'native_learning_receipt.json',result)
    (root/'SKILL.md').write_text('''# Source-learned table repair skill

Match confirmed source, field names and protected ID/class roles. Learning used
the supplied corrected teaching partition with source-separated checks; original rows
are backend only. Preserve every original candidate including KEEP and unknown
outcomes. Run learned basic inverses, independently review weak numeric base
edits, and execute a model-generated scoring configuration only through its
source-qualified permission. Never write an estimated continuous mean.

Tool: apply_verified_skill(batch). This is a source-checked research candidate,
pending independent output-frozen validation. The learning score is not a formal
result or universal quality guarantee. Historical skills/experiments are read-only.
''')
    files={str(q.relative_to(root)):sha256(q) for q in root.rglob('*') if q.is_file()}
    dump(root/'manifest.json',{'files':files,'format_version':load(root/'configuration.json').get('format_version',14),'status':'pending_independent_validation'})
    print('SOURCE_SKILL_SEALED',len(files),flush=True)


if __name__=='__main__':main()
