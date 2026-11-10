"""Run the existing native learning component on the authorized A100 host.

Stages only internal authorized teaching evidence and code. Archives every
request/response before copying receipts back; no test input/reference is sent.
"""
from pathlib import Path
import argparse,sys,subprocess,shlex,json,tarfile,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import fresh,load,dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--name',required=True)
    p.add_argument('--tag',required=True);p.add_argument('--remote-helper',required=True);a=p.parse_args()
    if not a.name.isalnum() or not a.tag.isalnum():raise ValueError('Alphanumeric unique name/tag required')
    stage=ROOT/'work'/(a.name+'_native_stage')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/stage_native.py'),'--prepared',a.skill,'--out',str(stage)],check=True)
    archive=stage.with_suffix('.tar');remote='/root/fsas/home/anonymous/experiments/dataforge_v23_'+a.name+'_20261013'
    uploaded='/tmp/dataforge-v23-'+a.name+'-20261013.tar'
    subprocess.run(['scp',str(archive),'a100:'+uploaded],check=True)
    script='''from pathlib import Path
import tarfile,subprocess,json,hashlib
root=Path(ROOT_VALUE);root.mkdir(exist_ok=False)
with tarfile.open(ARCHIVE_VALUE) as t:t.extractall(root,filter='data')
cmd=[PYTHON_VALUE,'-B',str(root/'application/native_learning.py'),'--work',str(root/'work'),'--out',str(root/'native'),'--codex','/root/fsas/home/anonymous/bin/codex','--remote-helper',HELPER_VALUE,'--tag',TAG_VALUE]
proc=subprocess.run(cmd)
files={str(q.relative_to(root)):hashlib.sha256(q.read_bytes()).hexdigest() for folder in ['work','native'] for q in (root/folder).rglob('*') if q.is_file()}
(root/'outputs_freeze.json').write_text(json.dumps({'files':files,'native_returncode':proc.returncode},indent=2))
with tarfile.open(root/'receipt.tar','w') as t:
 for name in [*files,'outputs_freeze.json']:t.add(root/name,arcname=name)
raise SystemExit(proc.returncode)
'''
    python='/root/fsas/home/anonymous/envs/dataforge-run/bin/python'
    for key,value in {'ROOT_VALUE':remote,'ARCHIVE_VALUE':uploaded,'PYTHON_VALUE':python,'HELPER_VALUE':a.remote_helper,'TAG_VALUE':a.tag}.items():script=script.replace(key,repr(value))
    receipt=ROOT/'research'/(a.name+'_native_receipts');receipt.mkdir(exist_ok=False)
    dump(receipt/'local_dispatch.json',{'remote_root':remote,'stage_sha256':sha256(archive),'formal_gold_staged':False})
    with (receipt/'online.log').open('x') as f:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','a100',shlex.join([python,'-B','-'])],input=script,text=True,stdout=f,stderr=subprocess.STDOUT)
    subprocess.run(['scp','a100:'+remote+'/receipt.tar',str(ROOT/'work'/(a.name+'_native_receipt.tar'))],check=True)
    with tarfile.open(ROOT/'work'/(a.name+'_native_receipt.tar')) as t:t.extractall(receipt,filter='data')
    fixed=load(receipt/'outputs_freeze.json')
    if any(sha256(receipt/n)!=h for n,h in fixed['files'].items()):raise ValueError('Native receipt hash mismatch')
    dump(receipt/'local_archive_audit.json',{'checked_files':len(fixed['files']),'all_SHA_passed':True,'remote_exit_code':proc.returncode})
    if proc.returncode:raise RuntimeError('Native failed; full receipts retained')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/seal_skill.py'),'--skill',a.skill,'--native',str(receipt/'native')],check=True)
    print('REAL_NATIVE_AND_RECEIPTS_COMPLETE',a.name,load(receipt/'native/result.json')['real9B_calls'],flush=True)


if __name__=='__main__':main()
