"""Actual9B transport: stdin JSON to the existing authorized A100 helper."""
import sys,os,subprocess,shlex
if len(sys.argv)!=2 or sys.argv[1]!='call':raise SystemExit('Only call is available; this bridge cannot start/stop any service')
host=os.environ.get('TABLE_REPAIR_9B_HOST','a100')
helper=os.environ.get('TABLE_REPAIR_9B_HELPER','/root/fsas/home/anonymous/experiments/dataforge_v8_9b_v11Learning20261012/remote_probe.py')
python=os.environ.get('TABLE_REPAIR_9B_PYTHON','/root/fsas/home/anonymous/envs/dataforge-infer/bin/python')
command=shlex.join([python,'-B',helper,'call'])
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],
    input=sys.stdin.read(),text=True,stdout=sys.stdout,stderr=sys.stderr,timeout=330)
raise SystemExit(result.returncode)
