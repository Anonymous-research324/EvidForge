"""Reject nonpublic benchmark reference reads before opening a file."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
def install():
 if getattr(sys,'_evidforge_reference_guard',False):return
 sys._evidforge_reference_guard=True
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if ROOT not in p.parents:return
  if 'private' in p.parts or 'local_private' in p.parts or p.name=='holdout_dirty.csv':
   raise PermissionError('Only the independent scorer/generator may read benchmark references or holdout')
  if 'downstream_training' in p.parts and os.environ.get('EVIDFORGE_SUPERVISED_COMPONENT')!='caafe':
   raise PermissionError('Prediction targets are unavailable to cell-repair methods')
 sys.addaudithook(guard)
