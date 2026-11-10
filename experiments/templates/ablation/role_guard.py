from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('@DOWNSTREAM_ROOT@')
def install():
 if getattr(sys,'_evidforge_ablation_guard',False):return
 sys._evidforge_ablation_guard=True
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if ROOT not in p.parents and OLD not in p.parents:return
  if 'private' in p.parts or 'local_private' in p.parts or 'downstream_training' in p.parts or p.name=='holdout_dirty.csv':
   raise PermissionError('Ablation repair/learning cannot read formal references, predictive labels or holdout')
 sys.addaudithook(guard)
