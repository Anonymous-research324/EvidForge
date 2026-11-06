"""Official HoloClean with source-only mined FDs, same Raha detections and owned PostgreSQL."""
from pathlib import Path
import argparse,os,sys,time,subprocess,random
from common import ROOT,SEED,load,save,read_csv,write_csv,freeze
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
os.environ['PGHOST']='127.0.0.1';os.environ['PGPORT']='55421'
if not hasattr(time,'clock'):time.clock=time.perf_counter
import numpy as np,pandas as pd,torch
torch.set_num_threads(4)
for k,v in {'int':int,'float':float,'bool':bool}.items():
 if k not in np.__dict__:setattr(np,k,v)
sys.path.insert(0,str(ROOT/'sources/holoclean'))
from dataset.dbengine import DBengine
original_db_init=DBengine.__init__
def owned_db_init(self,user,pwd,db,host='localhost',port=5432,**kw):
 return original_db_init(self,user,pwd,db,host='127.0.0.1',port=55421,**kw)
DBengine.__init__=owned_db_init
from holoclean import HoloClean
from detect.detector import Detector
from detect.nulldetector import NullDetector
from repair.featurize import InitAttrFeaturizer,FreqFeaturizer,OccurAttrFeaturizer,ConstraintFeaturizer
from utils import NULL_REPR
def access_guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=str(Path(os.fsdecode(args[0])).resolve())
  if '/datasets/' in p and '/private/' in p or '/local_private/' in p:raise PermissionError('Formal reference unavailable')
  if p.endswith(('development_bundle.json','validation_only.json')):raise PermissionError('Only first10 teaching may train')
sys.addaudithook(access_guard)
class RahaDetector(Detector):
 def __init__(self,cells,cols):super().__init__('same-source-Raha');self.cells=cells;self.cols=cols
 def setup(self,dataset,env):self.ds=dataset
 def detect_noisy_cells(self):return pd.DataFrame([{'_tid_':int(i),'attribute':self.cols[int(j)]} for i,j in self.cells],columns=['_tid_','attribute'])
def main(name):
 start=time.monotonic();pub=ROOT/'datasets'/name/'public';out=ROOT/'outputs'/name/'raha_holoclean';out.mkdir(parents=True,exist_ok=False)
 cols=load(pub/'declaration.json')['targets'];groups=load(pub/'learning_only.json')['learning']
 teach=[{**r,'_source_group':g['group']} for g in groups for r in g['records']]
 fields,val=read_csv(pub/'validation_dirty.csv');_,formal=read_csv(pub/'repair_dirty.csv')
 n=len(teach);nv=len(val)
 mask=load(ROOT/'outputs'/name/'raha_detection/detections.json')
 assert mask['first_learning_rows']==n and mask['validation_rows']==nv
 cells=[(int(i),int(j)) for i,j in mask['cells'] if int(i)>=n]
 write_csv(out/'authorized_input.csv',cols,[{c:r['truth'][c] for c in cols} for r in teach]+[{c:r[c] for c in cols} for r in val+formal])
 rules=[];support=[]
 for a in cols:
  for b in cols:
   if a==b:continue
   buckets={}
   for r in teach:
    v,w=r['truth'][a],r['truth'][b]
    if v and w:buckets.setdefault(v,[]).append((w,r['_source_group']))
   if len(buckets)<2 or any(len({w for w,g in values})!=1 for values in buckets.values()):continue
   repeated=[values for values in buckets.values() if len({g for w,g in values})>=3]
   if len(repeated)<2:continue
   rules.append(f't1&t2&EQ(t1.{a},t2.{a})&IQ(t1.{b},t2.{b})')
   support.append({'from':a,'to':b,'distinct_values':len(buckets),'independent_groups':len({r['_source_group'] for r in teach}),'repeated_values_at_least3_groups':len(repeated)})
 save(out/'source_rules_freeze.json',{'rules':rules,'support':support,'selection_uses_learning_only':True,'test_gold_read':False})
 db='df1014_'+name+'_'+str(os.getpid())
 subprocess.run([str(ROOT/'environments/postgres/bin/createdb'),'-h','127.0.0.1','-p','55421','-U','dataforge1014',db],check=True)
 session=HoloClean(db_user='dataforge1014',db_pwd='',db_host='127.0.0.1',db_name=db,threads=1,seed=SEED,epochs=20,batch_size=1,verbose=False).session
 session.load_data('targets',str(out/'authorized_input.csv'),na_values=[''])
 if rules:
  (out/'source_constraints.txt').write_text('\n'.join(rules)+'\n')
  session.load_dcs(str(out/'source_constraints.txt'));session.ds.set_constraints(session.get_dcs())
 else:session.ds.set_constraints([])
 session.detect_errors([RahaDetector(cells,cols),NullDetector()])
 session.setup_domain()
 featurizers=[InitAttrFeaturizer(),FreqFeaturizer(),OccurAttrFeaturizer()]
 if rules:featurizers.append(ConstraintFeaturizer())
 session.repair_errors(featurizers)
 result=session.ds.repaired_data.df.sort_values('_tid_').reset_index(drop=True)
 initial=session.ds.raw_data.df.sort_values('_tid_').reset_index(drop=True)
 assert len(result)==n+nv+len(formal)
 for zone,rows,offset in [('validation',val,n),('repair',formal,n+nv)]:
  final=[]
  for i,r in enumerate(rows):
   x=dict(r)
   for c in cols:
    v=result.iloc[offset+i][c];base=initial.iloc[offset+i][c]
    if str(v)!=str(base):x[c]='' if pd.isna(v) or str(v)==NULL_REPR else str(v)
   final.append(x)
  write_csv(out/(zone+'_output.csv'),fields,final)
 save(out/'receipt.json',{'method':'Raha detection→official HoloClean','seed':SEED,'source_groups':len(groups),'source_rows':n,
 'same_raha_mask':True,'native_null_detector_added':True,'source_only_FDs':len(rules),'official_default_numeric_hyperparameters':True,
 'postgres_owned_port':55421,'compatibility':['Python time.clock alias','numpy type aliases','owned DB port','serial native featurization avoids Torch IPC FD exhaustion'],
 'lexical_normalization_only_is_not_counted_as_repair':True,'test_gold_read':False,'elapsed_seconds':time.monotonic()-start})
 save(out/'output_freeze.json',{'files':freeze(out),'formal_reference_read':False})
 print('HOLOCLEAN_OUTPUT_CLOSED',name,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
