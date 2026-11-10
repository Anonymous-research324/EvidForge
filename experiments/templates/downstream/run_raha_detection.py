"""Raha official detection API with annotation scope restricted to first10%.
No formal clean_dataframe is constructed, and official truth-evaluation calls fail.
"""
from pathlib import Path
import sys,os,json,argparse,random,time,types,multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,save,load,read_csv,write_csv,digest,freeze,SEED
sys.path.insert(0,str(ROOT/'sources/raha'))
import pandas as pd
if not hasattr(pd.DataFrame,"map"):pd.DataFrame.map=pd.DataFrame.applymap
import numpy as np
np.random.seed(SEED);random.seed(SEED)
import raha
OriginalDataset=raha.dataset.Dataset
class TeachingOnlyDataset(OriginalDataset):
 def get_actual_errors_dictionary(self):raise PermissionError('Whole-table ground truth is unavailable')
 def get_data_cleaning_evaluation(self,*a,**k):raise PermissionError('Only independent scorer may read formal truth')
raha.dataset.Dataset=TeachingOnlyDataset
OriginalPool=multiprocessing.Pool
def bounded_pool(*args,**kwargs):
 kwargs.pop('processes',None)
 return OriginalPool(4,*args[1:],**kwargs)
raha.detection.multiprocessing.Pool=bounded_pool
raha.correction.Pool=bounded_pool
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=str(Path(os.fsdecode(args[0])).resolve())
  if '/datasets/' in p and '/private/' in p or '/local_private/' in p:raise PermissionError('Baseline cannot read reference files')
  if p.endswith(('development_bundle.json','validation_only.json')):raise PermissionError('Training labels restricted to learning_only.json')
sys.addaudithook(audit)
class ScopedSampling:
 def sample_tuple(self,d):
  original=np.random.choice
  def choose(a,size=None,replace=True,p=None):
   population=np.arange(a) if isinstance(a,(int,np.integer)) else np.asarray(a)
   weights=np.ones(len(population)) if p is None else np.array(p,dtype=float)
   mask=np.array([int(i)<self.source_count and int(i) not in d.labeled_tuples for i in population])
   weights[~mask]=0
   if weights.sum()<=0:weights=mask.astype(float)
   if weights.sum()<=0:raise RuntimeError('No authorized teaching tuple remains')
   return original(population,size=size,replace=replace,p=weights/weights.sum())
  np.random.choice=choose
  try:return super().sample_tuple(d)
  finally:np.random.choice=original
 def label_with_ground_truth(self,d):
  i=int(d.sampled_tuple)
  if i>=self.source_count:raise PermissionError('Annotation request outside teaching source')
  d.labeled_tuples[i]=1
  for j in range(d.dataframe.shape[1]):
   value=self.source_truth[i][j];d.labeled_cells[(i,j)]=[int(d.dataframe.iloc[i,j]!=value),value]
class ScopedDetection(ScopedSampling,raha.detection.Detection):
 def initialize_dataset(self,dd):
  d=super().initialize_dataset(dd);d.has_ground_truth=True;self.last_dataset=d;return d
class ScopedCorrection(ScopedSampling,raha.correction.Correction):
 def sample_tuple(self,d):
  import math
  remaining_column_erroneous_cells={};remaining_column_erroneous_values={}
  for j in d.column_errors:
   for cell in d.column_errors[j]:
    if cell not in d.corrected_cells:
     self._to_model_adder(remaining_column_erroneous_cells,j,cell)
     self._to_model_adder(remaining_column_erroneous_values,j,d.dataframe.iloc[cell])
  tuple_score=np.ones(d.dataframe.shape[0])
  tuple_score[list(d.labeled_tuples.keys())]=0.
  for j in remaining_column_erroneous_cells:
   for cell in remaining_column_erroneous_cells[j]:
    value=d.dataframe.iloc[cell]
    column_score=math.exp(len(remaining_column_erroneous_cells[j])/len(d.column_errors[j]))
    cell_score=math.exp(remaining_column_erroneous_values[j][value]/len(remaining_column_erroneous_cells[j]))
    tuple_score[cell[0]]*=column_score*cell_score
  allowed=np.array([i<self.source_count and i not in d.labeled_tuples for i in range(len(tuple_score))])
  if not allowed.any():raise RuntimeError('No authorized source remains')
  tuple_score[~allowed]=-np.inf
  d.sampled_tuple=np.random.choice(np.argwhere(tuple_score==np.amax(tuple_score)).flatten())

def main(name):
 start=time.monotonic();pub=ROOT/'datasets'/name/'public'
 declaration=load(pub/'declaration.json');cols=declaration['targets']
 groups=load(pub/'learning_only.json')['learning']
 records=[r for g in groups for r in g['records']]
 fields,val=read_csv(pub/'validation_dirty.csv');_,formal=read_csv(pub/'repair_dirty.csv')
 rows=[r['dirty'] for r in records]+val+formal;n=len(records);nv=len(val)
 out=ROOT/'outputs'/name/'raha_detection'
 if out.exists():raise FileExistsError('Do not overwrite a baseline run')
 out.mkdir(parents=True)
 write_csv(out/'authorized_dirty_input.csv',cols,[{c:r[c] for c in cols} for r in rows])
 truth=[[r['truth'][c] for c in cols] for r in records]
 detector=ScopedDetection();detector.source_count=n;detector.source_truth=truth
 detector.LABELING_BUDGET=n;detector.SAVE_RESULTS=False;detector.VERBOSE=False
 detector.HISTORICAL_DATASETS=[];detector.STRATEGY_FILTERING=False
 detector.ERROR_DETECTION_ALGORITHMS=['OD','PVD','RVD'] # external KB disabled uniformly, not oracle data.
 detector.run({'name':name+'1014','path':str(out/'authorized_dirty_input.csv')})
 d=detector.last_dataset
 save(out/'detections.json',{'cells':[list(c) for c in d.detected_cells],'first_learning_rows':n,'validation_rows':nv,'formal_rows':len(formal)})

 save(out/'receipt.json',{'method':'official Raha source-restricted detection for HoloClean','seed':SEED,'source_rows':n,'source_groups':len(groups),'unique_label_budget_rows':n,'test_gold_read':False,'external_KB_used':False,'elapsed_seconds':time.monotonic()-start,'cpu_workers':4})
 save(out/'output_freeze.json',{'files':freeze(out),'test_reference_read':False})
 print('RAHA_DETECTION_CLOSED',name,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args().dataset)
