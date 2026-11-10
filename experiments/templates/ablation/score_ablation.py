"""Independent once-only scoring after all new outputs and learning artifacts freeze."""
from pathlib import Path
import sys,time
from common import ROOT,OLD,load,save,read_csv,digest
from prediction import frame,evaluate_frames
ENGINE=ROOT/'frozen_workspace/DataForge_UnseenTable_v23'
sys.path.insert(0,str(ENGINE/'source_runtime'))
from unseen_table.common import equal
NAMES=['real_estate','concrete','airfoil','tic_tac_toe','car_evaluation','retinopathy']
ARMS=['A','B','D']
def metrics(dirty,output,reference,cols):
 changes=exact=errors=normal=damage=0
 for a,b,t in zip(dirty,output,reference):
  for c in cols:
   if t[c]=='':continue
   bad=not equal(a[c],t[c]);changed=not equal(a[c],b[c])
   errors+=bad;normal+=not bad;changes+=changed
   exact+=bool(changed and bad and equal(b[c],t[c]));damage+=bool(changed and not bad)
 p=exact/changes if changes else None;r=exact/errors if errors else 0
 return {'changes':changes,'exact':exact,'errors':errors,'normal':normal,'normal_damage':damage,'P':p,'R':r,'F1':2*p*r/(p+r) if p and r else 0,'damage':damage/normal if normal else None}
def main():
 assert (ROOT/'status/all_outputs_closed.json').exists()
 assert not (ROOT/'audit/scoring_started.json').exists()
 for n in NAMES:
  for a in ['A_saved',*ARMS]:
   folder=ROOT/'outputs'/n/a
   for rel,h in load(folder/'output_freeze.json')['files'].items():assert digest(folder/rel)==h
 save(ROOT/'audit/scoring_started.json',{'time':time.time(),'all42_outputs_closed':True,'expected_outputs':len(NAMES)*(len(ARMS)+1),'only_reader':'score_ablation.py','old_study_not_regraded':True})
 results=[]
 for n in NAMES:
  pub=OLD/'datasets'/n/'public';meta=load(pub/'dataset_metadata.json');cols=meta['targets']
  fields,dirty=read_csv(pub/'repair_dirty.csv');_,truth=read_csv(OLD/'datasets'/n/'private/repair_reference.csv');_,held=read_csv(OLD/'datasets'/n/'private/holdout_reference.csv')
  assert [r['entity_id'] for r in dirty]==[r['entity_id'] for r in truth]
  y=[r['class'] for r in truth];yt=[r['class'] for r in held];test=frame(held,cols)
  for a in ARMS:
   _,rows=read_csv(ROOT/'outputs'/n/a/'repair_output.csv')
   assert [r['entity_id'] for r in rows]==[r['entity_id'] for r in dirty]
   item={'dataset':n,'arm':a,'task':meta['task'],'repair':metrics(dirty,rows,truth,cols),
    'downstream':evaluate_frames(frame(rows,cols),y,test,yt,meta['task'],meta['categorical_columns'])}
   results.append(item);save(ROOT/'reports/scores.json',results);print('ABLATION_SCORED',n,a,flush=True)
 from report import build
 build(results)
 save(ROOT/'status/scoring_complete.json',{'rows':len(results),'time':time.time(),'single_seed':1014})
if __name__=='__main__':main()
