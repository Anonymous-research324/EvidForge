"""Prepare→real native9B→seal two frozen candidates→held validation; no test truth."""
from pathlib import Path
import argparse,sys,json,subprocess,time,shutil,os
from common import ROOT,save,load,status,digest
ENGINE=ROOT/'frozen_workspace/DataForge_UnseenTable_v23'
PY=ROOT/'code/v23-python'
SERVICE=ROOT/'code/model_service.py'
def run(args,log):
 with log.open('x') as f:subprocess.run([str(PY),'-B',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def native(skill,name):
 stage=ROOT/'outputs/stages'/name
 run([ENGINE/'tools/stage_native.py','--prepared',skill,'--out',stage],ROOT/'logs'/(name+'_stage.log'))
 run([stage/'application/native_learning.py','--work',stage/'work','--out',stage/'native',
     '--codex','@CODEX@','--remote-helper',SERVICE,'--tag',name],
     ROOT/'logs'/(name+'_native.log'))
 run([ENGINE/'tools/seal_skill.py','--skill',skill,'--native',stage/'native'],ROOT/'logs'/(name+'_seal.log'))
def main(name):
 public=ROOT/'datasets'/name/'public';workflow=ROOT/'outputs/v23'/name/'onboarding'
 status('v23_source_learning',dataset=name)
 run([ENGINE/'repair_platform.py','prepare-heldout','--labels',public/'development_bundle.json',
      '--declaration',public/'declaration.json','--context',public/'context.json','--out',workflow],
     ROOT/'logs'/(name+'_v23_prepare.log'))
 prior=workflow/'prior_prepared';tag=name.replace('_','').title()+'Prior1014'
 native(prior,tag)
 context=ROOT/'outputs/v23'/name/'context_prepared'
 run([ENGINE/'repair_platform.py','prepare-context','--prior',prior,'--out',context],ROOT/'logs'/(name+'_v23_context_prepare.log'))
 native(context,name.replace('_','').title()+'Context1014')
 workflows={'context':workflow,'prior':ROOT/'outputs/v23'/name/'prior_validation'};workflows['prior'].mkdir()
 for n in ['heldout_validation_local.json','learning_only_bundle.json','partition_freeze.json','declaration.json','context.json']:
  shutil.copyfile(workflow/n,workflows['prior']/n)
 skills={'prior':prior,'context':context}
 for kind,skill in skills.items():
  run([ENGINE/'repair_platform.py','freeze-validation','--workflow',workflows[kind],'--skill',skill],ROOT/'logs'/(name+'_'+kind+'_freeze.log'))
 results={}
 for kind,skill in skills.items():
  out=ROOT/'outputs/v23'/name/(kind+'_assessment')
  run([ENGINE/'repair_platform.py','validate','--workflow',workflows[kind],'--out',out],ROOT/'logs'/(name+'_'+kind+'_assess.log'))
  results[kind]=load(out/'assessment.json')
 passed=[k for k in results if results[k]['passed']]
 chosen=max(passed,key=lambda k:(results[k]['metrics']['F1'],k=='prior')) if passed else 'prior'
 save(ROOT/'outputs/v23'/name/'selection.json',{'kind':chosen,'skill':str(skills[chosen]),'passed':bool(passed),
   'candidate_assessments':results,'policy':'best held-validation F1 among passed; tie prior; no pass research-only prior',
   'formal_reference_read':False,'refit_after_validation':False,'seed':1014})
 status('v23_source_ready',dataset=name,selected=chosen,qualified=bool(passed))
 print('V23_SOURCE_READY',name,'seed1014',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();main(a.dataset)
