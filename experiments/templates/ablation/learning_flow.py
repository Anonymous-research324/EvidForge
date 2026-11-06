"""Paired native learning from the same source-only starting material."""
from pathlib import Path
import sys,os,subprocess,shutil,time
from common import ROOT,OLD,load,save,freeze,digest
ENGINE=ROOT/'frozen_workspace/DataForge_UnseenTable_v23'
PY='@RUN_PYTHON@'
WRAP=ROOT/'code/v23-python'
def run(args,log,env):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('x') as f:subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def native(skill,name,variant,kind,env,paired=None):
 case=ROOT/'work'/name/variant;stage=case/(kind+'_stage');logs=ROOT/'logs'/name/variant
 run([WRAP,'-B',ENGINE/'tools/stage_native.py','--prepared',skill,'--out',stage],logs/(kind+'_stage.log'),env)
 local={**env,'EVIDFORGE_NATIVE_STAGE':str(stage),'EVIDFORGE_NATIVE_VARIANT':{'A':'full','E':'no_feedback','F':'no_hook'}[variant]}
 if variant=='E':
  state=load(ROOT/'work'/name/'A'/(kind+'_stage/work/hypothesis_state.json'))
  local['EVIDFORGE_PROPOSAL_TARGET']=str(state['attempted_proposals'])
 tag='Ab'+name.replace('_','').title()+variant+kind.title()+'S1014'
 run([PY,'-B',ROOT/'code/run_native_variant.py','--work',stage/'work','--out',stage/'native','--codex','@CODEX@','--remote-helper',ROOT/'code/model_service.py','--tag',tag],logs/(kind+'_native.log'),local)
 receipt=load(stage/'native/result.json')
 assert receipt['closed'] and receipt['native_exit_code']==0
 assert receipt['project_hook_registered']==(variant!='F')
 run([WRAP,'-B',ENGINE/'tools/seal_skill.py','--skill',skill,'--native',stage/'native'],logs/(kind+'_seal.log'),env)
 return receipt
def learn(name,variant,env):
 base=ROOT/'work'/name/variant;logs=ROOT/'logs'/name/variant;base.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=True)
 prior=base/'prior';original=OLD/'outputs/v23'/name/'onboarding/prior_prepared'
 shutil.copytree(original,prior)
 # Keep fixed source fit, remove only previous native outcomes and seal before a fresh session.
 for f in ['manifest.json','SKILL.md','compiled_candidate.json','native_learning_receipt.json','hypothesis_state.json','hypothesis_trace.jsonl']:
  p=prior/f
  if p.exists():p.unlink()
 prior_receipt=native(prior,name,variant,'prior',env)
 context=base/'context'
 run([WRAP,'-B',ENGINE/'repair_platform.py','prepare-context','--prior',prior,'--out',context],logs/'context_prepare.log',env)
 context_receipt=native(context,name,variant,'context',env)
 assessments={}
 for kind,skill in [('prior',prior),('context',context)]:
  wf=base/(kind+'_external');wf.mkdir()
  for f in ['heldout_validation_local.json','learning_only_bundle.json','partition_freeze.json','declaration.json','context.json']:
   shutil.copyfile(OLD/'outputs/v23'/name/'onboarding'/f,wf/f)
  run([WRAP,'-B',ENGINE/'repair_platform.py','freeze-validation','--workflow',wf,'--skill',skill],logs/(kind+'_freeze.log'),env)
  out=base/(kind+'_assessment')
  run([WRAP,'-B',ENGINE/'repair_platform.py','validate','--workflow',wf,'--out',out],logs/(kind+'_assessment.log'),env)
  assessments[kind]=load(out/'assessment.json')
 passed=[k for k,v in assessments.items() if v['passed']]
 chosen=max(passed,key=lambda k:(assessments[k]['metrics']['F1'],k=='prior')) if passed else 'prior'
 selected=base/chosen
 save(base/'selection.json',{'variant':variant,'skill':str(selected),'kind':chosen,'passed':bool(passed),'candidate_assessments':assessments,'prior_native':prior_receipt,'context_native':context_receipt,'seed':1014,'formal_reference_read':False,'no_refit_after_external':True})
 save(base/'learning_freeze.json',{'files':freeze(base),'formal_reference_read':False})
 return selected
