"""Trusted process-local switches; original core and sealed packages unchanged."""
import argparse,sys,os,time,collections
from pathlib import Path
from common import ROOT,OLD,load,save,read_csv,write_csv,freeze
def main(name,arm,skill,out):
 os.environ['EVIDFORGE_ABLATION_ARM']=arm
 import seed_runtime;seed_runtime.install()
 from workflow.packages import execute
 counters=collections.Counter()
 if arm=='B':
  import unseen_table.candidate_pool as pool,unseen_table.rank_candidates as ranks
  original=pool.candidates;remove={'affix','scale','shift','mapping','strip','casefold','edit'}
  def candidates(row,col,fitted,context=None,source=None):
   rules=[x for x in fitted['rules'] if x['kind'] not in remove]
   counters['filtered_rule_applications']+=sum(col in x['fields'] for x in fitted['rules'] if x['kind'] in remove)
   return original(row,col,{**fitted,'rules':rules},context,source)
  pool.candidates=candidates;ranks.candidates=candidates
 if arm=='D':
  import unseen_table.validate_skill as base,learning.candidate_verifier as generated
  import skill_runtime.execute,learning.execute,supplementary.execute
  ob,og=base.permitted,generated.permitted
  def no_threshold(c,policy):
   counters['base_gate_checks']+=1
   if 'confidence_threshold' in policy:allowed=c['top']['kind']!='keep' and not c.get('tied',False)
   else:
    p=policy.get('by_region',{}).get(base.region(c),policy if 'score_threshold' in policy else {})
    allowed=c['top']['kind']!='keep' and not c.get('tied',False) and bool(p)
   counters['base_newly_admitted']+=bool(allowed and not ob(c,policy))
   return allowed
  def no_generated_threshold(q,policy):
   counters['generated_gate_checks']+=1
   allowed=bool(q['accept'] and q['differs_from_baseline'] and policy.get(q['channel']))
   counters['generated_newly_admitted']+=bool(allowed and not og(q,policy))
   return allowed
  for m in list(sys.modules.values()):
   if m is None:continue
   if vars(m).get('permitted') is ob:setattr(m,'permitted',no_threshold)
   elif vars(m).get('permitted') is og:setattr(m,'permitted',no_generated_threshold)
 fields,rows=read_csv(OLD/'datasets'/name/'public/repair_dirty.csv')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 output,decisions=execute(rows,skill)
 write_csv(out/'repair_output.csv',fields,output);save(out/'decisions.json',decisions)
 parity=None
 if arm=='A_saved':
  parity=(out/'repair_output.csv').read_bytes()==(OLD/'outputs'/name/'primary_llm/v23_output.csv').read_bytes()
  if not parity:raise ValueError('Unmodified full execution does not reproduce old output')
 save(out/'receipt.json',{'arm':arm,'dataset':name,'skill':str(skill),'elapsed_seconds':time.monotonic()-start,'switch_counters':dict(counters),'old_full_byte_parity':parity,'formal_reference_read':False,'seed':1014})
 save(out/'output_freeze.json',{'files':freeze(out),'formal_reference_read':False})
 print('ARM_CLOSED',name,arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',required=True);p.add_argument('--skill',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.dataset,a.arm,Path(a.skill),a.out)
