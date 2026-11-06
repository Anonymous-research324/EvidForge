"""No feedback until a predeclared matched proposal count; same evaluator."""
import os,json
def install():
 import application.native_protocol as module
 if getattr(module,'_nofeedback_installed',False):return
 module._nofeedback_installed=True
 Original=module.Protocol
 from application.bounded_protocol import state
 from unseen_table.common import load,dump
 class Independent(Original):
  def payload(self):
   request=super().payload();current=state(self.work)
   k=int(os.environ['EVIDFORGE_PROPOSAL_TARGET']);attempts=current.get('attempted_proposals',0)
   valid=[q for q in current['programs'] if q.get('eligible_for_research_compile')]
   schema=request['structured_outputs']['json'];branches=schema.get('anyOf',[schema])
   if attempts<k:
    proposed=next(q for q in branches if q['properties']['action']['enum']==['propose'])
    request['structured_outputs']['json']=proposed
   else:
    compile_=[q for q in branches if q['properties']['action']['enum']==['compile']]
    request['structured_outputs']['json']=compile_[0] if compile_ and len(current['programs'])>=2 else {'type':'object','properties':{'action':{'type':'string','enum':['defer']}},'required':['action'],'additionalProperties':False}
   payload=json.loads(request['messages'][1]['content'])
   own=[]
   for i in range(1,load(self.root/'budget.json')['calls']+1):
    p=self.root/f'actual9b-decision-{i}.json'
    if p.exists():
     a=load(p)
     if a.get('action')=='propose':own.append(a)
   payload['previous_answer']={'own_proposed_configurations':own,'proposal_target':k,'attempted_proposals':attempts,'evaluation_feedback_hidden_until_proposals_end':True}
   request['messages'][1]['content']=json.dumps(payload,separators=(',',':'))
   request['messages'][0]['content']+='\nABLATION: Propose independent configurations without access to their evaluation results. Your previous proposals are shown solely to avoid duplicates. You may not revise based on hidden evaluation. At the closing action, compile the permitted ID or defer; no new proposal after the fixed count.'
   return request
  def process(self,n,answer):
   current=state(self.work);k=int(os.environ['EVIDFORGE_PROPOSAL_TARGET'])
   valid=[q for q in current['programs'] if q.get('eligible_for_research_compile')]
   if answer=={'action':'defer'} and current.get('attempted_proposals',0)>=k and not (valid and len(current['programs'])>=2):
    budget=load(self.root/'budget.json')
    if budget['pending_call']!=n or answer!=load(self.root/f'actual9b-decision-{n}.json'):raise ValueError('Original recorded decision required')
    current.update(closed=True,deferred=True);result={'deferred':True,'fixed_independent_proposals_exhausted':True,'proposal_target':k}
    dump(self.work/'hypothesis_state.json',current)
    with (self.work/'hypothesis_trace.jsonl').open('a') as f:f.write(json.dumps({'action':answer,'result':result,'feedback_for_rewrite':False})+'\n')
    budget['last_tool_result']=result;budget['pending_call']=None;dump(self.root/'budget.json',budget)
    return result
   return super().process(n,answer)
 module.Protocol=Independent
