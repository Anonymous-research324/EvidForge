"""Independent once-only scorer; opens formal references only after every arm closes."""
from pathlib import Path
import argparse,json,time,math
from decimal import Decimal,InvalidOperation
from common import ROOT,SEED,load,save,read_csv,digest,freeze
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
METHODS=['pure_llm','tool_llm','raha_holoclean','garf','cdc','v23']
LABELS={'pure_llm':'纯9B','tool_llm':'9B＋代码工具','raha_baran':'Raha＋Baran','raha_holoclean':'Raha＋HoloClean','garf':'GARF（SQLite移植）','cdc':'CDC（源内校验、CPU限时）','v23':'EvidForge'}
def number(v):
 try:
  x=Decimal(str(v).strip())
  return x if x.is_finite() else None
 except InvalidOperation:return None
def equal(a,b):
 x,y=number(a),number(b)
 return x==y if x is not None and y is not None else str(a)==str(b)
def paths(name,method):
 base=ROOT/'outputs'/name
 if method in ['pure_llm','v23']:
  folder=base/'primary_llm';file=folder/('pure_output.csv' if method=='pure_llm' else 'v23_output.csv')
 else:folder=base/method;file=folder/'repair_output.csv'
 return folder,file
def repair_metrics(dirty,output,truth,cols):
 assert len(dirty)==len(output)==len(truth)
 E=N=C=T=FP=0;known=unknown=0;labels=[];pred=[]
 for d,o,t in zip(dirty,output,truth):
  assert d['entity_id']==o['entity_id']==t['entity_id'],'Rows or protected IDs changed'
  for c in cols:
   if t[c]=='':unknown+=1;continue
   wrong=not equal(d[c],t[c]);changed=not equal(d[c],o[c])
   known+=1;E+=wrong;N+=not wrong;C+=changed
   T+=bool(wrong and changed and equal(o[c],t[c]));FP+=bool(not wrong and changed)
   labels.append(int(wrong));pred.append(int(changed))
 P=T/C if C else 0.;R=T/E if E else 0.
 return {'P':P,'R':R,'F1':2*P*R/(P+R) if P+R else 0.,'normal_damage':FP/N if N else 0.,
 'detection_AUC':float(roc_auc_score(labels,pred)) if len(set(labels))==2 else None,
 'correct_repairs':T,'all_changed_known_cells':C,'known_errors':E,'normal_known_cells':N,'normal_cells_changed':FP,
 'known_cells':known,'natural_unknown_cells_excluded':unknown,
 'detection_AUC_scope':'same binary actual-edit indicator for all six methods; single operating point, no confidence-ranking claim'}
def array(rows,cols):
 return np.array([[float(number(r[c])) if number(r[c]) is not None and abs(number(r[c]))<Decimal('1e100') else np.nan for c in cols] for r in rows],dtype=float)
def classification(train,train_reference,holdout,hold_reference,cols):
 assert [r['entity_id'] for r in train]==[r['entity_id'] for r in train_reference]
 assert [r['entity_id'] for r in holdout]==[r['entity_id'] for r in hold_reference]
 y=np.array([r['class'] for r in train_reference]);yt=np.array([r['class'] for r in hold_reference])
 model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
  LogisticRegression(C=1.,max_iter=2000,random_state=SEED))
 model.fit(array(train,cols),y)
 prediction=model.predict(array(holdout,cols));prob=model.predict_proba(array(holdout,cols));classes=model[-1].classes_
 auc_reason=None
 try:
  if len(classes)==2:auc=float(roc_auc_score((yt==classes[-1]).astype(int),prob[:,1]))
  else:auc=float(roc_auc_score(yt,prob,labels=classes,multi_class='ovr',average='macro'))
  if not math.isfinite(auc):auc=None;auc_reason='Undefined AUC for this fixed holdout class distribution; no split changed.'
 except ValueError as error:
  auc=None;auc_reason='Undefined AUC for this fixed holdout: '+str(error)[:200]
 return {'macro_F1':float(f1_score(yt,prediction,average='macro')),'AUC':auc,'AUC_unavailable_reason':auc_reason,'fixed_classifier':'LR C=1 max_iter2000 seed1014',
 'same_preprocessing':'train-only median imputation→train-only standardization','classification_labels_only_in_independent_downstream_scorer':True,
 'holdout_untouched_and_not_given_to_repair':True}
def verify(folder,receipt):
 for rel,h in load(receipt).get('files',{}).items():
  p=folder/rel
  if not p.is_file() or digest(p)!=h:raise ValueError('Frozen method output changed: '+str(p))
def self_test():
 d=[{'entity_id':'a','x':'9','y':'8','z':'3','w':'4','u':''}]
 t=[{'entity_id':'a','x':'1','y':'2','z':'3','w':'4','u':''}]
 o=[{'entity_id':'a','x':'1','y':'7','z':'6','w':'4','u':'5'}]
 x=repair_metrics(d,o,t,['x','y','z','w','u'])
 assert x['P']==1/3 and x['R']==.5 and x['F1']==.4 and x['normal_damage']==.5 and x['natural_unknown_cells_excluded']==1
 assert abs(x['detection_AUC']-.75)<1e-12
 print('SCORER_SYNTHETIC_ROLE_AND_COUNT_TEST_PASS; zero dataset references read')
def main(close_incomplete=False):
 names=load(ROOT/'configs/experiment.json')['datasets'];out=ROOT/'reports/final_seed1014'
 if out.exists():raise FileExistsError('Already scored; do not alter parameters and regrade')
 closed={};missing=[]
 for name in names:
  for method in METHODS:
   folder,file=paths(name,method);receipt=folder/'output_freeze.json'
   if not file.exists() or not receipt.exists():
    closure=ROOT/'status/closed_unavailable'/f'{name}_{method}.json'
    if not (close_incomplete and closure.exists()):missing.append(name+':'+method);continue
    closed[name+':'+method]={'status':'unavailable','reason':load(closure)}
   else:
    verify(folder,receipt);closed[name+':'+method]={'status':'closed','output_sha256':digest(file),'file':str(file),'freeze_sha256':digest(receipt)}
 if missing:raise RuntimeError('Scorer refuses formal truth before all arms close: '+', '.join(missing))
 out.mkdir(parents=True)
 save(out/'all_outputs_closed_before_scoring.json',{'arms':closed,'algorithm_adapters':freeze(ROOT/'code'),'seed':SEED,'timestamp':time.time(),
 'formal_reference_not_yet_read':True,'ranking_policy_not_changed_from_test_scores':True})
 scores={}
 for name in names:
  pub=ROOT/'datasets'/name/'public';private=ROOT/'datasets'/name/'private'
  cols=load(pub/'declaration.json')['targets'];_,dirty=read_csv(pub/'repair_dirty.csv');_,hold=read_csv(pub/'holdout_dirty.csv')
  _,truth=read_csv(private/'repair_reference.csv');_,holdtruth=read_csv(private/'holdout_reference.csv')
  score={'dataset':name,'metadata':load(pub/'dataset_metadata.json'),'methods':{}}
  for method in METHODS:
   state=closed[name+':'+method]
   if state['status']=='unavailable':score['methods'][method]=state;continue
   _,result=read_csv(state['file'])
   m=repair_metrics(dirty,result,truth,cols)
   try:down=classification(result,truth,hold,holdtruth,cols)
   except Exception as e:down={'status':'failed','error':type(e).__name__,'message':str(e)[:200]}
   score['methods'][method]={'status':'complete','repair':m,'downstream':down,'method_receipt':load(paths(name,method)[0]/'receipt.json'),'safety_guardrail_pass':m['P']>=.98 and m['normal_damage']<=.005}
  try:score['no_repair_downstream']=classification(dirty,truth,hold,holdtruth,cols)
  except Exception as e:score['no_repair_downstream']={'status':'failed','error':type(e).__name__}
  scores[name]=score;save(out/(name+'.json'),score)
  print('SCORED_CLOSED_DATASET',name,flush=True)
 save(out/'scores.json',scores)
 report=['# 种子1014正式对照结果','',
 '固定V23；一个Qwen3.5-9B BF16；不看原文。每表按组约10%学习、10%独立校验、64%修复训练、16%分类留出；已知目标格污染70%。',
 '本轮只有一个种子，不提供虚构的均值±标准差。正常未知格不纳入精确修复评分。检测AUC统一用实际是否修改的0/1决定计算；分类AUC使用同一LR的预测概率。','']
 for sec,met,title in [('repair','F1','修复F1总览'),('downstream','macro_F1','同一分类器的分类F1总览')]:
  report+=['## '+title,'','| 方法 | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
  for method in METHODS:
   vals=[]
   for name in names:
    item=scores[name]['methods'][method]
    v=item.get(sec,{}).get(met)
    cell='N/A' if v is None else f'{v:.3f}'
    if item.get('method_receipt',{}).get('fallback'):cell+='†'
    vals.append(cell)
   report.append('| '+LABELS[method]+' | '+' | '.join(vals)+' |')
  report+=['','†表示代码工具未得到可执行程序、明确回退同次纯9B，不能当独立代码方法的证据。','']
 for name,score in scores.items():
  report+=['## '+name,'','| 方法 | 修复P% | 修复R% | 修复F1 | 正常格误改%↓ | 检测AUC | 分类F1 | 分类AUC |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for method in METHODS:
   x=score['methods'][method]
   if x['status']!='complete':report.append('| '+LABELS[method]+' | 未完成 | — | — | — | — | — | — |');continue
   m=x['repair'];d=x['downstream']
   def fmt(v):return '—' if v is None else f'{v:.3f}'
   method_label=LABELS[method]+('（回退纯9B）' if x.get('method_receipt',{}).get('fallback') else '')
   report.append('| '+method_label+f" | {m['P']*100:.2f} | {m['R']*100:.2f} | {m['F1']:.3f} | {m['normal_damage']*100:.3f} | "+fmt(m['detection_AUC'])+' | '+fmt(d.get('macro_F1'))+' | '+fmt(d.get('AUC'))+' |')
  report+=['']
 wins={}
 for method in METHODS[:-1]:
  pair=[]
  for name,score in scores.items():
   a=score['methods']['v23'];b=score['methods'][method]
   if a['status']==b['status']=='complete':pair.append((a,b))
  wins[method]={'available_pairs':len(pair)}
  if method=='tool_llm':
   active=[(a,b) for a,b in pair if not b.get('method_receipt',{}).get('fallback')]
   wins[method]['fallback_pairs']=len(pair)-len(active)
   wins[method]['executable_program_pairs']=len(active)
   wins[method]['executable_program_comparisons']={}
   for sec,met in [('repair','F1'),('repair','detection_AUC'),('downstream','macro_F1'),('downstream','AUC')]:
    v=[(a[sec].get(met),b[sec].get(met)) for a,b in active if a[sec].get(met) is not None and b[sec].get(met) is not None]
    wins[method]['executable_program_comparisons'][sec+'_'+met]={'wins':sum(a>b+1e-12 for a,b in v),'ties':sum(abs(a-b)<=1e-12 for a,b in v),'losses':sum(a<b-1e-12 for a,b in v)}
  for section,metric in [('repair','F1'),('repair','detection_AUC'),('downstream','macro_F1'),('downstream','AUC')]:
   valid=[(a[section].get(metric),b[section].get(metric)) for a,b in pair if a[section].get(metric) is not None and b[section].get(metric) is not None]
   wins[method][section+'_'+metric]={'wins':sum(a>b+1e-12 for a,b in valid),'ties':sum(abs(a-b)<=1e-12 for a,b in valid),'losses':sum(a<b-1e-12 for a,b in valid)}
 save(out/'paired_comparisons.json',wins)
 report+=['## V23与各方法的胜平负','','| 对照方法 | 有效配对表数 | 修复F1 胜/平/负 | 二值决定AUC 胜/平/负 | 分类F1 胜/平/负 | 分类AUC 胜/平/负 |','|---|---:|---|---|---|---|']
 for method,w in wins.items():
  counts=[]
  for key in ['repair_F1','repair_detection_AUC','downstream_macro_F1','downstream_AUC']:
   c=w.get(key,{});counts.append('/'.join(str(c.get(k,0)) for k in ['wins','ties','losses']))
  report.append('| '+LABELS[method]+' | '+str(w['available_pairs'])+' | '+' | '.join(counts)+' |')
 fallback_names=[name for name,s in scores.items() if s['methods']['tool_llm'].get('method_receipt',{}).get('fallback')]
 report+=['','代码工具回退纯9B的数据集：'+('、'.join(fallback_names) if fallback_names else '无')+'。回退不是代码程序修复成功，复用同次真实纯9B输出并单独记录，不据此声称击败独立强代码方法。','']
 tool=wins['tool_llm']
 report+=['代码工具的独立可执行程序配对表数：'+str(tool['executable_program_pairs'])+'；安全回退配对表数：'+str(tool['fallback_pairs'])+'。上方总胜平负包含透明标注的回退，不能把这些回退配对视为独立代码方法的证据。独立可执行程序的胜平负另见 paired_comparisons.json 中 executable_program_comparisons。','']
 report+=['## 不修复时的同一分类器成绩','','| 数据集 | 分类F1 | 分类AUC |','|---|---:|---:|']
 for name,s in scores.items():
  d=s['no_repair_downstream'];report.append('| '+name+' | '+('—' if d.get('macro_F1') is None else f"{d['macro_F1']:.3f}")+' | '+('—' if d.get('AUC') is None else f"{d['AUC']:.3f}")+' |')
 report+=['','MONKS1、CMC为既有来源重新生成种子1014的数据，不称全新盲测。学习与校验优先使用完整记录，天然缺失较多的表可能因此出现教学样本偏差。所有方法使用同一目标输入；CDC限时CPU设置与GARF移植等适配限制按方法单独披露。','']
 passed=[name for name,s in scores.items() if s['methods']['v23'].get('safety_guardrail_pass')]
 report+=['V23满足修复P≥98%、正常误改≤0.5%的数据集数量：'+str(len(passed))+'/'+str(len(names))+'。未达标表保持原分数，不删除或挑选。','']
 report+=['## 结论的范围','','具体胜、平、负数量见 paired_comparisons.json；仅对完成的同条件配对作结论。未完成方法不计成0分。','',
 '留出组若缺少某类别而无法定义分类AUC，保留已有分类F1，AUC记为不可计算并记录原因；不更换样本划分、不填0、不计入AUC胜负。',
 '这是一轮固定污染方式、固定样本划分和单种子对照，不能证明普遍最优或统计显著。GARF使用CDC作者SQLite移植；CDC采用源内分组校验、小样本类型适配及每字段60秒CPU预算，不称论文原设置完全复现。']
 (out/'对照表格和结论.md').write_text('\n'.join(report)+'\n')
 save(out/'scoring_complete.json',{'seed':SEED,'datasets':names,'formal_reference_used_only_by_scorer':True,'scored_once':True,'report':str(out/'对照表格和结论.md')})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--close-incomplete',action='store_true');a=p.parse_args()
 self_test() if a.self_test else main(a.close_incomplete)
