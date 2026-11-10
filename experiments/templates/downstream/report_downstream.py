from common import ROOT,save
from catalog import CATALOG,METHODS
LABELS={'dirty':'脏表＋基本预处理','pure_llm':'纯9B','tool_llm':'9B＋代码工具','raha_holoclean':'Raha＋HoloClean','garf':'GARF（SQLite移植）','cdc':'CDC（限时CPU适配）','v23':'EvidForge','clean_reference':'洁净训练表参考','caafe':'CAAFE（9B适配）'}
def build(results):
 out=ROOT/'reports/final_seed1014';out.mkdir(parents=True,exist_ok=True)
 table={(r['dataset'],r['method']):r for r in results};texts=[];paired=[]
 for task in ['classification','regression']:
  names=[n for n,s in CATALOG.items() if s['task']==task]
  text='# '+('分类下游：macro-F1 / 概率AUC' if task=='classification' else '回归下游：R² / MAE')+'\n\n'
  text+='| 方法 / 数据集 | '+' | '.join(names)+' |\n|---|'+'---:|'*len(names)+'\n'
  text+='| 样本数 | '+' | '.join(str(CATALOG[n]['rows']) for n in names)+' |\n'
  text+='| 特征列数 | '+' | '.join(str(CATALOG[n]['columns']) for n in names)+' |\n'
  metrics=['macro_F1','AUC'] if task=='classification' else ['R2','MAE']
  for method in [*METHODS,'clean_reference']+(['caafe'] if task=='classification' else []):
   cells=[]
   for n in names:
    r=table.get((n,method),{})
    if r.get('status')!='complete':cells.append('N/A');continue
    m=r['metrics'];cells.append(' / '.join('N/A' if m.get(k) is None else f'{m[k]:.3f}' for k in metrics))
   text+='| '+LABELS[method]+' | '+' | '.join(cells)+' |\n'
  text+='\n分类两项越高越好；回归R²越高越好、MAE越低越好。不同回归表的MAE单位不同，不直接平均。单种子1014，无±和显著性结论。\n'
  text+='所有方法使用相同固定XGBoost、训练/测试行和训练内预处理。测试表未污染且未提供给修复方法；本实验评价脏训练数据修复后对独立干净样本预测的帮助。\n'
  text+='GARF和CDC为明确披露的移植/限时适配，CAAFE为分类特征工程补充；不能称原论文全部设置复现。代码工具若回退纯LLM见scores.json，N/A不是0分。洁净表是参考条件而非理论上限。选定方法比较，不宣称覆盖全部已有方法。\n'
  filename='分类下游对照表.md' if task=='classification' else '回归下游对照表.md'
  (out/filename).write_text(text);texts.append(text)
  for other in [m for m in METHODS if m!='v23']+(['caafe'] if task=='classification' else []):
   for metric in metrics:
    counts={'win':0,'tie':0,'loss':0,'not_paired':0}
    for n in names:
     a=table.get((n,'v23'),{}).get('metrics',{}).get(metric);b=table.get((n,other),{}).get('metrics',{}).get(metric)
     if a is None or b is None:counts['not_paired']+=1;continue
     diff=(b-a) if metric=='MAE' else (a-b)
     counts['win' if diff>1e-12 else 'loss' if diff< -1e-12 else 'tie']+=1
    paired.append({'task':task,'comparison':other,'metric':metric,**counts})
 save(out/'paired_comparisons.json',paired)
 (out/'对照表格和结论.md').write_text('\n\n'.join(texts)+'\n\n逐指标有效配对胜平负见paired_comparisons.json；不得根据部分指标把全部任务概括为全面胜出。\n')
