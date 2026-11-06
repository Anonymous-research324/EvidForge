from common import ROOT,OLD,load,save
NAMES=['tic_tac_toe','car_evaluation','retinopathy','real_estate','concrete','airfoil']
LABELS={'A':'完整EvidForge（同轮重学）','B':'关闭纠错变换候选','C':'关闭9B新增规则','D':'关闭最终置信门槛','E':'关闭反馈改写','F':'关闭项目Hook'}
def build(results):
 lookup={(x['dataset'],x['arm']):x for x in results}
 lines=['# EvidForge 消融实验：选定六表','分类格为宏F1/AUC，回归格为R²/MAE。只用seed1014；同一数据、同XGBoost。A为本轮配对完整学习，不覆盖旧实验。','',
 '| 方法 / 数据集 | '+' | '.join(NAMES)+' |','|---|'+'---:|'*6]
 for a,label in LABELS.items():
  vals=[]
  for n in NAMES:
   d=lookup[(n,a)];m=d['downstream'];keys=['macro_F1','AUC'] if d['task']=='classification' else ['R2','MAE']
   vals.append(' / '.join('N/A' if m[k] is None else f'{m[k]:.3f}' for k in keys))
  lines.append('| '+label+' | '+' | '.join(vals)+' |')
 lines+=['','## 准确修复质量','每格为准确修复F1 / 正常格误改率(%)。','',
 '| 方法 / 数据集 | '+' | '.join(NAMES)+' |','|---|'+'---:|'*6]
 for a,label in LABELS.items():
  vals=[]
  for n in NAMES:
   m=lookup[(n,a)]['repair'];vals.append(f"{m['F1']:.3f} / {100*m['damage']:.3f}%")
  lines.append('| '+label+' | '+' | '.join(vals)+' |')
 lines+=['','A_saved逐格复现旧完整输出的记录单独保留；A/E/F分别从相同教学起点重新学习，E提案次数配对A且提案前不显示评价反馈，F只关闭项目Hook，后端权限和校验不关闭。','B/C/D只关闭相应推理机制，不重新训练或按测试结果补偿。组件未激活或输出相同必须如实展示；单组件差值不能直接相加。六表来自此前选定展示，属于同benchmark追加分析，不称新盲测或普遍泛化证据。']
 (ROOT/'reports/消融实验对照表.md').write_text('\n'.join(lines)+'\n')
