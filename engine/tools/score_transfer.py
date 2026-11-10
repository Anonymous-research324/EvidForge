"""Open references only AFTER all repair outputs and code are frozen."""
from pathlib import Path
import argparse,sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,fresh,read_table,sha256
from unseen_table.downstream import classify
from scoring_functions import score,downstream_auc


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--holdout',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    run=Path(a.run);freeze=load(run/'online_output_freeze.json');fixed=load(run/'input_and_code_freeze.json')
    for n,v in freeze['files'].items():
        if sha256(run/n)!=v:raise ValueError('Output changed before scoring')
    for n,v in {**fixed['runtime_code'],**fixed['skill_files']}.items():
        if sha256(n)!=v:raise ValueError('Method/skill changed after freeze')
    schema=load(Path(fixed['protocol']['skill_path'])/'base/skill.json')['declaration']
    _,dirty=read_table(run/'input.csv');_,truth=read_table(a.reference);_,holdout=read_table(a.holdout)
    targets=schema['targets'];identities=schema['entity_fields'];conditions={n:read_table(run/file)[1] for n,file in
        [('冻结前置流程，同学习组','prior_only.csv'),('V10算法，同学习组重学','v10_only.csv'),('V23程序','programme_only.csv'),('纯9B，无原文','A_output.csv'),('9B＋V23学习技能，无原文','B_output.csv')]}
    metrics={name:score(dirty,rows,truth,targets,identities) for name,rows in conditions.items()};down={}
    for name in ['冻结前置流程，同学习组','V10算法，同学习组重学','纯9B，无原文','9B＋V23学习技能，无原文']:
        down[name]=classify(truth,dirty,conditions[name],holdout,targets,identities)
        down[name]['probability_AUC_same_classifier']=downstream_auc(truth,dirty,conditions[name],holdout,targets)
    report={'scope':fixed['protocol']['evaluation_note'],'seed':20260927,'raw_reference_cases_visible':False,
        'real9B_calls':freeze['real9B_calls'],'results':metrics,'downstream_same_original_classifier':down,
        'seconds':freeze['elapsed_seconds'],'programme_seconds':freeze['programme_seconds'],'primary_metric':'exact repair F1',
        'no_formal_tuning':True,'single_seed_no_significance_claim':True,'all_cells_retained':True,
        'A_and_B_both_fresh_calls':True,'V10_algorithm_relearned_on_same_learning_partition':True,
        'qualification':{n:m['P']>=.98 and m['damage']<=.005 for n,m in metrics.items() if n!='纯9B，无原文'}}
    root=fresh(a.report);dump(root/'results.json',report)
    text=['| 方法 | 修复 P % | 修复 R % | 修复 F1 | 误改率 % ↓ | 二值检测 AUC |','|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():text.append(f"| {name} | {100*m['P']:.2f} | {100*m['R']:.2f} | {m['F1']:.3f} | {100*m['damage']:.2f} | {m['AUC']:.3f} |")
    text+=['',report['scope']+' 9B单种子、无原文；A/B均真实调用，输出封存后评分。',
        f"本次真实9B调用{report['real9B_calls']}次，程序{report['programme_seconds']:.1f}秒。"]
    (root/'主结果表.md').write_text('\n'.join(text)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
