"""Aggregate diagnosis only, after outputs freeze; never feed truth to repair.

The historical binary AUC uses actual edits. This diagnostic separates error
suspicion from exact restoration, so detected but unfilled cells are visible.
"""
from pathlib import Path
import argparse,sys,json
from collections import defaultdict
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,read_table,equal,number,sha256,fresh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run=Path(a.run);freeze=load(run/'online_output_freeze.json')
    for n,h in freeze['files'].items():
        if sha256(run/n)!=h:raise ValueError('Outputs changed before diagnostic scoring')
    _,dirty=read_table(run/'input.csv');_,truth=read_table(a.reference);ds=load(run/'programme_decisions.json')
    buckets=defaultdict(lambda:defaultdict(int));labels=[];scores=[];normal_flags=0;normal=0
    for d in ds:
        i,c=d['row'],d['field'];old=dirty[i][c];answer=truth[i][c];error=not equal(old,answer)
        suspect=float(d['detection_score'])>=.5;labels.append(int(error));scores.append(float(d['detection_score']))
        if not error:normal+=1;normal_flags+=suspect;continue
        kind='blank' if not old.strip() else 'affix' if old=='ERR:'+answer else 'scale' if number(old) is not None and number(old)==number(answer)*10 else 'digit'
        b=buckets[kind];b['errors']+=1;b['suspected_fixed_threshold_0_5']+=suspect
        changed=not equal(old,d['after']);correct=changed and equal(d['after'],answer)
        b['changed']+=changed;b['exactly_restored']+=correct;b['suspected_but_not_exactly_restored']+=suspect and not correct
    report={'aggregate_only':True,'no_individual_answer_exported':True,'formal_answers_only_scorer':True,
        'code_and_outputs_already_frozen':True,'threshold_predeclared_0_5_no_tuning':True,
        'buckets':dict(buckets),'normal':normal,'suspected_normal_cells':normal_flags,
        'continuous_detector_AUC':float(roc_auc_score(labels,scores)),
        'continuous_AUC_is_not_the_primary_binary_edit_AUC':'Do not mix this with pureLLM edit AUC or replace old reports',
        'suspected_does_not_mean_restorable':'Exact repair recall counts only correct replacements, not detected errors'}
    root=fresh(a.out);dump(root/'aggregate.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
