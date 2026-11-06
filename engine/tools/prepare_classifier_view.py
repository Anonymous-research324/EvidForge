"""Export classifier training input without modifying the core repaired CSV."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from learning.downstream_view import fit,apply
from unseen_table.common import load,dump,read_table,write_table,fresh,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--input')
    p.add_argument('--out',required=True);p.add_argument('--policy');a=p.parse_args()
    root=fresh(a.out)
    if a.policy:policy=load(a.policy)
    else:policy=fit(a.skill)
    dump(root/'policy.json',policy)
    if a.input:
        fields,rows=read_table(a.input);view,flags=apply(rows,a.skill,policy)
        write_table(root/'classifier_training_view.csv',fields,view);dump(root/'numeric_flags.json',flags)
        dump(root/'view_freeze.json',{'files':{n:sha256(root/n) for n in ['policy.json','classifier_training_view.csv','numeric_flags.json']},
             'input_sha256':sha256(a.input),'core_repaired_table_modified':False,'classifier_setting_change':False,
             'formal_reference_read':False,'flagged_numeric_cells':len(flags),'repair_metrics_apply_to':'original core output only'})
    print('CLASSIFIER_VIEW_ONLY_FROZEN',flush=True)


if __name__=='__main__':main()
