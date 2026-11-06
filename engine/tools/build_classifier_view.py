"""Keep core table immutable; seal a CV-selected classifier training view."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from learning.classifier_selection import select
from unseen_table.common import read_table,write_table,dump,fresh,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=fresh(a.out);fields,rows=read_table(a.input);view,result=select(rows,a.skill)
    write_table(root/'classifier_training_view.csv',fields,view);dump(root/'training_cv_selection.json',result)
    dump(root/'view_freeze.json',{'files':{n:sha256(root/n) for n in ['classifier_training_view.csv','training_cv_selection.json']},
         'input_sha256':sha256(a.input),'core_table_modified':False,'holdout_read':False,'formal_cell_gold_read':False})
    print('CLASSIFIER_CV_VIEW_FROZEN',result['selected'],flush=True)


if __name__=='__main__':main()
