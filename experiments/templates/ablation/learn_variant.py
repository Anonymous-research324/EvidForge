import argparse,os
from common import ROOT,save
from learning_flow import learn
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',choices=['A','E','F'],required=True)
a=p.parse_args();selected=learn(a.dataset,a.arm,dict(os.environ))
print('LEARNING_VARIANT_CLOSED',a.dataset,a.arm,str(selected),flush=True)
