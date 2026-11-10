"""Run the three paper experiments without consuming recorded expected metrics."""
from pathlib import Path
import argparse,subprocess,sys
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--work",default="runs");p.add_argument("--seeds",nargs="+",type=int,default=[1013,1014,1015],choices=[1013,1014,1015]);a=p.parse_args()
for seed in a.seeds:
 for study in ["repair","downstream","ablation"]:
  subprocess.run([sys.executable,str(R/"reproduce.py"),"run","--study",study,"--seed",str(seed),"--work",a.work,"--config",a.config],check=True)
if len(a.seeds)==3:subprocess.run([sys.executable,str(R/"reproduce.py"),"tables","--source","runs","--work",a.work,"--output","results/rerun_tables"],check=True)
