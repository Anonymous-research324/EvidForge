"""Zero-new-model execution of a sealed package. Run through a generated case."""
from pathlib import Path
import argparse,sys,time,json,hashlib
from common import ROOT,load,read_csv,write_csv,save,freeze
import seed_runtime
seed_runtime.install()
from workflow.packages import execute
p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--out",required=True);a=p.parse_args()
selection=load(ROOT/"outputs/v23"/a.dataset/"selection.json");skill=Path(selection["skill"])
fields,rows=read_csv(ROOT/"datasets"/a.dataset/"public/repair_dirty.csv")
before=len(list((ROOT/"models_service/requests").glob("*/request.json")))
t=time.monotonic();output,decisions=execute(rows,skill)
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);write_csv(out/"repair_output.csv",fields,output)
assert (out/"repair_output.csv").read_bytes()==(ROOT/"outputs"/a.dataset/"primary_llm/v23_output.csv").read_bytes(),"Sealed-skill replay differs"
after=len(list((ROOT/"models_service/requests").glob("*/request.json")));assert after==before
save(out/"receipt.json",{"byte_identical":True,"additional_model_calls":0,"seconds":time.monotonic()-t,"formal_truth_read":False})
save(out/"output_freeze.json",{"files":freeze(out)})
print("SEALED_SKILL_REUSE_PASS")
