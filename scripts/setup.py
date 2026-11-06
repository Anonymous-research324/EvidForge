"""Create isolated environments and fetch pinned baseline sources."""
from pathlib import Path
import argparse,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
def run(args):subprocess.run(list(map(str,args)),check=True)
def environment(name,python):
 lock=json.loads((R/"environments"/(name+".lock.json")).read_text())
 actual=subprocess.check_output([python,"-c","import sys;print('.'.join(map(str,sys.version_info[:2])))"],text=True).strip()
 expected=".".join(lock["python"].split(".")[:2])
 if actual!=expected:raise ValueError(name+" requires Python "+expected+"; got "+actual)
 folder=R/".venvs"/name;exe=folder/"bin/python"
 if not exe.exists():run([python,"-m","venv",folder])
 run([exe,"-m","pip","install","-r",R/"environments"/(name+".txt")])
 if name=="core":run([exe,"-m","pip","install","-r",R/"environments/extra_core.txt"])
 if name=="cdc":
  for source in ["conformal-inference","jenga"]:run([exe,"-m","pip","install","--no-deps",R/".sources"/source])
def sources():
 pins=json.loads((R/"provenance/baseline_sources.json").read_text())
 for name,spec in pins.items():
  dest=R/".sources"/name
  if not dest.exists():run(["git","clone","--no-checkout",spec["url"],dest])
  if subprocess.run(["git","-C",dest,"cat-file","-e",spec["commit"]],capture_output=True).returncode:run(["git","-C",dest,"fetch","origin",spec["commit"]])
  run(["git","-C",dest,"checkout","--detach",spec["commit"]])
  if subprocess.check_output(["git","-C",dest,"rev-parse","HEAD"],text=True).strip()!=spec["commit"]:raise ValueError("Source fingerprint mismatch")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("component",choices=["core","baselines","inference","tabclean","all"]);p.add_argument("--python312",default=sys.executable);p.add_argument("--python39",default="python3.9");a=p.parse_args()
 if a.component in ["core","all"]:environment("core",a.python312)
 if a.component in ["baselines","all"]:
  sources()
  for name in ["raha","holoclean","garf","cdc"]:environment(name,a.python39)
 if a.component in ["inference","all"]:environment("inference",a.python312)
 if a.component=="tabclean":
  sources()
  run([R/".venvs/core/bin/python","-m","pip","install","--target",R/".dependencies/tabclean","--no-deps","-r",R/"environments/tabclean_target.txt"])
 print("Install Codex CLI 0.153.1 and PostgreSQL separately as documented.")
if __name__=="__main__":main()
