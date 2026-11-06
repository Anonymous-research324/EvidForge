from pathlib import Path
import importlib.util,json,re,subprocess,sys,tempfile,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"experiments"))
from tables import stats,build
from release_runtime import check_engine,settings
class ArtifactTests(unittest.TestCase):
 def test_immutable_core(self):self.assertEqual(check_engine(),124)
 def test_sample_sd_and_missing(self):
  self.assertEqual(stats([1.,2.,3.])["sample_sd"],1.)
  x=stats([1.,None,3.]);self.assertEqual(x["n"],2);self.assertIsNone(x["mean"]);self.assertIsNone(x["sample_sd"])
 def test_virtualenv_interpreter_path(self):
  with tempfile.TemporaryDirectory() as td:
   alias=Path(td)/"python";alias.symlink_to(sys.executable)
   config=Path(td)/"machine.json";config.write_text(json.dumps({"run_python":str(alias)}))
   self.assertEqual(settings(config)["run_python"],str(alias))
 def test_materialized_worker_denies_private_and_y(self):
  from release_runtime import initialize
  with tempfile.TemporaryDirectory() as td:
   root=initialize(Path(td)/"runs","repair",1014,settings(None))
   paths=[root/"datasets/example/private/probe.txt",root/"curator_cache/private/probe.txt",root/"downstream_training/labels.csv",root/"datasets/example/public/holdout_dirty.csv"]
   for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_text("synthetic")
   code="import sys;sys.path.insert(0,"+repr(str(root/"code"))+");from role_guard import install;install()\n"
   code+="for name in "+repr(list(map(str,paths)))+":\n try:open(name).read();raise AssertionError('Private read permitted')\n except PermissionError:pass\n"
   subprocess.run([sys.executable,"-B","-c",code],check=True,capture_output=True,text=True)
 def test_appendix_risk_tables(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);build(R,out);rows=json.loads((out/"aggregate.json").read_text())["records"];by={(x["study"],x["dataset"],x["method"]):x for x in rows}
   protocol=json.loads((R/"experiments/protocol.json").read_text())
   for study,name,methods,metrics,precision in [
    ("repair","rq1_risk",["v23"],["P","R","normal_damage"],2),
    ("ablation","rq3_risk_three",["A","B","C","D","E"],["damage"],3)]:
    actual=re.findall(r"(-?\d+\.\d+)\\,\\pm\\,(-?\d+\.\d+)",(R/"tests/fixtures/main_tables"/(name+".tex")).read_text())
    expected=[];datasets=protocol["datasets"]["repair" if study=="repair" else "downstream"]
    for method in methods:
     for dataset in datasets:
      for metric in metrics:
       v=by[(study,dataset,method)]["metrics"][metric]
       expected.append((format(v["mean"]*100,"."+str(precision)+"f"),format(v["sample_sd"]*100,"."+str(precision)+"f")))
    self.assertEqual(actual,expected,name)
 def test_all_aggregates_independent_formula(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);build(R,out)
   for row in json.loads((out/"aggregate.json").read_text())["records"]:
    for metric,v in row["metrics"].items():
     x=list(v["per_seed"].values())
     if any(a is None for a in x):self.assertIsNone(v["sample_sd"]);continue
     mean=sum(x)/3;sd=(sum((a-mean)**2 for a in x)/2)**.5
     self.assertAlmostEqual(v["mean"],mean,places=12);self.assertAlmostEqual(v["sample_sd"],sd,places=12)
 def test_paper_main_tables(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);build(R,out);data=json.loads((out/"aggregate.json").read_text())
   scope=json.loads((R/"experiments/protocol.json").read_text())
   methodorders={"repair":scope["methods"],"downstream":["dirty","pure_llm","tool_llm","raha_holoclean","garf","cdc","caafe","v23"],"ablation":scope["ablation_arms"]}
   filenames={"repair":"rq1_three_seeds","downstream":"rq2_downstream","ablation":"rq3_ablation"}
   by={(x["study"],x["dataset"],x["method"]):x for x in data["records"]}
   for study in filenames:
    actual=[];text=(R/"tests/fixtures/main_tables"/(filenames[study]+".tex")).read_text()
    for a,b in re.findall(r"(-?\d+\.\d+)\\,\\pm\\,(-?\d+\.\d+)",text):actual.append((a,b))
    expected=[]
    for m in methodorders[study]:
     if study=="ablation" and m=="F":continue # historical five-arm main-table fixture; final full appendix F tested separately
     for name in scope["datasets"]["repair" if study=="repair" else "downstream"]:
      x=by.get((study,name,m))
      if x is None:continue
      k="F1" if study!="downstream" else "macro_F1" if "macro_F1" in x["metrics"] else "R2"
      y=x["metrics"][k]
      if y["mean"] is not None:expected.append((f'{y["mean"]:.3f}',f'{y["sample_sd"]:.3f}'))
    self.assertEqual(actual,expected,study)
   missing=by[("repair","qsar","cdc")]["metrics"]["F1"]
   self.assertEqual(missing["n"],2);self.assertIsNone(missing["mean"])
if __name__=="__main__":unittest.main()
