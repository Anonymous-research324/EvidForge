"""Recompute mean and sample SD from individual seeds, without reading gold data."""
from pathlib import Path
import csv,json,math,statistics
SEEDS=[1013,1014,1015]
LABELS={"pure_llm":"纯 LLM（9B）","tool_llm":"LLM＋代码工具","raha_holoclean":"Raha＋HoloClean","garf":"GARF（SQLite 移植）","cdc":"CDC（限时 CPU 适配）","v23":"EvidForge","dirty":"未修复＋基础预处理","caafe":"CAAFE（9B 适配）","A":"完整 EvidForge","B":"关闭学得纠错变换","C":"关闭新增规则执行","D":"关闭最终置信门槛","E":"关闭源内反馈改写","F":"关闭项目Hook"}
DISPLAY={"monks1":"MONKS1","cmc":"CMC","bc_original":"BC Original","mammographic":"Mammographic","ilpd":"ILPD","climate":"Climate","voting":"Voting","qsar":"QSAR","tic_tac_toe":"Tic-Tac-Toe","car_evaluation":"Car","retinopathy":"Retinopathy","real_estate":"Real Estate","concrete":"Concrete","airfoil":"Airfoil"}
def stats(values):
 good=[x for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
 return {"per_seed":dict(zip(map(str,SEEDS),values)),"n":len(good),"expected_n":3,"mean":statistics.mean(good) if len(good)==3 else None,"sample_sd":statistics.stdev(good) if len(good)==3 else None,"ddof":1}
def cell(s):
 return "N/A ("+str(s["n"])+"/3)" if s["mean"] is None else f'{s["mean"]:.3f} ± {s["sample_sd"]:.3f}'
def normalise(data,study):
 if study=="repair" and isinstance(data,dict):
  return [{"dataset":n,"method":m,"status":x["status"],"metrics":x.get("repair"),"fallback":bool(x.get("method_receipt",{}).get("fallback"))} for n,v in data.items() for m,x in v["methods"].items() if m!="raha_baran"]
 if study=="ablation":
  return [{**x,"method":x.get("method",x.get("arm")),"status":x.get("status","complete")} for x in data if x.get("method",x.get("arm")) in ["A","B","C","D","E","F"]]
 return data
def build(repo,out,source="recorded",work=None):
 out.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((repo/"experiments/protocol.json").read_text());records=[];tables=[]
 for study in ["repair","downstream","ablation"]:
  datasets=protocol["datasets"]["repair" if study=="repair" else "downstream"]
  byseed={}
  for s in SEEDS:
   if source=="recorded":path=repo/"results/recorded"/f"{study}_seed{s}.json"
   else:
    root=work/f"{study}_seed{s}"
    path=root/("reports/scores.json" if study=="ablation" else f"reports/final_seed{s}/scores.json")
   rows=normalise(json.loads(path.read_text()),study)
   byseed[s]={(r["dataset"],r["method"]):r for r in rows}
  methods=(protocol["methods"] if study=="repair" else ["dirty","pure_llm","tool_llm","raha_holoclean","garf","cdc","caafe","v23"] if study=="downstream" else protocol["ablation_arms"])
  main=[]
  for method in methods:
   cells=[]
   for name in datasets:
    if method=="caafe" and name in ["real_estate","concrete","airfoil"]:cells.append("—");continue
    entries=[byseed[s].get((name,method),{}) for s in SEEDS]
    metricnames=["P","R","F1","normal_damage","detection_AUC"] if study=="repair" else (["macro_F1","AUC"] if name in ["tic_tac_toe","car_evaluation","retinopathy"] else ["R2","MAE"]) if study=="downstream" else ["P","R","F1","damage"]
    metrics={}
    for metric in metricnames:
     values=[e.get("repair" if study=="ablation" else "metrics",{}).get(metric) if isinstance(e.get("repair" if study=="ablation" else "metrics"),dict) else None for e in entries]
     metrics[metric]=stats(values)
    fallbacks=[s for s,e in zip(SEEDS,entries) if e.get("fallback") or e.get("method_receipt",{}).get("fallback")]
    records.append({"study":study,"dataset":name,"method":method,"metrics":metrics,"fallback_seeds":fallbacks,"source_qualified_per_seed":{str(s):e.get("source_qualified") for s,e in zip(SEEDS,entries)}})
    key="F1" if study!="downstream" else "macro_F1" if "macro_F1" in metricnames else "R2"
    cells.append(cell(metrics[key])+("†" if fallbacks else ""))
   main.append("| "+LABELS[method]+" | "+" | ".join(cells)+" |")
  text="# "+{"repair":"表1：准确修复 F1","downstream":"表2：下游表现（分类宏 F1；回归 R²）","ablation":"表3：消融实验准确修复 F1"}[study]+"\n\n均值 ± 样本标准差；种子1013、1014、1015，ddof=1。N/A不填0。\n\n"
  sizes={"monks1":(432,6),"cmc":(1473,9),"bc_original":(699,9),"mammographic":(961,5),"ilpd":(583,10),"climate":(540,18),"voting":(435,16),"qsar":(1055,41),"tic_tac_toe":(958,9),"car_evaluation":(1728,6),"retinopathy":(1151,19),"real_estate":(414,6),"concrete":(1030,8),"airfoil":(1503,5)}
  metadata="| 样本数 | "+" | ".join(str(sizes[n][0]) for n in datasets)+" |\n| 特征数 | "+" | ".join(str(sizes[n][1]) for n in datasets)+" |\n"
  text+="| 方法 | "+" | ".join(DISPLAY[n] for n in datasets)+" |\n|---|"+"---:|"*len(datasets)+"\n"+metadata+"\n".join(main)+"\n\n†代码工具至少一个种子回退同次纯LLM；具体种子见aggregate.json。\n"
  (out/(study+".md")).write_text(text);tables.append(text)
 (out/"三个主表.md").write_text("\n\n".join(tables))
 (out/"aggregate.json").write_text(json.dumps({"source":source,"seeds":SEEDS,"ddof":1,"records":records},ensure_ascii=False,indent=2,allow_nan=False)+"\n")
 with (out/"metrics.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["study","dataset","method","metric","seed1013","seed1014","seed1015","n","mean","sample_sd"])
  for r in records:
   for k,v in r["metrics"].items():w.writerow([r["study"],r["dataset"],r["method"],k,*v["per_seed"].values(),v["n"],v["mean"],v["sample_sd"]])
 print("Generated three tables and metric CSV from "+source+" seed-level scores: "+str(out))
