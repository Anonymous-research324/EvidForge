"""Publication scope and operational adapters; numerical core remains immutable."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def scopes():return json.loads((ROOT/'provenance/paper_scope.json').read_text())['studies']
def validate_case(study,case):
 cfg=scopes()[study]
 for key,plural in [('dataset','datasets'),('seed','seeds'),('suite','suites')]:
  if key in case and plural in cfg and case[key] not in cfg[plural]:raise ValueError('Case outside publication scope: '+study+'/'+str(case.get('id',case)))
def runtime_source(study,name,source):
 """Narrow dispatch/reporting only. Scoring functions are never replaced."""
 changes=[]
 if study=='threshold' and name=='score_once.py':
  old='"all90_outputs_closed":True'
  if source.count(old)!=1:raise ValueError('Unrecognized threshold closure metadata')
  source=source.replace(old,'"all_outputs_closed":True,"outputs":len(cases)*len(OFFSETS)',1)
  changes.append('Closure metadata reports the selected case count; metric formulas unchanged')
 if study in ['advisor_supplement','robustness_retinopathy'] and name=='worker_v4.py':
  anchor=' execute(case,"full",env)\n'
  if source.count(anchor)!=1:raise ValueError('Unrecognized worker scope binding')
  source=source.replace(anchor,anchor+' save(case/"status/closed.json",{"arms":["full"],"at":time.time(),"formal_reference_read":False})\n return\n',1)
  changes.append('Only full output dispatched; source preparation, native learning and external selection unchanged')
 if study in ['advisor_supplement','robustness_retinopathy'] and name in ['report.py','report_v2.py']:
  source='''from common import ROOT,load,save
import statistics,collections
def stat(values):
 good=[v for v in values if v is not None]
 return {'mean':statistics.mean(good) if len(good)==3 else None,'sd':statistics.stdev(good) if len(good)==3 else None,'n':len(good),'seed_values':values}
def build(results):
 groups=collections.defaultdict(list)
 for row in results:groups[tuple(row[k] for k in ['suite','dataset','rate','error_type','learning_fraction','arm'])].append(row)
 aggregate=[]
 for key,rows in groups.items():
  item=dict(zip(['suite','dataset','rate','error_type','learning_fraction','arm'],key));by={r['seed']:r for r in rows}
  for metric in ['P','R','F1','normal_damage','detection_AUC']:item[metric]=stat([by.get(s,{}).get('repair',{}).get(metric) for s in [1013,1014,1015]])
  for metric in ['macro_F1','AUC','R2','MAE']:item[metric]=stat([(by.get(s,{}).get('downstream') or {}).get(metric) for s in [1013,1014,1015]])
  aggregate.append(item)
 save(ROOT/'reports/aggregate.json',aggregate)
 save(ROOT/'reports/paired_differences.json',[])
if __name__=='__main__':build(load(ROOT/'reports/all_scores.json'))
'''
  changes.append('Report only supplied paper-scope rows; no historical workbook/source-result dependency')
 return source,changes
