"""GARF using the published CDC authors' SQLite port, no truth evaluators."""
from pathlib import Path
import os,sys,time,argparse,random,types,sqlite3
from common import ROOT,SEED,load,save,read_csv,write_csv,freeze
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
os.environ['TF_NUM_INTEROP_THREADS']='4';os.environ['TF_NUM_INTRAOP_THREADS']='4'
import numpy as np,pandas as pd,tensorflow as tf
random.seed(SEED);np.random.seed(SEED);tf.compat.v1.disable_eager_execution();tf.compat.v1.experimental.output_all_intermediates(True);tf.compat.v1.set_random_seed(SEED)
import keras,keras.backend as K
K.set_session=tf.compat.v1.keras.backend.set_session
K.get_session=tf.compat.v1.keras.backend.get_session
from keras.optimizers.optimizer_v2.adam import Adam as LegacyAdam
keras.optimizers.Adam=LegacyAdam
wrappers=types.ModuleType('keras.layers.wrappers');wrappers.TimeDistributed=keras.layers.TimeDistributed;wrappers.Bidirectional=keras.layers.Bidirectional
sys.modules['keras.layers.wrappers']=wrappers
sys.path.insert(0,str(ROOT/'sources/cdc/garf'))
from SeqGAN.train import Trainer
from att_reverse import att_reverse
from rule_sample import rule_sample
import config
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=str(Path(os.fsdecode(args[0])).resolve())
  if '/datasets/' in p and '/private/' in p or '/local_private/' in p:raise PermissionError('Formal truth unavailable')
  if p.endswith(('development_bundle.json','validation_only.json')):raise PermissionError('Training restricted to authorized first10')
sys.addaudithook(guard)
def main(name,probe=False):
 if probe:print('GARF_IMPORT_OK',flush=True);return
 start=time.monotonic();pub=ROOT/'datasets'/name/'public';out=ROOT/'outputs'/name/'garf';out.mkdir(parents=True,exist_ok=False)
 cols=load(pub/'declaration.json')['targets'];groups=load(pub/'learning_only.json')['learning']
 teach=[r for g in groups for r in g['records']]
 fields,val=read_csv(pub/'validation_dirty.csv');_,formal=read_csv(pub/'repair_dirty.csv');n=len(teach);nv=len(val)
 data=[{c:r['truth'][c] for c in cols} for r in teach]+[{c:r[c] for c in cols} for r in val+formal]
 os.chdir(out)
 table='authorized_targets'
 pd.DataFrame(data,columns=cols).astype(str).assign(Label=None).to_sql(table,sqlite3.connect('database.db'),index=False)
 models=out/'models';models.mkdir()
 policy={'seed':SEED,'official_config':{k:getattr(config,k) for k in ['batch_size','g_e','g_h','d_e','d_h','n_sample','generate_samples','g_pre_epochs','d_pre_epochs','g_lr','d_lr','g_pre_lr','d_pre_lr','d_dropout']},
 'max_length':max(config.max_length,len(cols)+2),'runtime_port':'CDC authors published SQLite GARF port','source_groups':len(groups),'source_rows':n,
 'source_clean_teaching_appended':True,'self_supervised_all_dirty_rows':True,'test_gold_read':False,'numeric_hyperparameters_not_tuned_on_formal':True}
 save(out/'method_freeze.json',policy)
 for order in [1,0]:
  att_reverse(table,order,models)
  trainer=Trainer(order=order,B=config.batch_size,T=policy['max_length'],g_E=config.g_e,g_H=config.g_h,d_E=config.d_e,d_H=config.d_h,
    d_dropout=config.d_dropout,generate_samples=config.generate_samples,path_pos=table,path_neg=models/'generated.txt',
    path_rules=models/'rules.txt',models_base_path=models,g_lr=config.g_lr,d_lr=config.d_lr,n_sample=config.n_sample)
  gp=models/'generator_pre.hdf5';dp=models/'discriminator_pre.hdf5';gw=models/'generator.pkl';dw=models/'discriminator.hdf5'
  trainer.pre_train(g_epochs=config.g_pre_epochs,d_epochs=config.d_pre_epochs,g_pre_path=gp,d_pre_path=dp,g_lr=config.g_pre_lr,d_lr=config.d_pre_lr)
  trainer.load_pre_train(gp,dp);trainer.reflect_pre_train()
  trainer.train(steps=1,g_steps=1,head=10,g_weights_path=gw,d_weights_path=dw);trainer.save(gw,dw)
  count=rule_sample(models/'rules.txt',table,order)
  trainer.train_rules(count,models/'rules.txt');trainer.filter(table);att_reverse(table,1,models);trainer.repair(table)
  print('GARF_ORDER_COMPLETE',name,order,flush=True)
 result=pd.read_sql_query(f"SELECT * FROM '{table}'",sqlite3.connect('database.db')).drop(columns='Label')
 assert len(result)==len(data) and result.columns.tolist()==cols
 for zone,rows,offset in [('validation',val,n),('repair',formal,n+nv)]:
  final=[]
  for i,r in enumerate(rows):final.append({**r,**{c:'' if pd.isna(result.iloc[offset+i][c]) else str(result.iloc[offset+i][c]) for c in cols}})
  write_csv(out/(zone+'_output.csv'),fields,final)
 save(out/'receipt.json',{**policy,'elapsed_seconds':time.monotonic()-start,'formal_reference_read':False,'official_original_repo_requires_Oracle':True,
 'compatibility':['TFv1 graph mode','Keras backend session aliases','legacy Adam','TimeDistributed import alias']})
 save(out/'output_freeze.json',{'files':freeze(out),'test_gold_read':False})
 print('GARF_OUTPUT_CLOSED',name,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='bc_original');p.add_argument('--probe',action='store_true');a=p.parse_args();main(a.dataset,a.probe)
