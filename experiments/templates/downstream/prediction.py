"""Shared fixed XGBoost evaluator; fitting sees training rows only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'dependencies'))
import numpy as np,pandas as pd
from decimal import Decimal,InvalidOperation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,roc_auc_score,r2_score,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from catalog import XGB_PARAMETERS
def scalar(v):
 try:
  d=Decimal(str(v))
  return float(d) if d.is_finite() else np.nan
 except (ValueError,InvalidOperation,TypeError):return np.nan
def frame(rows,cols):
 return pd.DataFrame([[scalar(r.get(c,'')) for c in cols] for r in rows],columns=cols)
def prepare_frames(train,test,categorical):
 train=train.replace([np.inf,-np.inf],np.nan).copy();test=test.replace([np.inf,-np.inf],np.nan).copy()
 cat=[c for c in train if c in categorical];num=[c for c in train if c not in cat]
 for c in cat:
  for df in [train,test]:
   df[c]=df[c].map(lambda v:format(Decimal(str(v)).normalize(),'f') if pd.notna(v) else np.nan).astype(object)
 transform=ColumnTransformer([('num',SimpleImputer(strategy='median',keep_empty_features=True),num),
  ('cat',make_pipeline(SimpleImputer(strategy='most_frequent',keep_empty_features=True),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cat)],remainder='drop')
 a=transform.fit_transform(train);b=transform.transform(test)
 if a.shape[1]==0:raise ValueError('No usable training feature')
 return a,b,transform
def evaluate_frames(train,y,test,yt,task,categorical):
 a,b,prep=prepare_frames(train,test,categorical)
 if task=='classification':
  enc=LabelEncoder();encoded=enc.fit_transform(np.asarray(y,dtype=str));classes=enc.classes_
  if len(classes)<2:raise ValueError('Training partition contains fewer than two classes')
  model=XGBClassifier(**XGB_PARAMETERS,objective='binary:logistic' if len(classes)==2 else 'multi:softprob',eval_metric='logloss' if len(classes)==2 else 'mlogloss')
  model.fit(a,encoded);pred=enc.inverse_transform(model.predict(b));prob=model.predict_proba(b)
  score={'macro_F1':float(f1_score(np.asarray(yt,dtype=str),pred,average='macro')),'AUC':None,'AUC_unavailable_reason':None}
  try:
   if len(classes)==2:
    if any(v not in set(classes) for v in yt):raise ValueError('Unseen holdout class')
    auc=roc_auc_score((np.asarray(yt,dtype=str)==classes[1]).astype(int),prob[:,1])
   else:auc=roc_auc_score(np.asarray(yt,dtype=str),prob,labels=classes,multi_class='ovr',average='macro')
   if not np.isfinite(auc):raise ValueError('Undefined AUC')
   score['AUC']=float(auc)
  except ValueError as e:score['AUC_unavailable_reason']=str(e)
 else:
  model=XGBRegressor(**XGB_PARAMETERS,objective='reg:squarederror')
  model.fit(a,np.asarray(y,dtype=float));pred=model.predict(b)
  score={'R2':float(r2_score(np.asarray(yt,dtype=float),pred)),'MAE':float(mean_absolute_error(np.asarray(yt,dtype=float),pred))}
 return score
