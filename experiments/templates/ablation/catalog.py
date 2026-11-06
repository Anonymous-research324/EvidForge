CATALOG={'tic_tac_toe': {'id': 101, 'slug': 'tic+tac+toe+endgame', 'task': 'classification', 'rows': 958, 'columns': 9, 'layout': 'tic'}, 'car_evaluation': {'id': 19, 'slug': 'car+evaluation', 'task': 'classification', 'rows': 1728, 'columns': 6, 'layout': 'car'}, 'retinopathy': {'id': 329, 'slug': 'diabetic+retinopathy+debrecen', 'task': 'classification', 'rows': 1151, 'columns': 19, 'layout': 'retino'}, 'real_estate': {'id': 477, 'slug': 'real+estate+valuation+data+set', 'task': 'regression', 'rows': 414, 'columns': 6, 'layout': 'estate'}, 'concrete': {'id': 165, 'slug': 'concrete+compressive+strength', 'task': 'regression', 'rows': 1030, 'columns': 8, 'layout': 'concrete'}, 'airfoil': {'id': 291, 'slug': 'airfoil+self+noise', 'task': 'regression', 'rows': 1503, 'columns': 5, 'layout': 'airfoil'}}
NAMES=list(CATALOG)
METHODS=['dirty','pure_llm','tool_llm','raha_holoclean','garf','cdc','v23']
CLASSIFICATION=[n for n in NAMES if CATALOG[n]['task']=='classification']
XGB_PARAMETERS=dict(n_estimators=200,max_depth=4,learning_rate=.05,subsample=1.,colsample_bytree=1.,reg_lambda=1.,tree_method='hist',device='cpu',n_jobs=4,random_state=1014,verbosity=0)
# Public input-type encoding, no correction rules and no use of formal answers.
CAR_VOCAB=[['vhigh','high','med','low'],['vhigh','high','med','low'],['2','3','4','5more'],['2','4','more'],['small','med','big'],['low','med','high']]
TIC_VOCAB=[['x','o','b']]*9
