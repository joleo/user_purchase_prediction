import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from feature_extraction import *

# NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
data = pd.read_csv('data/data.csv')
log = pd.read_csv('data/log.csv')
# 组合特征(提高1个千分位)
data['V9_10'] = data['V9']  +  data['V10']
data['V19_18'] = data['V19'] + data['V18']
data['V21_23'] = data['V23'] + data['V21'] +  data['V27']
# data['V29_28'] = data['V28'] + data['V29']

# data['V_log_13'] = data['V13'].map(lambda x: np.log(x))
V19_18_col = ['V9','V10','V12','V13','V18', 'V19','V21','V23','V27']
data.drop(V19_18_col, axis=1, inplace=True )


# 添加特征
next_time_stat = user_next_time_stat(log)
user_tch_type_stat = user_tch_type_stat(log)
user_time_stat = user_time_stat(log)

data = pd.merge(data,next_time_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_tch_type_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_time_stat,on=['USRID'],how='left',copy=False)

data.fillna(0, inplace=True)
# print(data.shape)
train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

tpot_data = train

import time

time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('FLAG', axis=1).values

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['FLAG'].values, random_state=42)

# Score on the training set was:0.9642833026775461
exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=8, n_estimators=100, nthread=1, subsample=0.7500000000000001)

exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)
results = exported_pipeline.predict_proba(testing_features)

# print(list(results))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(testing_target, results[:,1])
print('auc value:', auc)

# test_userid = test.pop('USRID')
# test_y = test.pop('FLAG')
# col = test.columns
# test_x =  np.array(test[col].values)
#
#
# test_probs = exported_pipeline.predict(test_x)
#
#
#
# res = pd.DataFrame()
# res['USRID'] = list(test_userid.values)
# res['RST'] = test_probs
# res.to_csv('data/submit/xgb_%s.csv'%str(time_date), index=False, sep='\t')
