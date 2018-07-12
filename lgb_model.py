# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lgb_model
   Description :
   Author :       Administrator
   date：          2018/6/8 0008
-------------------------------------------------
   Change Activity:
                   2018/6/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
# https://github.com/wanglei5205/Machine_learning/blob/master/Boosting--LightGBM/lgb-python/2.lightgbm调参案例.py

import pandas as pd
import numpy as np
from feature_extraction import *

data = pd.read_csv('data/data.csv')
log = pd.read_csv('data/log.csv')

# oversample
# pos_train = data[data['FLAG'] == 1]
# neg_train = data[data['FLAG'] == 0]
# test_data = data[data['FLAG'] == -1]
#
# p = 0.04
# scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
# while scale > 1:
#     neg_train = pd.concat([neg_train, neg_train])
#     scale -= 1
# neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
# print(len(pos_train) / (len(pos_train) + len(neg_train)))
#
# data = pd.concat([pos_train, neg_train,test_data])
# del pos_train, neg_train

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
# user_other = user_other(log)
# user_time_stat2 = user_time_stat2(log)

data = pd.merge(data,next_time_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_tch_type_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_time_stat,on=['USRID'],how='left',copy=False)
# data = pd.merge(data,user_other,on=['USRID'],how='left',copy=False)
# data = pd.merge(data,user_time_stat2,on=['USRID'],how='left',copy=False)

data.fillna(0, inplace=True)
print(data.shape)

from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]


# train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
col = test.columns
test = test[col].values

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []

for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc', 'binary_logloss'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'scale_pos_weight': 6
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train, # 训练集
                    valid_sets=lgb_eval,  # 验证集
                    num_boost_round=40000, # 迭代次数 40000 -> 10000
                    verbose_eval=250, # 每隔250次，打印日志
                    early_stopping_rounds=50)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0
for i in xx_pre:
    s = s + i

s = s /N

res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(s)

print('best xx score',np.mean(xx_cv))

### 保存模型
# from sklearn.externals import joblib
# joblib.dump(gbm,'lgb.pkl')



# import operator
# importance = dict(gbm.feature_importance(train.columns.tolist()))
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# df = pd.DataFrame(gbm.feature_importance(train.columns.tolist()), columns=['feature', 'importance'])
# df['importance'] = df['importance'] / df['importance'].sum()
# df.to_csv('{0}_lgbm_{1}{2}'.format(model_path, exec_time, model_feature_importance_csv), index=False)

import time
time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('data/submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')
