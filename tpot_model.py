# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tpot_model
   Description :
   Author :       Administrator
   date：          2018/6/12 0012
-------------------------------------------------
   Change Activity:
                   2018/6/12 0012:
-------------------------------------------------
"""
__author__ = 'Administrator'
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
from feature_extraction import *

data = pd.read_csv('data/data.csv')
log = pd.read_csv('data/log.csv')


# 组合特征(提高1个千分位)
data['V9_10'] = data['V9']  +  data['V10']
data['V19_18'] = data['V19'] + data['V18']
data['V21_23'] = data['V23'] + data['V21'] +  data['V27']
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
print(data.shape)
# print(data[['v2_1_evt_num','v2_2_evt_num']].head())


from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

# train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
col = test.columns
test = test[col].values

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)


print(tpot.score(X_test, y_test))


tpot.export('tpot_mnist_pipeline.py')

# tpot.predict_proba(test)
