# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lr_model
   Description :
   Author :       Administrator
   date：          2018/6/11 0011
-------------------------------------------------
   Change Activity:
                   2018/6/11 0011:
-------------------------------------------------
"""
__author__ = 'Administrator'
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from feature_extraction import *


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


data = pd.read_csv('data/data.csv')
log = pd.read_csv('data/log.csv')


# 添加特征
user_next_time_stat = user_next_time_stat(log)
user_tch_type_stat = user_tch_type_stat(log)
user_time_stat = user_time_stat(log)

user_next_time_stat.fillna(0, inplace=True)
user_tch_type_stat.fillna(0, inplace=True)
user_time_stat.fillna(0, inplace=True)

# scaler = MinMaxScaler()
# user_next_time_stat = scaler.fit_transform(user_next_time_stat)
# user_tch_type_stat = scaler.transform(user_tch_type_stat) #48875
# user_time_stat = scaler.transform(user_time_stat)
user_next_time_stat = normalize_cols(user_next_time_stat)
user_tch_type_stat = normalize_cols(user_tch_type_stat)
user_time_stat = normalize_cols(user_time_stat)


data = pd.merge(data,user_next_time_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_tch_type_stat,on=['USRID'],how='left',copy=False)
data = pd.merge(data,user_time_stat,on=['USRID'],how='left',copy=False)
data.fillna(0, inplace=True)

print(data.shape)

from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')
y = train.pop('FLAG') # train_y
col = train.columns
X = train[col].values # train_x

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

"""
https://blog.csdn.net/sun_shengyun/article/details/53811483
正则化选择参数：penalty
"""
clf = LogisticRegression(dual=True,tol=0.00001,C=1.0,class_weight='balanced',
                         random_state=27,solver='liblinear',max_iter=10000, verbose=0,n_jobs=1)

auc_list = []
for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    probas = clf.fit(X_train,y_train ).predict_proba(X_test)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, probas[:,1])
    print('auc value:', auc)
    auc_list.append(auc)

print('validate result:', np.mean(auc_list))

# 预测
pred = clf.predict_proba(test)

result_name = test[['USRID']]
result_name['RST'] = pred
result_name.to_csv('test_result.csv', index=None, sep='\t')



# lrW = LogisticRegression(class_weight ='auto')#针对样本不均衡问题，设置参数"class_weight

if __name__ == '__main__':
    print(train_userid)