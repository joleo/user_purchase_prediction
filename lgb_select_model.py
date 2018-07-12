# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lgb_select_model
   Description :
   Author :       Administrator
   date：          2018/6/11 0011
-------------------------------------------------
   Change Activity:
                   2018/6/11 0011:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
import numpy as np
import lightgbm as lgb

from feature_extraction import *
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from MLFeatureSelection import sequence_selection as ss
from feature_extraction import *

data = pd.read_csv('data/data.csv')
log = pd.read_csv('data/log.csv')
# 添加特征
user_next_time_stat = user_next_time_stat(log)
user_tch_type_stat = user_tch_type_stat(log)
user_time_stat = user_time_stat(log)

data = pd.merge(data, user_next_time_stat, on=['USRID'], how='left', copy=False)
data = pd.merge(data, user_tch_type_stat, on=['USRID'], how='left', copy=False)
data = pd.merge(data, user_time_stat, on=['USRID'], how='left', copy=False)
data.fillna(0, inplace=True)

print(data.shape)
train = data[data['FLAG']!=-1]
# test = data[data['FLAG']==-1]
def score1(pred, real): #评分系统
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    # print('auc value:', auc)
    return auc

def prepareData(): #读入你自己的数据集


    return train

def validate(train_x, train_y, features, clf, score):
    Performance = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    for train_index, test_index in skf.split(train_x, train_y):
        # print('Train: %s | test: %s' % (train_index, test_index))
        Testtemp = train_x[test_index]

        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        # clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=False,
        #         early_stopping_rounds=200)
        # prep = clf.predict_proba(X_test)[:, 1]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
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
                        lgb_train,  # 训练集
                        valid_sets=lgb_eval,  # 验证集
                        num_boost_round=40000,  # 迭代次数 40000 -> 10000
                        verbose_eval=250,  # 每隔250次，打印日志
                        early_stopping_rounds=50)
        print('Start predicting...')
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        Performance.append(score(y_pred, y_test))

    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance), clf


def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}
def main():
    sf = ss.Select(Sequence = True, Random = False, Cross = False) #初始化选择器，选择你需要的流程
    sf.ImportDF(prepareData(),label = 'buy') #导入数据集以及目标标签
    sf.ImportLossFunction(score1, direction = 'ascend') #导入评价函数以及优化方向
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['USRID','FLAG']) #初始化不能用的特征
    combine_col = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']#,
       # 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       # 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       # 'V30']
    sf.InitialFeatures(combine_col) #初始化其实特征组合
    sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
    sf.SetSample(1, samplemode = 1) #初始化抽样比例和随机过程
    sf.SetTimeLimit(100) #设置算法运行最长时间，以分钟为单位
    sf.clf = lgb.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000, max_depth=3, learning_rate = 0.2, n_jobs=8) #设定回归模型
    sf.SetLogFile('record.log') #初始化日志文件
    sf.run(validate) #输入检验函数并开始运行

if __name__ == "__main__":
    main()
    # train = prepareData()
    # print(train.head())