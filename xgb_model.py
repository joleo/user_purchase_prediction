# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_intergrate
   Description :
   Author :       Administrator
   date：          2018/6/8 0008
-------------------------------------------------
   Change Activity:
                   2018/6/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import sys
import time
import scipy as sp
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from feature_extraction import *
OFF_LINE = False


def xgb_model(train_set_x, train_set_y, test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,  # 03.8
              'subsample': 0.7, #0.7500000000000001
              'min_child_weight': 9,  # 2 3
              'scale_pos_weight': 0.85,
              'silent': 1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict


if __name__ == '__main__':

    time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    data = pd.read_csv('data/data.csv')
    log = pd.read_csv('data/log.csv')

    # oversample
    # pos_train = data[data['FLAG'] == 1]
    # neg_train = data[data['FLAG'] == 0]
    # test_data = data[data['FLAG'] == -1]
    #
    # p = 0.1 #0.165
    # scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    # while scale > 1:
    #     neg_train = pd.concat([neg_train, neg_train])
    #     scale -= 1
    # neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    # print(len(pos_train) / (len(pos_train) + len(neg_train)))
    #
    # data = pd.concat([pos_train, neg_train, test_data])
    # # y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    # del pos_train, neg_train


    # 添加特征
    next_time_stat = user_next_time_stat(log)
    user_tch_type_stat = user_tch_type_stat(log)
    user_time_stat = user_time_stat(log)

    data = pd.merge(data, next_time_stat, on=['USRID'], how='left', copy=False)
    data = pd.merge(data, user_tch_type_stat, on=['USRID'], how='left', copy=False)
    data = pd.merge(data, user_time_stat, on=['USRID'], how='left', copy=False)
    data.fillna(0, inplace=True)

    data['V9_10'] = data['V9'] + data['V10']
    data['V19_18'] = data['V19'] + data['V18']  # 微微增
    data['V21_23'] = data['V23'] + data['V21'] + data['V27']

    V19_18_col = ['V9', 'V10', 'V12', 'V13', 'V18', 'V19', 'V21', 'V23', 'V27']
    data.drop(V19_18_col, axis=1, inplace=True)
    print(data.shape)

    # 划分数据集
    all_train = data[data['FLAG'] != -1]
    test_set = data[data['FLAG'] == -1]

    train_userid = all_train.pop('USRID')
    train_y = all_train.pop('FLAG')  # train_y
    col = all_train.columns
    train_x = all_train[col].values  # train_x
    auc_list = []

    # 分层抽样
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    for train_index, test_index in skf.split(train_x, train_y):
        # print('Train: %s | test: %s' % (train_index, test_index))
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        pred_value = xgb_model(X_train, y_train, X_test)
        # print(pred_value)
        # print(y_test)

        pred_value = np.array(pred_value)
        pred_value = [ele + 1 for ele in pred_value]

        y_test = np.array(y_test)
        y_test = [ele + 1 for ele in y_test]

        fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)

        auc = metrics.auc(fpr, tpr)
        print('auc value:', auc)
        auc_list.append(auc)

    print('validate result:', np.mean(auc_list))

    # 测试集
    test_userid = test_set.pop('USRID')
    test_y = test_set.pop('FLAG')
    col = test_set.columns
    test_x = test_set[col].values

    # 预测
    pred_result = xgb_model(train_x, train_y, test_x)

    # 特征重要度
    # import operator
    # importance = model.get_fscore()
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df['fscore'] = df['fscore'] / df['fscore'].sum()
    # # print(list(df['feature']), list(df['fscore']))
    # df.to_csv('data/submit/%s_feature_importance.csv'%str(time_date), index=False)


    res = pd.DataFrame()
    res['USRID'] = list(test_userid.values)
    res['RST'] = pred_result
    res.to_csv('data/submit/xgb_%s.csv'%str(time_date), index=False, sep='\t')