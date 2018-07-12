# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Smote
   Description :
   Author :       Administrator
   date：          2018/6/11 0011
-------------------------------------------------
   Change Activity:
                   2018/6/11 0011:
-------------------------------------------------
"""
__author__ = 'Administrator'

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn import pipeline as pl
from imblearn.metrics import (geometric_mean_score,
                              make_index_balanced_accuracy)
from imblearn.over_sampling import SMOTE, ADASYN

import pandas as pd
import numpy as np

data = pd.read_csv('data/data.csv')
data.fillna(0, inplace=True)

'''
['USRID', 'FLAG', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30']
'''
# print(data.columns)
print(data.head())
train = data[data['FLAG'] != -1]
y_train = train.pop('FLAG')
col = train.columns
X_train = train[col].values
# print(len(X_train)) # 80000
print(X_train)

oversampler = SMOTE(ratio= 0.1, random_state=3, k_neighbors=5)
os_X_train, os_y_train = oversampler.fit_sample(X_train,y_train)

print(len(os_X_train)) # 153648
print(os_X_train)

# res = pd.DataFrame(np.array(os_X_train), columns=)
# print(res)


