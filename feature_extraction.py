# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_extraction
   Description :
   Author :       Administrator
   date：          2018/6/8 0008
-------------------------------------------------
   Change Activity:
                   2018/6/8 0008:
-------------------------------------------------
"""
__author__ = 'Administrator'
"""

类别型特征：
V2、V3、V4、V5、V16、V22、V26
V5和V22 正相关，0.53

数值型特征：
V9与V10、V12、V13四者之间存在强正相关，与V11、V15中等强度正相关
V11与V9,V10,V12,V13,V15,V17,V18中等强度正相关，与V20,V23存在负相关

"""
import time
import numpy as np
import pandas as pd
# log特征
def user_v_stat(data):
    user_v2_stat = data.groupby(['USRID','V2'])['EVT_LBL_1'].agg({"v2_evt_num": "count"}).reset_index()
    v2_evt_num = pd.pivot_table(user_v2_stat, index='USRID', columns='V2', values='v2_evt_num',
                                      fill_value=0).reset_index()
    v2_evt_num.rename(columns={-0.90689: 'v2_1_evt_num', 1.10266: 'v2_2_evt_num'}, inplace=True)
    return v2_evt_num

def user_v4_stat(data):
    user_v4_stat = data.groupby(['USRID','V4'])['EVT_LBL_1'].agg({"v4_evt_num": "count"}).reset_index()
    v4_evt_num = pd.pivot_table(user_v4_stat, index='USRID', columns='V4', values='v4_evt_num',
                                      fill_value=0).reset_index()
    # v4_evt_num.rename(columns={-0.90689: 'v4_1_evt_num', 1.10266: 'v4_2_evt_num'}, inplace=True)
    return v4_evt_num


def user_v5_stat(data):
    user_v5_stat = data.groupby(['USRID','V5'])['EVT_LBL_1'].agg({"v5_evt_num": "count"}).reset_index()
    v5_evt_num = pd.pivot_table(user_v5_stat, index='USRID', columns='V5', values='v5_evt_num',
                                      fill_value=0).reset_index()
    # v5_evt_num.rename(columns={-0.90689: 'v5_1_evt_num', 1.10266: 'v5_2_evt_num'}, inplace=True)
    return v5_evt_num

def user_next_time_stat(data):

    # 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
    # 这个部分可以发掘的特征其实很多很多很多很多
    time_data = data.loc[:, ['USRID','OCC_TIM_ST']]
    # time_data['OCC_TIM'] = time_data['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    # time_data = time_data.sort_values(['USRID', 'OCC_TIM'])
    time_data['next_time'] = time_data.groupby(['USRID'])['OCC_TIM_ST'].diff(-1).apply(np.abs) # 对时间求差分

    # 用户花的时间平均值、方差、最大最小值
    next_time_stat = time_data.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })

    return next_time_stat

def user_tch_type_stat(data):

    EVT_LBL_set_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg(
        {'EVT_LBL_set_len': lambda x: len(set(x))})

    # 按照TCH_TYP分组计算EVT_LBL频数
    tch_typ_num = data.groupby(['USRID', 'TCH_TYP'])['EVT_LBL'].agg({"tch_typ_num": "count"}).reset_index()
    tch_typ_num_stat = pd.pivot_table(tch_typ_num, index='USRID', columns='TCH_TYP', values='tch_typ_num',
                                      fill_value=0).reset_index()
    tch_typ_num_stat.rename(columns={0: 'tch_typ_0_num', 2: 'tch_typ_2_num'}, inplace=True)

    # tch_typ_num_stat['evt_diff_rate'] = EVT_LBL_set_len['EVT_LBL_set_len'] / tch_typ_num_stat['tch_typ_0_num']

    user_tch_type_stat = pd.merge(EVT_LBL_set_len, tch_typ_num_stat, on=['USRID'], how='left')


    # EVT_LBL_1 频数
    EVT_LBL_1_num = data.groupby(['USRID', 'EVT_LBL_1'])['EVT_LBL'].agg({"evt_lbl_1_num": "count"}).reset_index()
    evt_lbl_1_stat = pd.pivot_table(EVT_LBL_1_num, index='USRID', columns='EVT_LBL_1', values='evt_lbl_1_num',fill_value=0).reset_index()
    evt_lbl_1_stat.rename(columns={0 : 'evt_lbl_1_0',10 : 'evt_lbl_1_10',102 : 'evt_lbl_1_102',139 : 'evt_lbl_1_139',162 : 'evt_lbl_1_162',
                                    163 : 'evt_lbl_1_163',181 : 'evt_lbl_1_181',257 : 'evt_lbl_1_257',259 : 'evt_lbl_1_259',326 : 'evt_lbl_1_326',
                                    359 : 'evt_lbl_1_359',372 :'evt_lbl_1_372',38 : 'evt_lbl_1_38',396 : 'evt_lbl_1_396',438 : 'evt_lbl_1_438',
                                    460 : 'evt_lbl_1_460',508 : 'evt_lbl_1_508',518 : 'evt_lbl_1_518',520 : 'evt_lbl_1_520',540 : 'evt_lbl_1_540',
                                    604 : 'evt_lbl_1_604'}, inplace=True)

    user_tch_type_stat_1 = EVT_LBL_1_num.groupby(['USRID'], as_index=False)['evt_lbl_1_num'].agg({
        'evt_lbl_1_num_mean': np.mean,
        'evt_lbl_1_num_std': np.std,
        # 'evt_lbl_1_num_min': np.min,
        # 'evt_lbl_1_num_max': np.max
    })
    user_tch_type_stat = pd.merge(user_tch_type_stat, evt_lbl_1_stat, on=['USRID'], how='left')
    user_tch_type_stat = pd.merge(user_tch_type_stat, user_tch_type_stat_1, on=['USRID'], how='left')

    # 统计各类别在此次出现前的count数



    # 微降
    # EVT_LBL_1_len = len(set(data['EVT_LBL_1']))
    # # # EVT_LBL_1_num['evt_lbl_1_frep'] = EVT_LBL_1_num['evt_lbl_1_num'] / EVT_LBL_1_len
    # EVT_LBL_1_num['evt_lbl_1_idf'] = np.log2( EVT_LBL_1_len / EVT_LBL_1_num['evt_lbl_1_num']+1 )
    # # # EVT_LBL_1_num['evt_lbl_1_tf_idf'] = EVT_LBL_1_num['evt_lbl_1_frep'] * EVT_LBL_1_num['evt_lbl_1_idf']
    # user_evt_lbl_1_tf_idf = pd.pivot_table(EVT_LBL_1_num, index='USRID', columns='EVT_LBL_1', values='evt_lbl_1_idf',fill_value=0).reset_index()
    # drop_col = [508, 181, 10, 540, 460, 162, 438, 396, 372, 102, 139, 163]
    # user_evt_lbl_1_tf_idf.drop(drop_col, axis=1, inplace=True )

    return user_tch_type_stat



def user_time_stat(data):
    # 时间频数分布
    # 统计天
    evt_lbl_1_day_num = data.groupby(['USRID', 'day'])['EVT_LBL_1'].agg({"evt_lbl_1_day_num": "count"}).reset_index()

    user_evt_lbl_1_day_num = pd.pivot_table(evt_lbl_1_day_num, index='USRID', columns='day', values='evt_lbl_1_day_num',
                                      fill_value=0).reset_index()
    user_evt_lbl_1_day_num.rename(columns={1 : 'day_1',2 : 'day_2',3 : 'day_3',4 : 'day_4',5 : 'day_5',6 : 'day_6'
                                           ,7 : 'day_7',8 : 'day_8',9 : 'day_9',10 : 'day_10',11 : 'day_11'
                                            ,12 : 'day_12',13 : 'day_13',14 : 'day_14',15 : 'day_15',16 : 'day_16'
                                            ,17 : 'day_17',18 : 'day_18',19 : 'day_19',20 : 'day_20',21 : 'day_21'
                                            ,22 : 'day_22',23 : 'day_23',24 : 'day_24',25 : 'day_25',26 : 'day_26'
                                            ,27 : 'day_27',28: 'day_28',29 : 'day_29',30 : 'day_30',31 : 'day_31'}, inplace=True)

    # evt_lbl_1_day_num_stat= evt_lbl_1_day_num.groupby(['USRID'], as_index=False)['evt_lbl_1_day_num'].agg({
    #     'evt_lbl_1_day_num_mean': np.mean,
    #     'evt_lbl_1_day_num_std': np.std,
        # 'evt_lbl_1_day_num_min': np.min,
        # 'evt_lbl_1_day_num_max': np.max
    # })
    # user_evt_lbl_1_day_kurt = evt_lbl_1_day_num.groupby('USRID')['evt_lbl_1_day_num'].apply(lambda x: x.kurt()).reset_index()
    # user_evt_lbl_1_day_skew = evt_lbl_1_day_num.groupby(['USRID'])['evt_lbl_1_day_num'].skew().reset_index()
    #
    # user_time_tch_type_stat = pd.merge(user_evt_lbl_1_day_num, evt_lbl_1_day_num_stat, on=['USRID'], how='left')
    # # SKEW | kurt
    # user_time_tch_type_stat = pd.merge(user_time_tch_type_stat, user_evt_lbl_1_day_kurt, on=['USRID'], how='left')
    # user_time_tch_type_stat = pd.merge(user_time_tch_type_stat, user_evt_lbl_1_day_skew, on=['USRID'], how='left')

    # 统计小时
    evt_lbl_1_hour_num = data.groupby(['USRID', 'hour'])['EVT_LBL_1'].agg({"evt_lbl_1_hour_num": "count"}).reset_index()
    user_evt_lbl_1_hour_num = pd.pivot_table(evt_lbl_1_hour_num, index='USRID', columns='hour', values='evt_lbl_1_hour_num',
                                      fill_value=0).reset_index()

    user_evt_lbl_1_hour_num.rename(columns={1: 'hour_1', 2: 'hour_2', 3: 'hour_3', 4: 'hour_4', 5: 'hour_5'
                                             , 6: 'hour_6', 7: 'hour_7', 8: 'hour_8', 9: 'hour_9', 10: 'hour_10', 11: 'hour_11', 12: 'hour_12',
                                            13: 'hour_13', 14: 'hour_14', 15: 'hour_15', 16: 'hour_16', 17: 'hour_17',
                                            18: 'hour_18', 19: 'hour_19', 20: 'hour_20', 21: 'hour_21', 22: 'hour_22',
                                            23: 'hour_23', 0: 'hour_0'}, inplace=True)

    # evt_lbl_1_hour_num_stat = evt_lbl_1_hour_num.groupby(['USRID'], as_index=False)['evt_lbl_1_hour_num'].agg({
    #     'evt_lbl_1_hour_num_mean': np.mean,
    #     'evt_lbl_1_hour_num_std': np.std,
    #     'evt_lbl_1_hour_num_min': np.min,
    #     'evt_lbl_1_hour_num_max': np.max
    # })
    user_time_tch_type_stat = pd.merge(user_evt_lbl_1_day_num, user_evt_lbl_1_hour_num, on=['USRID'], how='left')
    # user_time_tch_type_stat = pd.merge(user_time_tch_type_stat, evt_lbl_1_hour_num_stat, on=['USRID'], how='left')

    return user_time_tch_type_stat

# from fuzzywuzzy import fuzz
# def user_fuzz_stat(data):
#     data['EVT_LBL'].apply(lambda x:fuzz.ratio() )
#     pass

def user_other(data):
    # 每个事件时长（频数降分、时间升分）
    EVT_LBL_set_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg(
        {'EVT_LBL_set_len': lambda x: len(set(x))})
    data['time_diff_data'] = data.groupby(['USRID','day'])['OCC_TIM_ST'].diff(1)# 对时间求差分
    time_diff_sum = data.groupby(['USRID'], as_index=False)['time_diff_data'].agg({'time_diff_sum' : np.sum })
    time_diff_sum['time_diff_evt_lvl_rate'] = time_diff_sum['time_diff_sum'] / EVT_LBL_set_len['EVT_LBL_set_len']

    # # 用户打开app次数(频数降分、时间升分)
    data['access_num'] = data['EVT_LBL'].astype('str').map(lambda x: 1 if x == '38-115-117' else 0)
    user_access_num = data.groupby('USRID', as_index=False)['access_num'].sum()
    user_access_num.loc[:, 'user_access_rate'] = 1 / (user_access_num['access_num'] + 1)
    user_access_num.drop('access_num', axis=1, inplace=True )

    user_tch_type_stat = pd.merge(user_access_num, time_diff_sum, on=['USRID'], how='left')

    return user_tch_type_stat

def user_time_stat2(data):

    # 时间均值方差等信息
    time_day = data.groupby(['USRID'], as_index=False)['day'].agg({
        'day_mean': np.mean,
        'day_var': np.std,
        'day_min': np.min,
        'day_max': np.max
    })
    time_hour = data.groupby(['USRID'], as_index=False)['hour'].agg({
        'hour_mean': np.mean,
        'hour_var': np.std,
        'hour_min': np.min,
        'hour_max': np.max
    })
    time_merge = pd.merge(time_day, time_hour, on=['USRID'], how='left')
    time_minute = data.groupby(['USRID'], as_index=False)['minute'].agg({
        'minute_mean': np.mean,
        'minute_var': np.std,
        'minute_min': np.min,
        'minute_max': np.max
    })
    time_merge = pd.merge(time_merge, time_minute, on=['USRID'], how='left')
    time_second = data.groupby(['USRID'], as_index=False)['second'].agg({
        'second_mean': np.mean,
        'second_var': np.std,
        'second_min': np.min,
        'second_max': np.max
    })
    time_merge = pd.merge(time_merge, time_second, on=['USRID'], how='left')

    # 天-小时
    time_day_hour = data.groupby(['USRID','day'], as_index=False)['hour'].agg({
        'day_hour_mean': np.mean,
        'day_hour_std': np.var,
        'day_hour_min': np.min,
        'day_hour_max': np.max
    })
    user_day_hour_mean = pd.pivot_table(time_day_hour, index='USRID', columns='day',
                                             values='day_hour_mean', fill_value=0).reset_index()
    time_merge = pd.merge(time_merge, user_day_hour_mean, on=['USRID'], how='left')
    user_day_hour_std = pd.pivot_table(time_day_hour, index='USRID', columns='day',
                                        values='day_hour_std', fill_value=0).reset_index()
    time_merge = pd.merge(time_merge, user_day_hour_std, on=['USRID'], how='left')
    user_day_hour_min = pd.pivot_table(time_day_hour, index='USRID', columns='day',
                                        values='day_hour_min', fill_value=0).reset_index()
    time_merge = pd.merge(time_merge, user_day_hour_min, on=['USRID'], how='left')
    user_day_hour_max = pd.pivot_table(time_day_hour, index='USRID', columns='day',
                                        values='day_hour_max', fill_value=0).reset_index()
    time_merge = pd.merge(time_merge, user_day_hour_max, on=['USRID'], how='left')

    return time_merge




if __name__ == '__main__':
    train = pd.read_csv('data/data.csv')
    log = pd.read_csv('data/log.csv')

    # oversample
    pos_train = train[train['FLAG'] == 1]
    neg_train = train[train['FLAG'] == 0]

    p = 0.04
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print(len(pos_train) / (len(pos_train) + len(neg_train)))

    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train
    data = pd.merge(x_train,log , on=['USRID'], how='left', copy=False)

    print(data.head())
    print(data.shape)
    print(data.FLAG.value_counts())

    # print(x_train[['USRID', 'FLAG']].head(50).T)

