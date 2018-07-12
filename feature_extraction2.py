# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_extraction2
   Description :
   Author :       Administrator
   date：          2018/6/10 0010
-------------------------------------------------
   Change Activity:
                   2018/6/10 0010:
-------------------------------------------------
"""
__author__ = 'Administrator'
import numpy as np
import pandas as pd

# def user_tch_stat(data):
#     data.

def user_next_time_stat(data):

    time_data = data.loc[:, ['USRID','OCC_TIM_ST']]
    time_data['time_delta'] = time_data.groupby(['USRID'])['OCC_TIM_ST'].diff(-1).apply(np.abs) # 对时间求差分

    # 用户花的时间平均值、方差、最大最小值
    user_time_delta = time_data.groupby(['USRID'], as_index=False)['time_delta'].agg({
        'time_delta_mean': np.mean,
        'time_delta_std': np.std,
        'time_delta_min': np.min,
        'time_delta_max': np.max,
        'time_delta_size': np.size
    })
    user_time_delta['time_delta_diff'] = (user_time_delta['time_delta_max'] - user_time_delta['time_delta_min'])

    # 时间均值方差等信息
    time_day = data.groupby(['USRID'], as_index=False)['day'].agg({
        'day_mean': np.mean,
        'day_var': np.std,
        'day_min': np.min,
        'day_max': np.max,
        'day_size' : np.size
    })


    time_hour = data.groupby(['USRID'], as_index=False)['hour'].agg({
        'hour_mean': np.mean,
        'hour_var': np.std,
        'hour_min': np.min,
        'hour_max': np.max,
        'hour_size': np.size
    })
    time_merge = pd.merge(time_day, time_hour, on=['USRID'], how='left')
    # time_merge = pd.merge(time_merge, time_week, on=['USRID'], how='left')

    # 天-小时
    time_day_hour = data.groupby(['USRID', 'day'], as_index=False)['hour'].agg({
        'day_hour_mean': np.mean,
        'day_hour_std': np.var,
        'day_hour_min': np.min,
        'day_hour_max': np.max,
        'day_hour_size': np.size

    })

    day_hour_mean_kurt = time_day_hour.groupby('USRID')['day_hour_mean'].apply(lambda x: x.kurt()).reset_index()
    day_hour_mean_skew = time_day_hour.groupby(['USRID'])['day_hour_mean'].skew().reset_index()
    time_merge = pd.merge(time_merge, day_hour_mean_kurt, on=['USRID'], how='left')
    time_merge = pd.merge(time_merge, day_hour_mean_skew, on=['USRID'], how='left')

    time_merge = pd.merge(time_merge, time_day_hour, on=['USRID'], how='left')
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

    time_merge = pd.merge(time_merge, user_time_delta, on=['USRID'], how='left')

    # tch_type_num = data.groupby('USRID', as_index=False)['TCH_TYP'].count()
    # time_merge = pd.merge(time_merge, tch_type_num, on=['USRID'], how='left')

    return time_merge

import datetime

if __name__ == '__main__':
    log = pd.read_csv('data/log.csv')
    log['week'] = log['OCC_TIM'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())

    # time_day_hour = user_next_time_stat(log)
    log.to_csv('data/log.csv', header=True, index=False)
    # print(log.head())
