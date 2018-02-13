#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:39:23 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from strategy_data import strategy_data
from position import position

# 策略类，根据各种策略选取股票的类，这是一个基本类

class strategy(object):
    """ This is the base class of strategy classes, which does nothing but served as a base.
    
    foo
    """
    def __init__(self):
        # 策略中会用到的数据类
        self.strategy_data = strategy_data()
        # 策略中用到的持仓类，策略最终就是为了选择这个持仓，在没有调仓日和股票代码的情况下，暂设为一个空DataFrame
        self.position = position()
        # 调仓日，注意调仓日与策略数据中的日期一般不同，暂设为一个空series
        self.holding_days = pd.Series()
        # 策略的股票池
        self.strategy_data.stock_pool = 'all'
        
    # 初始化持仓类的函数
    def initialize_position(self, standard_data):
        self.position = position(standard_data)
    
    # 将持仓矩阵归零的函数，（当要重复测试策略，不想初始化，直接改变选股条件时，会用到）
    def reset_position(self):
        self.position = position(self.position.holding_matrix)

    # 对策略的原始数据进行描述，通过对原始数据的分析，得到启发
    # 这里为一个空函数
    def data_description(self):
        pass
        
    # 选股函数，这里的选股函数为空
    def select_stocks(self, *, select_ratio=[0.8, 1], direction='+', weight=0, use_factor_expo=True,
                      expo_weight=1):
        pass
    
    # 生成调仓日，这里的默认生成函数为直接从目录中读取
    def generate_holding_days(self, *, holding_freq='w', start_date='default', end_date='default', loc=-1):
        self.holding_days = pd.read_csv(str(os.path.abspath('.'))+'/holding_days.csv', 
                                        parse_dates = [0], squeeze = True)
        print('Please note that, as the default method, '
              'the holding days has been read from the holding_days.csv file in current directory\n')
        
    # 在持仓矩阵中过滤掉那些不能交易的股票
    def filter_untradable(self):
        tradable_data = self.strategy_data.if_tradable.ix['if_tradable', self.position.holding_matrix.index, :]
        # 凡是有持仓的，但不能交易的，全部设为0持仓，注意此函数并未进行重新归一化
        self.position.holding_matrix *= tradable_data

    # 在持仓矩阵中过滤掉那些不能投资的股票
    def filter_uninv(self):
        inv_data = self.strategy_data.if_tradable.ix['if_inv', self.position.holding_matrix.index, :]
        # 凡是有持仓的，但不能投资的，全部设为0持仓，注意此函数并未进行重新归一化
        self.position.holding_matrix *= inv_data

    # 根据一个传入的时间序列数据，根据频率生成调仓周期的函数
    # 传入的时间序列可以是series，也可以是dataframe
    @staticmethod
    def resample_tradingdays(time_series, *, freq='m', loc=-1):
        # 所取调仓日为每个调仓周期的第loc天, 注意调仓时间是调仓日的早上,
        # 即调仓日当天早上调仓时,拥有上一个周期的所有数据
        # 首先定义要apply的函数, 函数在每个调仓周期之间进行处理, 返回需要的调仓日
        def holdingday_func(ts, *, loc):
            # 如果loc为非负数, 则要求size必须大于等于loc+1
            if loc >= 0:
                if ts.size >= loc+1:
                    return ts.index[loc]
                else:
                    return np.nan
            # 如果loc为负数, 因为负数是按照-1开始数的, 因此size必须大于loc的绝对值
            else:
                if ts.size >= np.abs(loc):
                    return ts.index[loc]
                else:
                    return np.nan
        resampled_tds = time_series.resample(freq).apply(holdingday_func, loc=loc).dropna()
        # 将resampled_tds改为一个索引和值都是做好的交易日的series
        resampled_tds = pd.Series(resampled_tds.values, index=resampled_tds.values)
        return resampled_tds

    # 根据策略, 在类的内部自己构造因子的函数, 即不从外部读取因子, 而是内部构造
    # 根据不同的策略构造不同的因子, 因此这里的函数什么也不做
    def construct_factor(self):
        pass



                                          







































































































































































































































































































