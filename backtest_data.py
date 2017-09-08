#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:07:18 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 回测用到的数据类
class backtest_data(data):
    """ This is the data class used for backtesting
    
    stock_price (pd.Panel): price data of stocks
    benchmark_price (pd.Panel): price data of benchmarks
    if_tradable (pd.Panel): information of whether a stock is enlisted/delisted or suspended from trading
    """
    
    def __init__(self):
        data.__init__(self)

    # 生成涨停跌停的数据矩阵, 在回测中不能买入已涨停的股票, 也不能卖出已跌停的股票
    # 之所以放在回测数据中, 是因为在策略中是无法知道选出的股票在调仓日时是否涨跌停, 且一般策略中不会用到这个数据
    # 如果策略中一定要用到这个数据, 则必须是shift后的数据, 即至少延迟了一天
    # 默认使用收盘价来判断这只股票今天是否已经涨停, 回测中, 尽管用vwap调仓, 但是此时已经知道了是否涨停
    # 注意这里并没有用未来数据, 因为尽管调仓时还不知道收盘价, 但是这里只是用收盘价来判断这只股票今天是否盘中涨跌停
    # 如果因为要使用其他的数据进行调仓, 从而要用其他的数据来进行判断, 则可自行传入
    def generate_if_buyable_sellable(self, *, file_name=['PrevClosePrice', 'ClosePrice'],
                                     item_name=['PrevClosePrice', 'ClosePrice']):
        # 读取入收盘价数据
        price_data = data.read_data(file_name, item_name)
        ClosePrice = price_data['ClosePrice']
        PrevClosePrice = price_data['PrevClosePrice']

        is_enlisted = self.if_tradable['is_enlisted']
        is_enlisted_lag = is_enlisted.shift(1)
        new_stocks = np.logical_and(is_enlisted==1, is_enlisted_lag==0)

        # 根据PrevClosePrice计算涨停价和跌停价
        CapPrice = PrevClosePrice.mul(1.1).applymap(data.round)
        BottomPrice = PrevClosePrice.mul(0.9).applymap(data.round)
        # 首先生成全是True的矩阵
        all_true = pd.DataFrame(True, index=self.if_tradable.major_axis, columns=self.if_tradable.minor_axis)
        # 根据这个价格来判断涨跌停, 注意还需要加入是否是新股的判断条件
        # 因为新股上市第一天没有涨跌停限制
        self.if_tradable['if_cap'] = all_true.where(np.logical_and(ClosePrice>=CapPrice,
                                        np.logical_not(new_stocks)), False)
        self.if_tradable['if_bottom'] = all_true.where(np.logical_and(ClosePrice<=BottomPrice,
                                            np.logical_not(new_stocks)), False)

        # # 取消涨跌停限制
        # self.if_tradable['if_cap'] = np.logical_not(all_true)
        # self.if_tradable['if_bottom'] = np.logical_not(all_true)
        # # 取消停牌限制
        # self.if_tradable['if_tradable'] = all_true


        # 生成是否可以买入或卖出的矩阵
        # 首先判断是否已经有了不可交易矩阵, 如果没有, 则报错
        assert 'if_tradable' in self.if_tradable.items, 'Please generate if_tradable before generating ' \
            'if buyable&sellable! \n'
        # 可交易, 且没有涨停, 才是可买入
        self.if_tradable['if_buyable'] = np.logical_and(self.if_tradable['if_tradable'],
                                                        np.logical_not(self.if_tradable['if_cap']))

        # 可交易, 且没有跌停, 才是可卖出
        self.if_tradable['if_sellable'] = np.logical_and(self.if_tradable['if_tradable'],
                                                         np.logical_not(self.if_tradable['if_bottom']))
