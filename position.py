#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:29:49 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

# 储存持仓的持仓类

class position(object):
    """ This is the class of holding matrix.
    
    holding_matrix (pd.DataFrame) : the holding_matrix of this position
    cash(pd.Series): the cash position of the whole position
    """
    
    def __init__(self, standard=None):
        # 现金资产比例, 为一个series, 现金资产代表了持仓里的杠杆, 现金资产的比例越高, 则杠杆月越低
        # 现在还不支持期货资产, 但是可以通过现金资产来模拟由于持有期货资产带来的保证金的影响
        # 只要根据期货资产的保证金比例, 相应的转化成现金资产的比例,
        # 就可以做到模拟持有期货资产保证金带来降杠杆的情况
        if standard is None:
            self.holding_matrix = pd.DataFrame()
            self.cash = pd.Series()
        else:
            self.holding_matrix = pd.DataFrame(0.0, index=standard.index, columns=standard.columns)
            self.cash = pd.Series(0.0, index=standard.index)

    # 根据某一指标，对持仓进行加权，如对市值进行加权
    def weighted_holding(self, weights):
        """ Get the weighted holding matrix
        
        foo
        """
        self.holding_matrix = self.holding_matrix.mul(weights, fill_value = 0)
        self.to_percentage()
        pass

    # 根据行业标签，进行分行业加权，可以选择行业内如何加权，以及行业间如何加权
    def weighted_holding_indus(self, industry, *, inner_weights=0, outter_weights=0):
        # 定义行业内加权的函数
        def inner_weighting(grouped_data):
            new_holding = grouped_data['holding'].mul(grouped_data['inner_weights'], fill_value=0)
            if not (new_holding==0).all():
                new_holding = new_holding.div(new_holding.sum())
            return new_holding
        # 如果行业内权重为0，则为行业内等权
        if type(inner_weights) == int and inner_weights == 0:
            inner_weights = pd.DataFrame(1, index=self.holding_matrix.index, columns=self.holding_matrix.columns)
        # 根据持仓日循环
        for time, curr_holding in self.holding_matrix.iterrows():
            curr_data = pd.DataFrame({'holding':curr_holding,
                                      'inner_weights':inner_weights.ix[time]})
            # 处理行业数据最后一天可能全是nan的特殊情况（易在取数据时出现）
            if industry.ix[time].isnull().all():
                continue
            grouped = curr_data.groupby(industry.ix[time])
            # 进行行业内加权
            after_inner = grouped.apply(inner_weighting).reset_index(level=0, drop=True)
            # 对行业间加权数据进行求和
            # 如果行业间权重为0，则为行业间等权，即每个行业的总权重为1，（注意不是每个股票的权重为1）
            if type(outter_weights) == int and outter_weights == 0:
                outter_weights_sum = pd.Series(1, index=after_inner.index)
            else:
                outter_weights_sum = outter_weights.ix[time].groupby(industry.ix[time]).transform('sum')
            after_outter = after_inner.mul(outter_weights_sum).fillna(0)
            self.holding_matrix.ix[time] = after_outter
        self.holding_matrix = self.holding_matrix.fillna(0)
        self.to_percentage()
        pass

    # 定义归一化的要apply的函数
    # infinitesimal为在做空情况下，多空组合的和可能非常接近于0，当多空组合的和小于这个值的时候，分多空的方法归一
    @staticmethod
    def to_percentage_func(input_series, *, infinitesimal=1e-4):
        # 注意如果一期持仓全是0，则不改动
        if (input_series == 0).all():
            return input_series
        # 当组合的和小于设定的无穷小量时，采用多空分别归一的方法归一
        elif input_series.sum() < infinitesimal:
            positive_part = input_series[input_series > 0]
            positive_part = positive_part.div(positive_part.sum())
            negative_part = input_series[input_series < 0]
            negative_part = negative_part.div(np.abs(negative_part.sum()))
            input_series[positive_part.index] = positive_part
            input_series[negative_part.index] = negative_part
            return input_series
        # 一般情况下，直接归一
        else:
            input_series = input_series.div(input_series.sum())
            return input_series

    # 将持仓归一化，成为加总为1的百分比数
    def to_percentage(self):
        # apply函数
        self.holding_matrix = self.holding_matrix.apply(position.to_percentage_func, axis=1)
        # 防止无持仓的变成nan
        self.holding_matrix = self.holding_matrix.fillna(0.0)
        
    # 添加持股的函数，即，将选出的股票加入到对应时间点的持仓中去
    def add_holding(self, time, to_be_added):
        """ Add the holding matrix with newly selected(or bought) stocks.
        
        foo
        """
        self.holding_matrix.ix[time] = self.holding_matrix.ix[time].add(to_be_added, fill_value = 0)
     
    # 减少持股的函数，即，将选出减持的股票从对应的时间点持仓中减去
    def subtract_holding(self, time, to_be_subtracted):
        """ Subtract the holding matrix with newly subtracted(or sold) stocks.
        
        foo
        """
        self.holding_matrix.ix[time] = self.holding_matrix.ix[time].sub(to_be_subtracted, fill_value = 0)

    # 添加现金资产的函数, 单位自定义
    def add_cash(self, time, to_be_addded):
        self.cash.ix[time] += to_be_addded

    # 减少现金资产的函数, 单位自定义
    def subtract_cash(self, time, to_be_subtracted):
        self.cash.ix[time] -= to_be_subtracted































             
        
        