#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:00:02 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 基本数据类
class data(object):
    """ This is the base class of a group of data classes, which is mostly used.
    
    stock_price (pd.Panel): price data of stocks, note the difference between stock_price
                         data and raw_data
    raw_data (pd.Panel): original data get from market or financial report, or intermediate data
                         which is used for factor calculation, note the difference between stock_price
                         data and raw_data
    benchmark_price (pd.Panel): price data of benchmarks
    if_tradable (pd.Panel): marks which indicate if stocks are enlisted, delisted, suspended, tradable, in stock pool
                            or investable.
    const_data (pd.DataFrame): const data, usually macroeconomic data, such as risk free rate or inflation rate.
    """
    
    def __init__(self):
        self.stock_price = pd.Panel()
        self.raw_data = pd.Panel()
        self.benchmark_price = pd.Panel()
        self.if_tradable = pd.Panel()
        self.const_data = pd.DataFrame()

    # 读取数据的函数
    @staticmethod
    def read_data(file_name, *, item_name=None, folder_name='ResearchData/', shift=False):
        """ Get the data from file.
        
        file_name: name of the file.
        item_name: name of the data in the panel.
        shift: denote that if the data read need to be shifted by 1. This is because, the strategy data we got
        will have one lag(i.e. on the start of day 2, you can not know data on day 2 but day 1 or before day 2,
        thus the decision you make on the start of day 2 is based on data before day 2.), while for backtest
        data, we don't need this lag. The default option will not condunct the lag, you can set shift to True
        to create the lag for strategy data.)
        """
        # 从文件中读取数据, 如果file_name是一个string, 则不进行操作, 如果file_name是list或tuple
        # 则把每个文件根据item_name, 做成一个panel返回
        if isinstance(file_name, str):
            obj = pd.read_hdf(str(os.path.abspath('.')) + '/' + folder_name + file_name, '123')
            if shift:
                # 由于单个读取有可能obj是panel, panel.shift会损失major_axis上被shift掉的索引, 因此要多做一步
                if isinstance(obj, pd.Panel):
                    obj = obj.shift(1).reindex(major_axis=obj.major_axis)
                else:
                    obj = obj.shift(1)
        else:
            obj = {}
            for i, s in enumerate(file_name):
                temp_df = pd.read_hdf(str(os.path.abspath('.')) + '/' + folder_name + s, '123')
                # 批量读取必须是dataframe
                if not isinstance(temp_df, pd.DataFrame):
                    raise IOError('When more than 2 files are read, each file must be dataframe!\n')
                if shift:
                    temp_df = temp_df.shift(1)
                if item_name is None:
                    obj[file_name[i]] = temp_df
                else:
                    obj[item_name[i]] = temp_df
            obj = pd.Panel.from_dict(obj)

        return obj

    # 写数据的函数
    @staticmethod
    def write_data(written_data, *, file_name=None, folder_name='ResearchData/', separate=False):
        """ Write the data to csv file

        :param written_data: data to be written to file
        :param file_name: (list) list of strings containing names of csv files, note it has to be the same length of
        items in written_data, if it sets to default, the file name will be the name of items of the written data
        """
        # 是否要分开储存, 分开储存会把一个panel存成很多dataframe
        if separate:
            if file_name is None:
                for cursor, item_name in enumerate(written_data.items):
                    written_data.ix[cursor].to_hdf(str(os.path.abspath('.')) + '/' + folder_name +
                                                   str(item_name), '123')
            else:
                for cursor, item_name in enumerate(written_data.items):
                    written_data.ix[cursor].to_hdf(str(os.path.abspath('.')) + '/' + folder_name +
                                                   file_name[cursor], '123')
        else:
            if file_name is None:
                raise ValueError('Please specify a file name for your data!\n')
            written_data.to_hdf(str(os.path.abspath('.')) + '/' + folder_name + file_name, '123')
        
    # 重新对齐索引的函数
    @staticmethod
    def align_index(standard, raw_data, *, axis='both'):
        """Align the index of second data to first data.
        
        standard (pd.DataFrame): data of standard index
        raw_data (pd.Panel): data to be aligned
        """
        if axis is 'both':
            aligned_data = raw_data.reindex(major_axis=standard.index, minor_axis=standard.columns)
        elif axis is 'major':
            aligned_data = raw_data.reindex(major_axis=standard.index)
        elif axis is 'minor':
            aligned_data = raw_data.reindex(minor_axis=standard.columns)

        return aligned_data
    
    # 读取上市、退市、停牌数据，并生成可否交易的矩阵
    def generate_if_tradable(self, *, file_name=['is_enlisted','is_delisted','is_suspended'],
                             item_name=['is_enlisted','is_delisted','is_suspended'],
                             shift=False):
        if 'is_enlisted' not in self.if_tradable.items or 'is_delisted' not in self.if_tradable.items or \
                'is_suspended' not in self.if_tradable.items:
            # 读取上市、退市、停牌数据
            self.if_tradable = data.read_data(file_name, item_name=item_name, shift = shift)
        # 填数据中的nan要分别处理, 对于is_enlisted和is_delisted, 把nan填成0. 即未上市的股票, is_delisted会变成
        # False, 这与正常逻辑相符. 对于is_suspended, 要将nan填成1, 即默认没有数据的股票是停牌的. 这会导致: 1.
        # 未上市或已退市的股票会被标记成停牌, 这没有问题; 2. 没有停牌的股票, 一定在SmartQuant中有行情数据,
        # 行情数据中会标记是否停牌, 停牌的数据不会是nan, 因此也没有问题; 3. 这样填主要是为了应对暂停上市的股票,
        # 这些暂停上市的股票, 在暂停上市的初期, SmartQuant中会有行情数据, 并把它们标记成停牌, 但是若暂停时间过长
        # SmartQuant中会没有这些股票的行情数据(根本原因是聚源数据库里会没有这些股票的行情数据), 即出现在这里的nan中,
        # 如果把nan填成0, 则它们变成可以交易的了, 就会出错, 因为实际上它们是不可交易的,
        # 至于为何不在退市标记中处理暂停上市标记, 是因为暂停上市的股票是被套牢的, 而退市的股票是可以被清算交易的,
        # 因此, 对于投资者来暂停上市和停牌的性质更相同, 将其处理成停牌即可. 这一点事实上也与在database中
        # 处理复权因子和价格数据时的思路一样, 因为在那里为nan的is_suspended数据会被bool转换成True, 即停牌
        self.if_tradable.ix['is_enlisted'].fillna(0, inplace=True)
        self.if_tradable.ix['is_delisted'].fillna(0, inplace=True)
        self.if_tradable.ix['is_suspended'].fillna(1, inplace=True)
        # 将已上市且未退市，未停牌的股票标记为可交易(if_tradable = True)
        # 注意没有停牌数据的股票默认为停牌
        self.if_tradable['if_tradable'] = (self.if_tradable.ix['is_enlisted', :, :] *
            np.logical_not(self.if_tradable.ix['is_delisted', :, :]) *
            np.logical_not(self.if_tradable.ix['is_suspended', :, :])).astype(np.bool)
            
    # 四舍五入的函数
    @staticmethod
    def round(x, *, decimal=2):
        if np.isnan(x):
            return np.nan
        x_sign = np.sign(x)
        y = np.abs(x) * (10 ** (decimal+1))
        y = int(y) + 5
        y = np.trunc(y/10)
        y = y / (10 ** decimal)
        return y * x_sign


# if __name__ == '__main__':
#     fname = os.listdir(str(os.path.abspath('.')) + '/RiskModelData/')
#     for f in fname:
#         if f[-4:] != '.csv':
#             continue
#         curr_f = pd.read_csv(str(os.path.abspath('.')) + '/RiskModelData/' + f,
#                              index_col=0, parse_dates=True, encoding='GB18030')
#         data.write_data(curr_f, file_name=f[:-4], folder_name='ResearchData')
#     pass
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            