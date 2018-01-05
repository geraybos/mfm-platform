#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:17:06 2017

@author: lishiwang
"""


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from barra_base import barra_base
from single_factor_strategy import single_factor_strategy

# 根据多个股票池进行一次完整的单因子测试
def sf_test_multiple_pools(factor=None, *, direction='+', bb_obj=None, discard_factor=(),
                           folder_name=None, holding_freq='w', benchmark=None,
                           stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None, bkt_end=None,
                           select_method=0, do_bb_pure_factor=False, do_pa=False, do_active_pa=False,
                           do_data_description=False, do_factor_corr_test=False, loc=-1):
    cp_adj = data.read_data('ClosePrice_adj')
    temp_position = position(cp_adj)

    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=1.5/1000, sell_cost=1.5/1000,
                       bkt_benchmark_data='ClosePrice_adj_hs300')
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj is None:
        bb_obj = barra_base()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    # 根据股票池进行循环
    for cursor, stock_pool in enumerate(stock_pools):
        # 建立单因子测试对象
        curr_sf = single_factor_strategy()
        # from intangible_info import intangible_info_earnings
        # curr_sf = intangible_info_earnings()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，尽管这里并不是必要的
        curr_sf.single_factor_test(factor=factor, loc=loc, direction=direction, bkt_obj=copy.deepcopy(bkt_obj),
                                   base_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
                                   folder_name=folder_name, bkt_start=bkt_start, bkt_end=bkt_end,
                                   holding_freq=holding_freq, benchmark=benchmark[cursor],
                                   stock_pool=stock_pool, select_method=select_method,
                                   do_base_pure_factor=do_bb_pure_factor,
                                   do_pa=do_pa, do_active_pa=do_active_pa,
                                   do_data_description=do_data_description,
                                   do_factor_corr_test=do_factor_corr_test)

# 根据多个股票池进行一次完整的单因子测试, 多进程版
def sf_test_multiple_pools_parallel(factor=None, *, direction='+', bb_obj=None, discard_factor=(),
                                    folder_name=None, benchmark=None,
                                    stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None,
                                    bkt_end=None, select_method=0, do_bb_pure_factor=False,
                                    do_pa=False, do_factor_corr_test=False, do_active_pa=False,
                                    holding_freq='w', do_data_description=False, loc=-1):
    cp_adj = data.read_data('ClosePrice_adj')
    temp_position = position(cp_adj)

    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=1.5/1000, sell_cost=1.5/1000)
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj is None:
        bb_obj = barra_base()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    def single_task(cursor, stock_pool):
        # curr_sf = single_factor_strategy()
        from intangible_info import intangible_info_earnings
        curr_sf = intangible_info_earnings()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，这是因为尽管bkt obj不会被改变，但是多进程同时操作可能出现潜在的问题
        curr_sf.single_factor_test(stock_pool=stock_pool, factor=factor, loc=loc, direction=direction,
                                   folder_name=folder_name, bkt_obj=copy.deepcopy(bkt_obj),
                                   base_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
                                   bkt_start=bkt_start, bkt_end=bkt_end, benchmark=benchmark[cursor],
                                   select_method=select_method, do_base_pure_factor=do_bb_pure_factor,
                                   holding_freq=holding_freq, do_pa=do_pa, do_active_pa=do_active_pa,
                                   do_data_description=do_data_description, do_factor_corr_test=do_factor_corr_test)

    import multiprocessing as mp
    mp.set_start_method('fork')
    # 根据股票池进行循环
    for cursor, stock_pool in enumerate(stock_pools):
        p = mp.Process(target=single_task, args=(cursor, stock_pool))
        p.start()


# 进行单因子测试
alpha = data.read_data('runner_value_63', shift=True)

sf_test_multiple_pools(factor=alpha, direction='+', folder_name='tar_holding_bkt/test',
                       bkt_start=pd.Timestamp('2016-01-04'), holding_freq='w',
                       bkt_end=pd.Timestamp('2017-11-14'), stock_pools=('hs300', ), benchmark=('hs300', ),
                       do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
                       do_data_description=False, do_factor_corr_test=False, loc=-1)

# sf_test_multiple_pools_parallel(factor='default', direction='+', bkt_start=pd.Timestamp('2010-04-02'),
#                                 bkt_end=pd.Timestamp('2017-06-20'), stock_pools=['sz50', 'zxb', 'cyb', 'hs300', 'zz500'],
#                                 do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
#                                 do_data_description=False, holding_freq='w', do_factor_corr_test=False,
#                                 loc=-1)


































































