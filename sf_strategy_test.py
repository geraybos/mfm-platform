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
def sf_test_multiple_pools(factor=None, sf_obj=single_factor_strategy(), *, direction='+', bb_obj=None,
                           discard_factor=(), folder_names=None, holding_freq='w', benchmarks=None,
                           stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None, bkt_end=None,
                           select_method=0, do_bb_pure_factor=False, do_pa=False, do_active_pa=False,
                           do_data_description=False, do_factor_corr_test=False, loc=-1):
    # 打印当前测试的策略名称
    print('Name Of Strategy Under Test: {0}\n'.format(sf_obj.__class__.__name__))

    cp_adj = data.read_data('ClosePrice_adj')
    temp_position = position(cp_adj)
    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=0.0015,
                       sell_cost=0.0015, bkt_stock_data=['ClosePrice_adj', 'ClosePrice_adj'])
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj is None:
        bb_obj = barra_base()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    # 根据股票池进行循环
    for cursor, stock_pool in enumerate(stock_pools):
        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，尽管这里并不是必要的
        sf_obj.single_factor_test(factor=factor, loc=loc, direction=direction, bkt_obj=copy.deepcopy(bkt_obj),
            base_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
            folder_name=folder_names[cursor], bkt_start=bkt_start, bkt_end=bkt_end,
            holding_freq=holding_freq, benchmark=benchmarks[cursor], stock_pool=stock_pool,
            select_method=select_method, do_base_pure_factor=do_bb_pure_factor,
            do_pa=do_pa, do_active_pa=do_active_pa, do_data_description=do_data_description,
            do_factor_corr_test=do_factor_corr_test)


# 根据多个股票池进行一次完整的单因子测试, 多进程版
def sf_test_multiple_pools_parallel(factor=None, sf_obj=single_factor_strategy(), *, direction='+',
                                    bb_obj=None, discard_factor=(), folder_names=None, benchmarks=None,
                                    stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None,
                                    bkt_end=None, select_method=0, do_bb_pure_factor=False,
                                    do_pa=False, do_factor_corr_test=False, do_active_pa=False,
                                    holding_freq='w', do_data_description=False, loc=-1):
    # 打印当前测试的策略名称
    print('Name Of Strategy Under Test: {0}\n'.format(sf_obj.__class__.__name__))

    cp_adj = data.read_data('ClosePrice_adj')
    temp_position = position(cp_adj)
    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=1.5/1000,
                       sell_cost=1.5/1000, bkt_stock_data=['ClosePrice_adj', 'ClosePrice_adj'])
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj is None:
        bb_obj = barra_base()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    def single_task(cursor, stock_pool):
        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，这是因为尽管bkt obj不会被改变，但是多进程同时操作可能出现潜在的问题
        sf_obj.single_factor_test(stock_pool=stock_pool, factor=factor, loc=loc, direction=direction,
            folder_name=folder_names[cursor], bkt_obj=copy.deepcopy(bkt_obj),
            base_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor, bkt_start=bkt_start,
            bkt_end=bkt_end, benchmark=benchmarks[cursor], select_method=select_method,
            do_base_pure_factor=do_bb_pure_factor, holding_freq=holding_freq, do_pa=do_pa,
            do_active_pa=do_active_pa, do_data_description=do_data_description,
            do_factor_corr_test=do_factor_corr_test)

    import multiprocessing as mp
    mp.set_start_method('fork')
    # 根据股票池进行循环
    for cursor, stock_pool in enumerate(stock_pools):
        p = mp.Process(target=single_task, args=(cursor, stock_pool))
        p.start()

if __name__ == '__main__':
    # 进行单因子测试
    # factor = data.read_data('runner_value_63', shift=True)
    from analyst_coverage import analyst_coverage_new
    acn = analyst_coverage_new()
    acn.construct_factor()
    factor = acn.strategy_data.factor.iloc[0]

    # foldername_prefix = 'tar_holding_bkt/damu_alpha/holding_test/08_'
    foldername_prefix = 'analyst_coverage_new/abn_barra_2&1&1_onlyGreat_allrpt_'

    # sf_test_multiple_pools(factor=factor, sf_obj=single_factor_strategy(), direction='+',
    #                        folder_names=(foldername_prefix+'hs300', ),
    #                        bkt_start=pd.Timestamp('2017-01-03'),
    #                        bkt_end=pd.Timestamp('2017-12-29'), holding_freq='w',
    #                        stock_pools=('hs300', ), benchmarks=('hs300', ),
    #                        do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
    #                        do_data_description=False, do_factor_corr_test=False, loc=-1)

    sf_test_multiple_pools_parallel(factor=factor, sf_obj=acn, direction='+',
                           folder_names=(foldername_prefix+'hs300', foldername_prefix+'zz500',
                                         foldername_prefix+'zz500_spec'),
                           bkt_start=pd.Timestamp('2009-05-04'),
                           bkt_end=pd.Timestamp('2018-01-16'), holding_freq='w',
                           stock_pools=('hs300', 'zz500', 'all'), benchmarks=('hs300', 'zz500', 'zz500'),
                           do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
                           do_data_description=False, do_factor_corr_test=False, loc=-1)
































































