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
def sf_test_multiple_pools(factor='default', *, direction='+', bb_obj='Empty', discard_factor=[], holding_freq='m',
                           stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default', bkt_end='default',
                           select_method=0, do_bb_pure_factor=False, do_active_bb_pure_factor=False,
                           do_pa=False, do_active_pa=False, do_data_description=False,
                           do_factor_corr_test=False):
    cp_adj = data.read_data(['ClosePrice_adj'])
    cp_adj = cp_adj['ClosePrice_adj']
    temp_position = position(cp_adj)

    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=0/1000, sell_cost=0/1000)
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj == 'Empty':
        bb_obj = barra_base()
        bb_obj.just_get_sytle_factor()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    # 根据股票池进行循环
    for stock_pool in stock_pools:
        # 建立单因子测试对象
        # curr_sf = single_factor_strategy()
        # from analyst_coverage import analyst_coverage
        # curr_sf = analyst_coverage()
        from residual_income import residual_income
        curr_sf = single_factor_strategy()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，尽管这里并不是必要的
        curr_sf.single_factor_test(factor=factor, direction=direction, bkt_obj=copy.deepcopy(bkt_obj),
                                   bb_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
                                   bkt_start=bkt_start, bkt_end=bkt_end, holding_freq=holding_freq,
                                   stock_pool=stock_pool, select_method=select_method,
                                   do_bb_pure_factor=do_bb_pure_factor,
                                   do_active_bb_pure_factor=do_active_bb_pure_factor,
                                   do_pa=do_pa, do_active_pa=do_active_pa,
                                   do_data_description=do_data_description,
                                   do_factor_corr_test=do_factor_corr_test)

# 根据多个股票池进行一次完整的单因子测试, 多进程版
def sf_test_multiple_pools_parallel(factor='default', *, direction='+', bb_obj='Empty', discard_factor=[],
                                    stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default',
                                    bkt_end='default', select_method=0, do_bb_pure_factor=False,
                                    do_active_bb_pure_factor=False, do_pa=False, do_factor_corr_test=False,
                                    do_active_pa=False, holding_freq='m', do_data_description=False):
    cp_adj = data.read_data(['ClosePrice_adj'])
    cp_adj = cp_adj['ClosePrice_adj']
    temp_position = position(cp_adj)

    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end, buy_cost=1.5/1000, sell_cost=1.5/1000)
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj == 'Empty':
        bb_obj = barra_base()
        bb_obj.just_get_sytle_factor()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    def single_task(stock_pool):
        # curr_sf = single_factor_strategy()
        from analyst_coverage import analyst_coverage
        # curr_sf = analyst_coverage()
        from residual_income import residual_income
        curr_sf = residual_income()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，这是因为尽管bkt obj不会被改变，但是多进程同时操作可能出现潜在的问题
        curr_sf.single_factor_test(stock_pool=stock_pool, factor=factor, direction=direction,
                                   bkt_obj=copy.deepcopy(bkt_obj), bb_obj=copy.deepcopy(bb_obj),
                                   discard_factor=discard_factor, bkt_start=bkt_start, bkt_end=bkt_end,
                                   select_method=select_method, do_bb_pure_factor=do_bb_pure_factor,
                                   do_active_bb_pure_factor=do_active_bb_pure_factor, holding_freq=holding_freq,
                                   do_pa=do_pa, do_active_pa=do_active_pa, do_data_description=do_data_description,
                                   do_factor_corr_test=do_factor_corr_test)

    import multiprocessing as mp
    mp.set_start_method('fork')
    # 根据股票池进行循环
    for stock_pool in stock_pools:
        p = mp.Process(target=single_task, args=(stock_pool,))
        p.start()




# 进行单因子测试

# # 测试eps_fy1, eps_fy2的varaition coeffcient
# eps_fy = data.read_data(['EPS_fy1', 'EPS_fy2'])
# eps_fy1 = eps_fy['EPS_fy1']
# eps_fy2 = eps_fy['EPS_fy2']
#
# eps_vc = 0.5*eps_fy1.rolling(252).std()/eps_fy1.rolling(252).mean() + \
#         0.5*eps_fy1.rolling(252).std()/eps_fy1.rolling(252).mean()

# # 测试wq101中的因子
# wq_data = data.read_data(['ClosePrice_adj', 'OpenPrice_adj', 'Volume', 'HighPrice', 'LowPrice', 'AdjustFactor'],
#                          ['ClosePrice_adj', 'OpenPrice_adj', 'Volume', 'HighPrice', 'LowPrice', 'AdjustFactor'],
#                          shift=True)
# wq_data['HighPrice_adj'] = wq_data['HighPrice'] * wq_data['AdjustFactor']
# wq_data['LowPrice_adj'] = wq_data['LowPrice'] * wq_data['AdjustFactor']

## 因子4
#low_rank = wq_data.ix['LowPrice'].rank(1)
#from scipy.stats import rankdata
#wq_f4 = -low_rank.rolling(10).apply(lambda x:rankdata(x)[-1])
## 因子4的moving average
#wq_f4_ma = wq_f4.rolling(5).mean()
# ret = np.log(wq_data['ClosePrice_adj']/wq_data['ClosePrice_adj'].shift(1))
# mom5 = -ret.rolling(5).sum()
# mom10 = -ret.rolling(10).sum()
# mom21 = -ret.rolling(21).sum()
#
# mom = mom5*3 + mom10*2 + mom21 * 1

# exp_w = barra_base.construct_expo_weights(5, 21)
# mom21 = -ret.rolling(21).apply(lambda x:(x*exp_w).sum())

# # 短期流动性因子
# volume = pd.read_csv('Volume.csv',index_col=0,parse_dates=True)
# freeshares = pd.read_csv('FreeShares.csv',index_col=0,parse_dates=True)
# turnover = (volume/freeshares).shift(1)
# short_liq = -turnover.rolling(21).apply(lambda x:(x*exp_w).sum())

# # 测试已有的因子
# contrarian = pd.read_csv('contrarian.csv', index_col=0, parse_dates=True)
# contrarian = contrarian.shift(1)

# # 测试analyst coverage
# coverage = pd.read_csv('unique_coverage.csv', index_col=0, parse_dates=True)
# coverage = coverage.shift(1)
#
# 论文里的abnormal coverage
# from analyst_coverage import analyst_coverage
# ac = analyst_coverage()
# ac.get_abn_coverage_poisson()
# abn_coverage = ac.strategy_data.factor.ix['abn_coverage']

# mv = data.read_data(['FreeMarketValue'])
# mv = mv['FreeMarketValue']
# # 测试经纬的sue
# sue = pd.read_csv('sue_signal500.csv', index_col=0, parse_dates=[2])
# sue = sue.pivot_table(index='TradingDay', columns='SecuCode', values='rawSignal')
# # sue = sue.shift(1)
# new_col = []
# for curr_stock in sue.columns:
#     new_col.append(str(curr_stock).zfill(6))
# sue.columns = new_col
# sue = sue.reindex(index=mv.index, columns=mv.columns, method='ffill').fillna(0.0)
# pass

sf_test_multiple_pools(factor='runner_value_36', direction='+', bkt_start=pd.Timestamp('2011-01-04'), holding_freq='w',
                       bkt_end=pd.Timestamp('2016-12-30'), stock_pools=['zz500'],
                       do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
                       do_data_description=False, do_factor_corr_test=False)

# sf_test_multiple_pools_parallel(factor='default', direction='+', bkt_start=pd.Timestamp('2011-01-04'),
#                                 bkt_end=pd.Timestamp('2017-05-31'), stock_pools=['hs300','zz500'],
#                                 do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
#                                 do_data_description=True, holding_freq='w', do_factor_corr_test=False)


































































