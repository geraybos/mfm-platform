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
                           folder_name=None, holding_freq='w',
                           stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None, bkt_end=None,
                           select_method=0, do_bb_pure_factor=False, do_pa=False, do_active_pa=False,
                           do_data_description=False, do_factor_corr_test=False, loc=-1):
    cp_adj = data.read_data(['ClosePrice_adj'])
    cp_adj = cp_adj['ClosePrice_adj']
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

    # 根据股票池进行循环
    for stock_pool in stock_pools:
        # 建立单因子测试对象
        curr_sf = single_factor_strategy()
        # from intangible_info import intangible_info_earnings
        # curr_sf = intangible_info_earnings()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，尽管这里并不是必要的
        curr_sf.single_factor_test(factor=factor, loc=loc, direction=direction, bkt_obj=copy.deepcopy(bkt_obj),
                                   base_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
                                   folder_name=folder_name,
                                   bkt_start=bkt_start, bkt_end=bkt_end, holding_freq=holding_freq,
                                   stock_pool=stock_pool, select_method=select_method,
                                   do_base_pure_factor=do_bb_pure_factor,
                                   do_pa=do_pa, do_active_pa=do_active_pa,
                                   do_data_description=do_data_description,
                                   do_factor_corr_test=do_factor_corr_test)

# 根据多个股票池进行一次完整的单因子测试, 多进程版
def sf_test_multiple_pools_parallel(factor=None, *, direction='+', bb_obj=None, discard_factor=(),
                                    folder_name=None,
                                    stock_pools=('all', 'hs300', 'zz500', 'zz800'), bkt_start=None,
                                    bkt_end=None, select_method=0, do_bb_pure_factor=False,
                                    do_pa=False, do_factor_corr_test=False, do_active_pa=False,
                                    holding_freq='w', do_data_description=False, loc=-1):
    cp_adj = data.read_data(['ClosePrice_adj'])
    cp_adj = cp_adj['ClosePrice_adj']
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

    def single_task(stock_pool):
        # curr_sf = single_factor_strategy()
        from intangible_info import intangible_info_earnings
        curr_sf = intangible_info_earnings()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，这是因为尽管bkt obj不会被改变，但是多进程同时操作可能出现潜在的问题
        curr_sf.single_factor_test(stock_pool=stock_pool, factor=factor, loc=loc, direction=direction,
                                   folder_name=folder_name,
                                   bkt_obj=copy.deepcopy(bkt_obj), base_obj=copy.deepcopy(bb_obj),
                                   discard_factor=discard_factor, bkt_start=bkt_start, bkt_end=bkt_end,
                                   select_method=select_method, do_base_pure_factor=do_bb_pure_factor,
                                   holding_freq=holding_freq, do_pa=do_pa, do_active_pa=do_active_pa,
                                   do_data_description=do_data_description, do_factor_corr_test=do_factor_corr_test)

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

# 测试wq101中的因子
# wq_data = data.read_data(['ClosePrice_adj', 'OpenPrice_adj', 'vwap_adj'],
#                          ['ClosePrice_adj', 'OpenPrice_adj', 'vwap_adj'],
#                          shift=True)

## 因子4
#low_rank = wq_data.ix['LowPrice'].rank(1)
#from scipy.stats import rankdata
#wq_f4 = -low_rank.rolling(10).apply(lambda x:rankdata(x)[-1])
## 因子4的moving average
#wq_f4_ma = wq_f4.rolling(5).mean()
# ret = np.log(wq_data['ClosePrice_adj']/wq_data['ClosePrice_adj'].shift(1))
# ret = np.log(wq_data['vwap_adj']/wq_data['vwap_adj'].shift(1))
# ret = ret.fillna(0)
# mom5 = -ret.rolling(5).sum()
# mom10 = -ret.rolling(10).sum()
# mom21 = -ret.rolling(21).sum()

# mom = mom5*3 + mom10*2 + mom21 * 1

# exp_w = barra_base.construct_expo_weights(5, 21)
# mom21 = ret.rolling(21).apply(lambda x:(x*exp_w).sum())
# exp_w2 = barra_base.construct_expo_weights(126, 504)
# mom504 = ret.rolling(504).apply(lambda x:(x*exp_w2).sum())
# mom252 = -ret.rolling(504).sum()
# rv1 = data.read_data(['runner_value_1'], shift=True)

# from intangible_info import intangible_info_earnings, intangible_info
# ii = intangible_info()
# ii.prepare_data()
# ii.get_bv_return_direct()
# ii.get_bv_return_indirect()
# ii.check_two_bv_returns()
# ii.get_intangible_return()
# mom = data.read_data(['momentum'], shift=True)
# mom = mom['momentum']
# pe = data.read_data(['PE_ttm'], shift=True)
# lagged_ep = (1/pe['PE_ttm']).shift(252*2)
# rv8 = data.read_data(['runner_value_8'], shift=True)
# rv8 = -rv8['runner_value_8']
# bb = data.read_data(['rv', 'liquidity', 'lncap'], shift=True)
# # bb = data.read_data(['runner_value_36'], shift=True)
# orth_mom = strategy_data.simple_orth_gs(mom504, bb)
# # orth_mom = strategy_data.simple_orth_gs(ii.intangible_return, bb)
# orth_mom = orth_mom[0]

rv = pd.read_hdf('stock_alpha_hs300_split', '123')
for iname, idf in rv.iteritems():

    sf_test_multiple_pools(factor=idf, direction='+', folder_name='naive_test/'+iname,
                           bkt_start=pd.Timestamp('2011-05-04'), holding_freq='w',
                           bkt_end=pd.Timestamp('2017-03-09'), stock_pools=('hs300', ),
                           do_bb_pure_factor=False, do_pa=False, select_method=1, do_active_pa=True,
                           do_data_description=False, do_factor_corr_test=False, loc=-1)

# sf_test_multiple_pools_parallel(factor='default', direction='+', bkt_start=pd.Timestamp('2010-04-02'),
#                                 bkt_end=pd.Timestamp('2017-06-20'), stock_pools=['sz50', 'zxb', 'cyb', 'hs300', 'zz500'],
#                                 do_bb_pure_factor=False, do_pa=True, select_method=1, do_active_pa=True,
#                                 do_data_description=False, holding_freq='w', do_factor_corr_test=False,
#                                 loc=-1)


































































