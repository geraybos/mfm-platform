#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:41:06 2017

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
import pathos.multiprocessing as mp

from data import data
from strategy_data import strategy_data
from position import position
from factor_base import factor_base

# barra base类，计算barra的风格因子以及行业因子，以此作为基础
# 注意：此类中参照barra计算的因子中，excess return暂时都没有减去risk free rate

class barra_base(factor_base):
    """This is the class for barra base construction.
    
    foo
    """
    def __init__(self, *, stock_pool='all'):
        factor_base.__init__(self, stock_pool=stock_pool)
        # 储存行业数据
        self.industry = pd.DataFrame()
        # 提示是否为数据更新
        self.is_update = False
        # 提示数据计算中是否尝试读取现有文件
        self.try_to_read = True

    # 读取在计算风格因子中需要用到的原始数据，注意：行业信息数据不在这里读取
    # 即使可以读取已算好的因子，也在这里处理，因为这样方便统一处理，不至于让代码太乱
    # 这里的标准为，读取前都检查一下是否已经存在数据，这样可以方便手动读取特定数据
    def read_original_data(self):
        # 先读取流通市值
        if self.base_data.stock_price.empty:
            self.base_data.stock_price = data.read_data(['FreeMarketValue'], ['FreeMarketValue'])
        elif 'FreeMarketValue' not in self.base_data.stock_price.items:
            fmv = data.read_data(['FreeMarketValue'], ['FreeMarketValue'])
            self.base_data.stock_price['FreeMarketValue'] = fmv.ix['FreeMarketValue']
        # 初始化无风险利率序列
        if os.path.isfile('const_data.csv'):
            self.base_data.const_data = pd.read_csv('const_data.csv', index_col=0, parse_dates=True, encoding='GB18030')
            if 'RiskFreeRate' not in self.base_data.const_data.columns:
                self.base_data.const_data['RiskFreeRate'] = 0
            else:
                print('risk free rate successfully loaded')
        else:
            self.base_data.const_data = pd.DataFrame(0, index=self.base_data.stock_price.major_axis,
                                                   columns=['RiskFreeRate'])
        # 读取市值
        if 'MarketValue' not in self.base_data.stock_price.items:
            temp_mv =data.read_data(['MarketValue'], ['MarketValue'])
            self.base_data.stock_price['MarketValue'] = temp_mv.ix['MarketValue']
        # 读取价格数据
        if 'ClosePrice_adj' not in self.base_data.stock_price.items:
            temp_closeprice = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'])
            self.base_data.stock_price['ClosePrice_adj'] = temp_closeprice.ix['ClosePrice_adj']
        # 计算每只股票的日对数收益率
        if 'daily_log_return' not in self.base_data.stock_price.items:
            self.base_data.stock_price['daily_log_return'] = np.log(self.base_data.stock_price.ix['ClosePrice_adj'].div(
                self.base_data.stock_price.ix['ClosePrice_adj'].shift(1)))
        # 计算每只股票的日超额对数收益
        if 'daily_excess_log_return' not in self.base_data.stock_price.items:
            self.base_data.stock_price['daily_excess_log_return'] = self.base_data.stock_price.ix['daily_log_return'].sub(
                self.base_data.const_data.ix[:, 'RiskFreeRate'], axis=0)
        # 计算每只股票的日简单收益，注意是按日复利的日化收益，即代表净值增值
        if 'daily_simple_return' not in self.base_data.stock_price.items:
            self.base_data.stock_price['daily_simple_return'] = self.base_data.stock_price.ix['ClosePrice_adj'].div(
                self.base_data.stock_price.ix['ClosePrice_adj'].shift(1)).sub(1.0)
        # 计算每只股票的日超额简单收益，注意是按日复利的日化收益，
        # 另外注意RiskFreeRate默认是连续复利，要将其转化成对应的简单收益
        if 'daily_excess_simple_return' not in self.base_data.stock_price.items:
            self.base_data.const_data['RiskFreeRate_simple'] = np.exp(self.base_data.const_data['RiskFreeRate']) - 1
            self.base_data.stock_price['daily_excess_simple_return'] = self.base_data.stock_price. \
                ix['daily_simple_return'].sub(self.base_data.const_data['RiskFreeRate_simple'], axis=0)
        # 读取交易量数据
        if 'Volume' not in self.base_data.stock_price.items:
            volume = data.read_data(['Volume'], ['Volume'])
            self.base_data.stock_price['Volume'] = volume.ix['Volume']
        # 读取流通股数数据
        if 'FreeShares' not in self.base_data.stock_price.items:
            shares = data.read_data(['FreeShares'], ['FreeShares'])
            self.base_data.stock_price['FreeShares'] = shares.ix['FreeShares']
        # 读取pb
        if self.base_data.raw_data.empty:
            self.base_data.raw_data = data.read_data(['PB'],['PB'])
            # 一切的数据标签都以stock_price为准
            self.base_data.raw_data = data.align_index(self.base_data.stock_price.ix[0], 
                                                     self.base_data.raw_data, axis = 'both')
        elif 'PB' not in self.base_data.raw_data.items:
            pb = data.read_data(['PB'], ['PB'])
            self.base_data.raw_data['PB'] = pb.ix['PB']
        # 读取ni_fy1, ni_fy2
        if 'NetIncome_fy1' not in self.base_data.raw_data.items:
            NetIncome_fy1 = data.read_data(['NetIncome_fy1'], ['NetIncome_fy1'])
            self.base_data.raw_data['NetIncome_fy1'] = NetIncome_fy1.ix['NetIncome_fy1']
        if 'NetIncome_fy2' not in self.base_data.raw_data.items:
            NetIncome_fy2 = data.read_data(['NetIncome_fy2'], ['NetIncome_fy2'])
            self.base_data.raw_data['NetIncome_fy2'] = NetIncome_fy2.ix['NetIncome_fy2']
        # 读取cash_earnings_ttm，现金净流入的ttm
        if 'CashEarnings_ttm' not in self.base_data.raw_data.items:
            CashEarnings_ttm = data.read_data(['CashEarnings_ttm'], ['CashEarnings_ttm'])
            self.base_data.raw_data['CashEarnings_ttm'] = CashEarnings_ttm.ix['CashEarnings_ttm']
        # 读取pe_ttm
        if 'PE_ttm' not in self.base_data.raw_data.items:
            pe_ttm = data.read_data(['PE_ttm'], ['PE_ttm'])
            self.base_data.raw_data['PE_ttm'] = pe_ttm.ix['PE_ttm']
        # 读取净利润net income ttm
        if 'NetIncome_ttm' not in self.base_data.raw_data.items:
            ni_ttm = data.read_data(['NetIncome_ttm'], ['NetIncome_ttm'])
            self.base_data.raw_data['NetIncome_ttm'] = ni_ttm.ix['NetIncome_ttm']
        # 读取ni ttm的2年增长率，用ni增长率代替eps增长率，因为ni增长率的数据更全
        if 'NetIncome_ttm_growth_8q' not in self.base_data.raw_data.items:
            ni_ttm_growth_8q = data.read_data(['NetIncome_ttm_growth_8q'], ['NetIncome_ttm_growth_8q'])
            self.base_data.raw_data['NetIncome_ttm_growth_8q'] = ni_ttm_growth_8q.ix['NetIncome_ttm_growth_8q']
        # 读取revenue ttm的2年增长率
        if 'Revenue_ttm_growth_8q' not in self.base_data.raw_data.items:
            Revenue_ttm_growth_8q = data.read_data(['Revenue_ttm_growth_8q'], ['Revenue_ttm_growth_8q'])
            self.base_data.raw_data['Revenue_ttm_growth_8q'] = Revenue_ttm_growth_8q.ix['Revenue_ttm_growth_8q']
        # 读取总资产和总负债，用资产负债率代替复杂的leverage因子
        if 'TotalAssets' not in self.base_data.raw_data.items:
            TotalAssets = data.read_data(['TotalAssets'], ['TotalAssets'])
            self.base_data.raw_data['TotalAssets'] = TotalAssets.ix['TotalAssets']
        if 'TotalLiability' not in self.base_data.raw_data.items:
            TotalLiability = data.read_data(['TotalLiability'], ['TotalLiability'])
            self.base_data.raw_data['TotalLiability'] = TotalLiability.ix['TotalLiability']
        # 生成可交易及可投资数据
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        # 需要建立一个储存完整数据的数据类, 因为如果股票池不是全市场, 一旦过滤掉uninv
        # 则会将某时刻不在股票池内的数据都过滤掉, 但是在因子计算过程中, 很多因子需要用到股票过去的数据
        # 如果股票过去不在股票池中, 则会出现没有数据的情况, 这是不合理的, 这时就需要从这里提取完整的数据
        # 即股票在a时刻的因子值要用到在a-t时刻的数据时, 都需要用这里的数据
        # 但是注意, 不可投资数据仍然是需要过滤的
        self.base_data.discard_untradable_data()
        self.complete_base_data = copy.deepcopy(self.base_data)
        # 读取完所有数据后，过滤数据
        # 注意：在之后的因子计算中，中间计算出的因子之间相互依赖的，都要再次过滤，如一个需要从另一个中算出
        # 或者回归，正交化，而且凡是由多个因子加权得到的因子，都属于这一类
        # 以及因子的计算过程中用到非此时间点的原始数据时，如在a时刻的因子值要用到在a-t时刻的原始数据
        # 需要算暴露的时候，一定要过滤uninv的数据，因为暴露是在股票池中计算的，正交化的时候也需要过滤uninv
        # 因为正交化是希望在股票池内正交化, 同样beta也要股票uninv, 因为这里的beta衡量的是对股票池的beta
        # 但同时注意，一旦过滤uninv，数据就不能再作为一般的因子值储存了
        self.base_data.discard_uninv_data()

    # 计算市值对数因子，市值需要第一个计算以确保各个panel有index和column
    def get_lncap(self):
        # 如果有文件，则直接读取
        if os.path.isfile('lncap'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            self.base_data.factor = data.read_data(['lncap'+self.filename_appendix], ['lncap'])
        # 没有就用市值进行计算
        else:
            self.base_data.factor = pd.Panel({'lncap':np.log(self.base_data.stock_price.ix['FreeMarketValue'])},
                                             major_axis=self.base_data.stock_price.major_axis,
                                             minor_axis=self.base_data.stock_price.minor_axis)

    # 计算beta因子
    def get_beta(self):
        if os.path.isfile('beta'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            beta = data.read_data(['beta'+self.filename_appendix+'.csv'], ['beta'])
            self.base_data.factor['beta'] = beta.ix['beta']
        else:
            # 所有股票的日简单收益的市值加权，加权用前一交易日的市值数据进行加权
            cap_wgt_universe_return = self.base_data.stock_price.ix['daily_excess_simple_return'].mul(
                                       self.base_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
                                       self.base_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)
            
            # 指数权重
            exponential_weights = strategy_data.construct_expo_weights(63, 252)
            # 回归函数
            def reg_func(y, *, x, weights):
                # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
                if y.notnull().sum() <= 63:
                    return pd.Series({'beta':np.nan,'hsigma':np.nan})
                x = sm.add_constant(x)
                model = sm.WLS(y, x, weights=weights, missing='drop')
                results = model.fit()
                resid = results.resid.reindex(index=y.index)
                # 在这里提前计算hsigma----------------------------------------------------------------------
                # 求std，252个交易日，63的半衰期
                exponential_weights_h = strategy_data.construct_expo_weights(63, 252)
                # 给weights加上index以索引resid
                exponential_weights_h = pd.Series(exponential_weights_h, index=y.index)
                # 给resid直接乘以权重, 然后按照权重计算加权的std
                weighted_resid = strategy_data.multiply_weights(resid, exponential_weights_h, multiply_power=0.5)
                hsigma = weighted_resid.std()
                # ----------------------------------------------------------------------------------------
                return pd.Series({'beta':results.params[1],'hsigma':hsigma})
            # 按照Barra的方法进行回归
            # 储存回归结果的dataframe
            temp_beta = self.base_data.stock_price.ix['daily_excess_simple_return']*np.nan
            temp_hsigma = self.base_data.stock_price.ix['daily_excess_simple_return']*np.nan
            for cursor, date in enumerate(self.base_data.stock_price.ix['daily_excess_simple_return'].index):
                # 至少第252期时才回归
                if cursor<=250:
                    continue
                # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据
                curr_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return',cursor-251:cursor+1,:]
                curr_x = cap_wgt_universe_return.ix[cursor-251:cursor+1]
                temp = curr_data.apply(reg_func, x=curr_x, weights=exponential_weights)
                temp_beta.ix[cursor,:] = temp.ix['beta']
                temp_hsigma.ix[cursor,:] = temp.ix['hsigma']
                print(cursor)
                pass
            self.base_data.factor['beta'] = temp_beta
            self.temp_hsigma = temp_hsigma
            pass

    # beta parallel
    def get_beta_parallel(self):
        if os.path.isfile('beta'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            beta = data.read_data(['beta'+self.filename_appendix], ['beta'])
            self.base_data.factor['beta'] = beta.ix['beta']
        else:
            # 所有股票的日简单收益的市值加权，加权用前一交易日的市值数据进行加权
            cap_wgt_universe_return = self.base_data.stock_price.ix['daily_excess_simple_return'].mul(
                self.base_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
                self.base_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)

            # 指数权重
            exponential_weights = strategy_data.construct_expo_weights(63, 252)

            # 回归函数
            def reg_func(y, *, x, weights):
                # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
                if y.notnull().sum() <= 63:
                    return pd.Series({'beta': np.nan, 'hsigma': np.nan})
                x = sm.add_constant(x)
                model = sm.WLS(y, x, weights=weights, missing='drop')
                results = model.fit()
                resid = results.resid.reindex(index=y.index)
                # 在这里提前计算hsigma----------------------------------------------------------------------
                # 求std，252个交易日，63的半衰期
                exponential_weights_h = strategy_data.construct_expo_weights(63, 252)
                # 给weights加上index以索引resid
                exponential_weights_h = pd.Series(exponential_weights_h, index=y.index)
                # 给resid直接乘以权重, 然后按照权重计算加权的std
                weighted_resid = strategy_data.multiply_weights(resid, exponential_weights_h, multiply_power=0.5)
                hsigma = weighted_resid.std()
                # ----------------------------------------------------------------------------------------
                return pd.Series({'beta': results.params[1], 'hsigma': hsigma})
            # 按照Barra的方法进行回归
            # 股票收益的数据
            complete_return_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return']
            # 计算每期beta的函数
            def one_time_beta(cursor):
                # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据
                # curr_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return', cursor - 251:cursor+1, :]
                curr_data = complete_return_data.ix[cursor - 251:cursor+1, :]
                curr_x = cap_wgt_universe_return.ix[cursor - 251:cursor+1]
                temp = curr_data.apply(reg_func, x=curr_x, weights=exponential_weights)
                print(cursor)
                return temp

            ncpus = 20
            p = mp.ProcessPool(ncpus)
            # 从252期开始
            data_size = np.arange(251, self.base_data.stock_price.ix['daily_excess_simple_return'].shape[0])
            chunksize = int(len(data_size)/ncpus)
            results = p.map(one_time_beta, data_size, chunksize=chunksize)
            # 储存结果
            beta = pd.concat([i.ix['beta'] for i in results], axis=1).T
            hsigma = pd.concat([i.ix['hsigma'] for i in results], axis=1).T
            # 两个数据对应的日期，为原始数据的日期减去251，因为前251期的数据并没有计算
            data_index = self.base_data.stock_price.iloc[:, 251-self.base_data.stock_price.shape[1]:, :].major_axis
            beta = beta.set_index(data_index)
            hsigma = hsigma.set_index(data_index)
            self.base_data.factor['beta'] = beta
            self.temp_hsigma = hsigma.reindex(self.base_data.stock_price.major_axis)

    # 计算momentum因子 
    def get_momentum(self):
        if os.path.isfile('momentum'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            momentum = data.read_data(['momentum'+self.filename_appendix], ['momentum'])
            self.base_data.factor['momentum'] = momentum.ix['momentum']
        else:
            # 计算momentum因子
            # 首先数据有一个21天的lag， 注意收益要用对数收益
            # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据
            lag_return = self.complete_base_data.stock_price.ix['daily_excess_log_return'].shift(21)
            # rolling后求sum，504个交易日，126的半衰期
            exponential_weights = strategy_data.construct_expo_weights(126, 504)
            # 定义momentum的函数
            def func_mom(df, *, weights):
                iweights = pd.Series(weights, index=df.index)
                # 将权重乘在原始数据上, 然后加和计算momentum
                weighted_return = strategy_data.multiply_weights(df, iweights, multiply_power=1.0)
                mom = weighted_return.sum(0)
                # 设定阈值, 表示至少过去两年中有多少数据才能有momentum因子
                threshold_condition = df.notnull().sum(0) >= 63
                mom = mom.where(threshold_condition, np.nan)
                return mom
            momentum = self.base_data.stock_price.ix['daily_excess_log_return']*np.nan
            for cursor, date in enumerate(lag_return.index):
                # 至少504+21期才开始计算
                if cursor<=(502+21):
                    continue
                curr_data = lag_return.ix[cursor-503:cursor+1, :]
                temp = func_mom(curr_data, weights=exponential_weights)
                momentum.ix[cursor, :] = temp
            self.base_data.factor['momentum'] = momentum
            pass
        
     # 计算residual volatility中的dastd
    def get_rv_dastd(self):
        if os.path.isfile('dastd'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            dastd = data.read_data(['dastd'+self.filename_appendix], ['dastd'])
            dastd = dastd.ix['dastd']
        else:
            # rolling后求std，252个交易日，42的半衰期
            exponential_weights = strategy_data.construct_expo_weights(42, 252)
            # 定义dastd的函数
            def func_dastd(df, *, weights):
                iweights = pd.Series(weights, index=df.index)
                # 将权重乘在原始数据上, 然后计算std
                weighted_return = strategy_data.multiply_weights(df, iweights, multiply_power=0.5)
                dastd = weighted_return.std(0)
                return dastd
            dastd = self.base_data.stock_price.ix['daily_excess_simple_return']*np.nan
            for cursor, date in enumerate(self.base_data.stock_price.ix['daily_excess_simple_return'].index):
                # 至少252期才开始计算
                if cursor<=250:
                    continue
                # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据，且要用简单收益
                curr_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return', cursor-251:cursor+1,:]
                temp = func_dastd(curr_data, weights=exponential_weights)
                dastd.ix[cursor,:] = temp
  
        self.base_data.raw_data['dastd'] = dastd
    
    # 计算residual volatility中的cmra
    def get_rv_cmra(self):
        if os.path.isfile('cmra'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            cmra = data.read_data(['cmra'+self.filename_appendix], ['cmra'])
            cmra = cmra.ix['cmra']
        else:
            # 定义需要cmra的函数，这个函数计算252个交易日中的cmra
            def func_cmra(df):
                # 累计收益率
                cum_df = df.cumsum(axis=0)
                # 取每月的累计收益率
                months = np.arange(20,252,21)
                months_cum_df = cum_df.ix[months]
                z_max = months_cum_df.max(axis=0)
                z_min = months_cum_df.min(axis=0)
#                # 避免出现log函数中出现非正参数
#                z_min[z_min <= -1] = -0.9999
#                return np.log(1+z_max)-np.log(1+z_min)
                # 为避免出现z_min<=-1调整后的极端值，cmra改为z_max-z_min
                # 注意：改变后并未改变因子排序，而是将因子原本的scale变成了exp(scale)
                return z_max - z_min
            cmra = self.base_data.stock_price.ix['daily_excess_log_return']*np.nan
            for cursor, date in enumerate(self.base_data.stock_price.ix['daily_excess_log_return'].index):
                # 至少252期才开始计算
                if cursor <= 250:
                    continue
                # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据，且要用对数收益
                curr_data = self.complete_base_data.stock_price.ix['daily_excess_log_return', cursor-251:cursor+1, :]
                temp = func_cmra(curr_data)
                cmra.ix[cursor,:] = temp
        self.base_data.raw_data['cmra'] = cmra
    
    # 计算residual volatility中的hsigma
    def get_rv_hsigma(self):
        if os.path.isfile('hsigma'+self.filename_appendix+'.csv') and not self.is_update and self.try_to_read:
            hsigma = data.read_data(['hsigma'+self.filename_appendix], ['hsigma'])
            hsigma = hsigma.ix['hsigma']
        elif hasattr(self, 'temp_hsigma'):
            hsigma = self.temp_hsigma
        else:
            print('hsigma has not been accquired, if you have rv file stored instead, ingored this message.\n')
            hsigma = np.nan
        self.base_data.raw_data['hsigma'] = hsigma
    
    # 计算residual volatility
    def get_residual_volatility(self):
        if os.path.isfile('rv'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            rv = data.read_data(['rv'+self.filename_appendix], ['rv'])
            self.base_data.factor['rv'] = rv.ix['rv']
        else:
            self.get_rv_dastd()
            self.get_rv_cmra()
            self.get_rv_hsigma()
            # 过滤数据，因为1.此前的3个部分的因子均使用了过去时间点的数据, 且使用了完整版数据,
            # 2.因子数据之后要正交化
            self.base_data.discard_uninv_data()
            # 计算三个成分因子的暴露
            self.base_data.raw_data['dastd_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['dastd'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['cmra_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['cmra'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['hsigma_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['hsigma'], self.base_data.stock_price.ix['FreeMarketValue'])
            
            rv = 0.74*self.base_data.raw_data.ix['dastd_expo']+0.16*self.base_data.raw_data.ix['cmra_expo']+ \
                                                        0.1*self.base_data.raw_data.ix['hsigma_expo']
            # 计算rv的因子暴露，不再去极值
            y = strategy_data.get_cap_wgt_exposure(rv, self.base_data.stock_price.ix['FreeMarketValue'], percentile=0)
            # 计算市值因子与beta因子的暴露
            x = pd.Panel({'lncap_expo':strategy_data.get_cap_wgt_exposure(self.base_data.factor.ix['lncap'],
                                                                          self.base_data.stock_price.ix['FreeMarketValue']),
                          'beta_expo':strategy_data.get_cap_wgt_exposure(self.base_data.factor.ix['beta'],
                                                                         self.base_data.stock_price.ix['FreeMarketValue'])})
            # 正交化
            new_rv = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.base_data.stock_price.ix['FreeMarketValue']))[0]
            # 之后会再次的计算暴露，注意再次计算暴露后，new_rv依然保有对x的正交性
            self.base_data.factor['rv'] = new_rv
                           
    # 计算nonlinear size
    def get_nonlinear_size(self):
        if os.path.isfile('nls'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            nls = data.read_data(['nls'+self.filename_appendix], ['nls'])
            self.base_data.factor['nls'] = nls.ix['nls']
        else:
            # 计算市值因子的暴露，注意解释变量需要为一个panel
            x = pd.Panel({'lncap_expo': strategy_data.get_cap_wgt_exposure(self.base_data.factor.ix['lncap'],
                                                                           self.base_data.stock_price.ix['FreeMarketValue'])})
            # 将市值因子暴露取3次方, 得到size cube
            y = x['lncap_expo'] ** 3
            # 将size cube对市值因子做正交化
            new_nls = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.base_data.stock_price.ix['FreeMarketValue']))[0]
            self.base_data.factor['nls'] = new_nls

    # 计算pb
    def get_pb(self):
        if os.path.isfile('bp'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            pb = data.read_data(['bp'+self.filename_appendix], ['bp'])
            self.base_data.factor['bp'] = pb.ix['bp']
        else:
            self.base_data.factor['bp'] = 1/self.base_data.raw_data.ix['PB']

    
    # 计算liquidity中的stom
    def get_liq_stom(self):
        if os.path.isfile('stom'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            stom = data.read_data(['stom'+self.filename_appendix], ['stom'])
            stom = stom.ix['stom']
        else:
            # 注意, 这里的股票交易数据因为要用过去一段时间的数据, 因此要用完整的数据
            v2s = self.complete_base_data.stock_price.ix['Volume'].div(
                self.complete_base_data.stock_price.ix['FreeShares'])
            stom = v2s.rolling(21, min_periods=5).apply(lambda x:np.log(np.sum(x)))
        self.base_data.raw_data['stom'] = stom
        
    # 计算liquidity中的stoq
    def get_liq_stoq(self):
        if os.path.isfile('stoq'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            stoq = data.read_data(['stoq'+self.filename_appendix], ['stoq'])
            stoq = stoq.ix['stoq']
        else:
            # 定义stoq的函数
            def func_stoq(df):
                # 去过去3个月的stom
                months = np.arange(20,63,21)
                months_stom = df.ix[months]
                return np.log(np.exp(months_stom).mean(axis=0))
            stoq = self.base_data.stock_price.ix['daily_excess_log_return']*np.nan
            for cursor, date in enumerate(self.base_data.stock_price.ix['daily_excess_log_return'].index):
                # 至少63期才开始计算
                if cursor<=61:
                    continue
                curr_data = self.base_data.raw_data.ix['stom', cursor-62:cursor+1,:]
                temp = func_stoq(curr_data)
                stoq.ix[cursor,:] = temp
        self.base_data.raw_data['stoq'] = stoq

    # 计算liquidity中的stoa
    def get_liq_stoa(self):
        if os.path.isfile('stoa'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            stoa = data.read_data(['stoa'+self.filename_appendix], ['stoa'])
            stoa = stoa.ix['stoa']
        else:
            # 定义stoa的函数
            def func_stoa(df):
                # 去过去12个月的stom
                months = np.arange(20,252,21)
                months_stom = df.ix[months]
                return np.log(np.exp(months_stom).mean(axis=0))
            stoa = self.base_data.stock_price.ix['daily_excess_log_return']*np.nan
            for cursor, date in enumerate(self.base_data.stock_price.ix['daily_excess_log_return'].index):
                # 至少252期才开始计算
                if cursor<=250:
                    continue
                curr_data = self.base_data.raw_data.ix['stom', cursor-251:cursor+1,:]
                temp = func_stoa(curr_data)
                stoa.ix[cursor,:] = temp
        self.base_data.raw_data['stoa'] = stoa

    # 计算liquidity
    def get_liquidity(self):
        if os.path.isfile('liquidity'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            liquidity = data.read_data(['liquidity'+self.filename_appendix], ['liquidity'])
            self.base_data.factor['liquidity'] = liquidity.ix['liquidity']
        else:
            self.get_liq_stom()
            self.get_liq_stoq()
            self.get_liq_stoa()
            # 过滤数据, 理由同rv中一样
            self.base_data.discard_uninv_data()
            # 计算三个成分因子的暴露
            self.base_data.raw_data['stom_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['stom'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['stoq_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['stoq'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['stoa_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.base_data.raw_data.ix['stoa'], self.base_data.stock_price.ix['FreeMarketValue'])
            
            liquidity = 0.35*self.base_data.raw_data.ix['stom_expo']+0.35*self.base_data.raw_data.ix['stoq_expo']+ \
                                                              0.3*self.base_data.raw_data.ix['stoa_expo']
            # 计算liquidity的因子暴露，不再去极值
            y = strategy_data.get_cap_wgt_exposure(liquidity, self.base_data.stock_price.ix['FreeMarketValue'], percentile=0)
            # 计算市值因子的暴露
            x = pd.Panel({'lncap_expo': strategy_data.get_cap_wgt_exposure(self.base_data.factor.ix['lncap'],
                                                                           self.base_data.stock_price.ix['FreeMarketValue'])})
            # 正交化
            new_liq = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.base_data.stock_price.ix['FreeMarketValue']))[0]
            self.base_data.factor['liquidity'] = new_liq

    # 计算earnings yield中的epfwd
    def get_ey_epfwd(self):
        if os.path.isfile('epfwd'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            epfwd = data.read_data(['epfwd'+self.filename_appendix], ['epfwd'])
            epfwd = epfwd.ix['epfwd']
        else:
            # 定义计算epfwd的函数
            def epfwd_func(fy1_data, fy2_data):
                # 获取当前的月份数
                curr_month = fy1_data.index.month
                # 获取fy1数据与fy2数据的权重，注意：财年是以4月份结束的
                # 因此5月份时，全部用fy1数据，其权重为1，fy2权重为0
                # 4月份时，fy1权重为1/12， fy2权重为11/12
                # 6月份时，fy1权重为11/12，fy2权重为1/12
                # 当前月份与5月的差距
                diff_month = curr_month-5
                fy1_weight = np.where(diff_month>=0, (12-diff_month)/12, -diff_month/12)
                # fy1_weight为一个ndarray，将它改为series
                fy1_weight = pd.Series(fy1_weight, index=fy1_data.index)
                fy2_weight = 1-fy1_weight
                return (fy1_data.mul(fy1_weight, axis=0) + fy2_data.mul(fy2_weight, axis=0))
            # 用预测的净利润数据除以市值数据得到预测的ep
            ep_fy1 = self.base_data.raw_data.ix['NetIncome_fy1']/self.base_data.stock_price.ix['MarketValue']
            ep_fy2 = self.base_data.raw_data.ix['NetIncome_fy2']/self.base_data.stock_price.ix['MarketValue']
            epfwd = epfwd_func(ep_fy1, ep_fy2)
        self.base_data.raw_data['epfwd'] = epfwd
            
    # 计算earnings yield中的cetop
    def get_ey_cetop(self):
        if os.path.isfile('cetop'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            cetop = data.read_data(['cetop'+self.filename_appendix], ['cetop'])
            cetop = cetop.ix['cetop']
        else:
            # 用cash earnings ttm 除以市值
            cetop = self.base_data.raw_data.ix['CashEarnings_ttm']/self.base_data.stock_price.ix['MarketValue']
        self.base_data.raw_data['cetop'] = cetop
        
    # 计算earnings yield中的etop
    def get_ey_etop(self):
        if os.path.isfile('etop'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            etop = data.read_data(['etop'+self.filename_appendix], ['etop'])
            etop = etop.ix['etop']
        else:
            # 用pe_ttm的倒数来计算etop
            etop = 1/self.base_data.raw_data.ix['PE_ttm']
        self.base_data.raw_data['etop'] = etop

    # 计算earnings yield
    def get_earnings_yeild(self):
        if os.path.isfile('ey'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            EarningsYield = data.read_data(['ey'+self.filename_appendix], ['ey'])
            self.base_data.factor['ey'] = EarningsYield.ix['ey']
        else:
            self.get_ey_epfwd()
            self.get_ey_cetop()
            self.get_ey_etop()
            # 过滤数据
            self.base_data.discard_uninv_data()
            # 计算三个成分因子的暴露
            self.base_data.raw_data['epfwd_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['epfwd'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['cetop_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['cetop'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['etop_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['etop'], self.base_data.stock_price.ix['FreeMarketValue'])

            EarningsYield = 0.68*self.base_data.raw_data.ix['epfwd_expo']+0.21*self.base_data.raw_data.ix['cetop_expo']+ \
                                0.11*self.base_data.raw_data.ix['etop_expo']
            self.base_data.factor['ey'] = EarningsYield

    # 计算growth中的egrlf
    def get_g_egrlf(self):
        if os.path.isfile('egrlf'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            egrlf = data.read_data(['egrlf'+self.filename_appendix], ['egrlf'])
            egrlf = egrlf.ix['egrlf']
        else:
            # 用ni_fy2来代替长期预测的净利润
            egrlf = (self.base_data.raw_data.ix['NetIncome_fy2']/self.base_data.raw_data.ix['NetIncome_ttm'])**(1/2) - 1
        self.base_data.raw_data['egrlf'] = egrlf

    # 计算growth中的egrsf
    def get_g_egrsf(self):
        if os.path.isfile('egrsf'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            egrsf = data.read_data(['egrsf'+self.filename_appendix], ['egrsf'])
            egrsf = egrsf.ix['egrsf']
        else:
            # 用ni_fy1来代替短期预测净利润
            egrsf = self.base_data.raw_data.ix['NetIncome_fy1'] / self.base_data.raw_data.ix['NetIncome_ttm'] - 1
        self.base_data.raw_data['egrsf'] = egrsf

    # 计算growth中的egro
    def get_g_egro(self):
        if os.path.isfile('egro'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            egro = data.read_data(['egro'+self.filename_appendix], ['egro'])
            egro = egro.ix['egro']
        else:
            # 用ni ttm的两年增长率代替ni ttm的5年增长率
            egro = self.base_data.raw_data.ix['NetIncome_ttm_growth_8q']
        self.base_data.raw_data['egro'] = egro

    # 计算growth中的sgro
    def get_g_sgro(self):
        if os.path.isfile('sgro'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            sgro = data.read_data(['sgro'+self.filename_appendix], ['sgro'])
            sgro = sgro.ix['sgro']
        else:
            # 用历史营业收入代替历史sales per share
            sgro = self.base_data.raw_data.ix['Revenue_ttm_growth_8q']
        self.base_data.raw_data['sgro'] = sgro

    # 计算growth
    def get_growth(self):
        if os.path.isfile('growth'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            growth = data.read_data(['growth'+self.filename_appendix], ['growth'])
            self.base_data.factor['growth'] = growth.ix['growth']
        else:
            self.get_g_egrlf()
            self.get_g_egrsf()
            self.get_g_egro()
            self.get_g_sgro()
            # 过滤数据
            self.base_data.discard_uninv_data()
            # 计算四个成分因子的暴露
            self.base_data.raw_data['egrlf_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['egrlf'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['egrsf_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['egrsf'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['egro_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['egro'], self.base_data.stock_price.ix['FreeMarketValue'])
            self.base_data.raw_data['sgro_expo'] = strategy_data.get_cap_wgt_exposure(
                self.base_data.raw_data.ix['sgro'], self.base_data.stock_price.ix['FreeMarketValue'])

            growth = 0.18*self.base_data.raw_data.ix['egrlf_expo']+0.11*self.base_data.raw_data.ix['egrsf_expo']+ \
                             0.24*self.base_data.raw_data.ix['egro_expo']+0.47*self.base_data.raw_data.ix['sgro_expo']
            self.base_data.factor['growth'] = growth

    # 计算leverage
    def get_leverage(self):
        if os.path.isfile('leverage'+self.filename_appendix+'.csv') \
                and not self.is_update and self.try_to_read:
            leverage = data.read_data(['leverage'+self.filename_appendix], ['leverage'])
            self.base_data.factor['leverage'] = leverage.ix['leverage']
        else:
            # 用简单的资产负债率计算leverage
            leverage = self.base_data.raw_data.ix['TotalLiability']/self.base_data.raw_data.ix['TotalAssets']
            self.base_data.factor['leverage'] = leverage

    # 计算风格因子的因子暴露
    def get_style_factor_exposure(self):
        # 给因子暴露panel加上索引
        self.base_data.factor_expo = pd.Panel(data=None, major_axis=self.base_data.factor.major_axis,
                                            minor_axis=self.base_data.factor.minor_axis)
        # 循环计算暴露
        for item, df in self.base_data.factor.iteritems():
            # 通过内部因子加总得到的因子，或已经计算过一次暴露的因子（如正交化过），不再需要去极值
            if item in ['rv', 'nls', 'liquidity', 'ey', 'growth']:
                self.base_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.base_data.stock_price.ix['FreeMarketValue'], percentile=0)
            else:
                self.base_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.base_data.stock_price.ix['FreeMarketValue'])

    # 得到行业因子的虚拟变量
    def get_industry_factor(self):
        # 读取行业信息数据
        if self.industry.empty:
            industry = data.read_data(['Industry'],['Industry'])
            self.industry = industry.ix['Industry']
        # 对第一个拥有所有行业的日期取虚拟变量，以建立储存数据的panel
        industry_num = self.industry.apply(lambda x:x.unique().size, axis=1)
        # 注意所有行业28个，加上nan有29个
        first_valid_index = industry_num[industry_num==29].index[0]
        temp_dum = pd.get_dummies(self.industry.ix[first_valid_index], prefix='Industry')
        industry_dummies = pd.Panel(data=None, major_axis = temp_dum.index, minor_axis = temp_dum.columns)
        # 开始循环
        for time, ind_data in self.industry.iterrows():
            industry_dummies[time] = pd.get_dummies(ind_data, prefix='Industry')
        # 转置
        industry_dummies = industry_dummies.transpose(2,0,1)
        # 将行业因子暴露与风格因子暴露的索引对其
        industry_dummies = data.align_index(self.base_data.factor_expo.ix[0], industry_dummies)
        # 将nan填成0，主要是有些行业在某一时间点，没有一只股票属于它，这会造成在这个行业上的暴露是nan
        # 因此需要把这个行业的暴露填成0，而uninv的nan同样会被填上，但会在之后的filter中再次变成nan
        industry_dummies = industry_dummies.fillna(0)
        # 将行业因子暴露与风格因子暴露衔接在一起
        self.base_data.factor_expo = pd.concat([self.base_data.factor_expo, industry_dummies])
        
    # 加入国家因子，也即回归中用到的截距项
    def add_country_factor(self):
        # 给items中的最后加上截距项，即barra里的country factor
        constant = pd.DataFrame(1, index=self.base_data.factor_expo.major_axis,
                                columns=self.base_data.factor_expo.minor_axis)
        constant = constant.astype(float)
        constant.name = 'country_factor'
        self.base_data.factor_expo = pd.concat([self.base_data.factor_expo, constant])

    # 在完成全部因子暴露的计算或读取后, 得出该base中风格因子和行业因子的数量
    def get_factor_group_count(self):
        # 注意, 默认的排序是, 先排所有风格因子, 然后是行业因子, 最后是一个国家因子
        # 先判断行业因子, 行业因子一定是以industry开头的
        items = self.base_data.factor_expo.items
        industry = items.str.startswith('Industry')
        self.n_indus = industry[industry].size
        # 于是风格因子的数量为总数量减去行业因子数量, 再减去1(country factor)
        self.n_style = items.size - self.n_indus - 1

    # 创建因子值读取文件名的函数, 一般来说, 文件名与股票池相对应
    # 但是, 如在多因子研究框架中说到的那样, 为了增强灵活性, 用户可以选择在当前股票池设定下,
    # 读取在其他股票池中计算出的因子值, 然后将这些因子值在股票池内标准化.
    # 只要将在不同股票池中算出的原始因子值, 理解为定义不同的因子就可以了.
    # 这个功能一般很少用到, 暂时放在这里增加灵活性
    def construct_reading_file_appendix(self, *, filename_appendix='default'):
        # 默认就是用到当前的股票池
        if filename_appendix == 'default':
            self.filename_appendix = '_' + self.base_data.stock_pool
        # 有可能会使用到与当前股票池不同的股票池下计算出的因子值
        else:
            self.filename_appendix = filename_appendix
            # 需要输出提示用户, 因为这个改动比较重要, 注意, 这个改动只会影响读取的因子值,
            # 不会影响算出来的因子值, 算出来的因子值还是与当前股票池一样
            print('Attention: The stock pool you specify under which the base factors are calculated '
                  'is different from your base stock pool. Please aware that this will make the definition'
                  'of your base factors different. Also note that this will only affect factors read from'
                  'files, factors which are calculated will not be affected! \n')


    # 构建barra base的所有风格因子和行业因子
    def construct_factor_base(self, *, if_save=False):
        # 读取数据，更新数据则不用读取，因为已经存在
        if not self.is_update:
            self.read_original_data()
        # 构建风格因子前, 要设置读取文件的名称, 有可能会使用不同股票池下的因子定义
        self.construct_reading_file_appendix()
        # 创建风格因子
        self.get_lncap()
        print('get lncap completed...\n')
        self.get_beta_parallel()
        print('get beta completed...\n')
        self.get_momentum()
        print('get momentum completed...\n')
        self.get_residual_volatility()
        print('get rv completed...\n')
        self.get_nonlinear_size()
        print('get nls completed...\n')
        self.get_pb()
        print('get pb completed...\n')
        self.get_liquidity()
        print('get liquidity completed...\n')
        self.get_earnings_yeild()
        print('get ey completed...\n')
        self.get_growth()
        print('get growth completed...\n')
        self.get_leverage()
        print('get leverage completed...\n')
        # 计算风格因子暴露之前再过滤一次
        self.base_data.discard_uninv_data()
        # 计算风格因子暴露
        self.get_style_factor_exposure()
        print('get style factor expo completed...\n')
        # 加入行业暴露
        self.get_industry_factor()
        print('get indus factor expo completed...\n')
        # 添加国家因子
        self.add_country_factor()
        # 计算的最后，过滤数据
        self.base_data.discard_uninv_data()

        # 判定风格因子和行业因子的数量
        self.get_factor_group_count()

        # 如果显示指定了储存数据且股票池为所有股票，则储存因子值数据
        if not self.is_update and if_save:
            # 如果要储存因子数据, 必须保证当前股票池和文件后缀是完全一致的, 否则报错
            assert self.filename_appendix[1:] == self.base_data.stock_pool, 'Error: The stock ' \
                'pool of base is different from filename appendix, in order to avoid possible ' \
                'data loss, the saving procedure has been terminated! \n'
            # 根据股票池, 生成要储存的文件名
            written_filename = []
            for i in self.base_data.factor.items:
                written_filename.append(i+self.filename_appendix)
            data.write_data(self.base_data.factor, file_name=written_filename)
            print('Style factor data have been saved!\n')
            # 储存因子暴露
            self.base_data.factor_expo.to_hdf('bb_factor_expo' + self.filename_appendix, '123')
            print('factor exposure data has been saved!\n')

    # # 仅计算barra base的因子值，主要用于对于不同股票池，可以率先建立一个只有因子值而没有暴露的bb对象
    # def just_get_sytle_factor(self):
    #     # 读取数据，更新数据则不用读取，因为已经存在
    #     if not self.is_update:
    #         self.read_original_data()
    #     # 创建风格因子
    #     self.get_lncap()
    #     self.get_beta_parallel()
    #     print('get beta completed...\n')
    #     self.get_momentum()
    #     print('get momentum completed...\n')
    #     self.get_residual_volatility()
    #     print('get rv completed...\n')
    #     self.get_nonlinear_size()
    #     print('get nls completed...\n')
    #     self.get_pb()
    #     self.get_liquidity()
    #     print('get liquidity completed...\n')
    #     self.get_earnings_yeild()
    #     print('get ey completed...\n')
    #     self.get_growth()
    #     self.get_leverage()
    #     print('get leverage completed...\n')
    #     # 计算风格因子暴露之前再过滤一次
    #     self.base_data.discard_uninv_data()

    # # 仅计算barra base的因子暴露，主要用于对与不同股票池，可以在不重新建立新对象的情况下，根据已有因子值算不同的因子暴露
    # def just_get_factor_expo(self):
    #     self.base_data.discard_uninv_data()
    #     self.get_style_factor_exposure()
    #     self.get_industry_factor()
    #     self.add_country_factor()
    #     self.base_data.discard_uninv_data()
    #
    #     # 判定风格因子和行业因子的数量
    #     self.get_factor_group_count()

    # 回归计算各个基本因子的因子收益
    def get_base_factor_return(self, *, if_save=False):
        # 初始化储存因子收益的dataframe, 以及股票specific return的dataframe
        self.base_factor_return = pd.DataFrame(np.nan, index=self.base_data.factor_expo.major_axis,
                                             columns=self.base_data.factor_expo.items)
        self.specific_return = pd.DataFrame(np.nan, index=self.base_data.factor_expo.major_axis,
                                            columns=self.base_data.factor_expo.minor_axis)
        # 因子暴露要用上一期的因子暴露，用来加权的市值要用上一期的市值
        lag_factor_expo = self.base_data.factor_expo.shift(1).reindex(
                          major_axis=self.base_data.factor_expo.major_axis)
        lag_mv = self.base_data.stock_price.ix['FreeMarketValue'].shift(1)
        # 循环回归，计算因子收益
        for time, temp_data in self.base_factor_return.iterrows():
            outcome = strategy_data.constrained_gls_barra_base(
                       self.base_data.stock_price.ix['daily_simple_return', time, :],
                       lag_factor_expo.ix[:, time, :],
                       weights = np.sqrt(lag_mv.ix[time, :]),
                       indus_ret_weights = lag_mv.ix[time, :],
                       n_style=self.n_style, n_indus=self.n_indus)
            self.base_factor_return.ix[time, :] = outcome[0]
            self.specific_return.ix[time, :] = outcome[1]
        print('get bb factor return completed...\n')

        # 如果需要储存, 则储存因子收益数据
        # 如果要储存因子数据, 必须保证当前股票池和文件后缀是完全一致的, 否则报错
        if not self.is_update and if_save:
            assert self.filename_appendix[1:] == self.base_data.stock_pool, 'Error: The stock ' \
            'pool of base is different from filename appendix, in order to avoid possible ' \
            'data loss, the saving procedure has been terminated! \n'

            self.base_factor_return.to_csv('bb_factor_return_'+self.base_data.stock_pool+'.csv',
                                         index_label='datetime', na_rep='NaN', encoding='GB18030')
            self.specific_return.to_csv('bb_specific_return_'+self.base_data.stock_pool+'.csv',
                                        index_label='datetime', na_rep='NaN', encoding='GB18030')
            # self.base_factor_return.to_csv('barra_factor_return_'+self.base_data.stock_pool+'.csv',
            #                              index_label='datetime', na_rep='NaN', encoding='GB18030')
            # self.specific_return.to_csv('barra_specific_return_'+self.base_data.stock_pool+'.csv',
            #                             index_label='datetime', na_rep='NaN', encoding='GB18030')
            print('The bb factor return has been saved! \n')


    # 更新数据
    def update_factor_base_data(self, *, start_date=None):
        self.is_update = True
        # 更新的时候不允许读取数据
        self.try_to_read = False
        # 首先读取原始数据
        self.read_original_data()
        # 根据此次更新的股票池, 设置读取文件的文件名, 注意, 更新数据的时候, 文件名强制要求为当前股票池
        self.construct_reading_file_appendix(filename_appendix='default')
        # 读取旧的因子数据
        original_old_base_factor_names = ['lncap', 'beta', 'momentum', 'rv', 'nls', 'bp', 'liquidity',
                                          'ey', 'growth', 'leverage']
        # 加上文件后缀
        old_base_factor_names = []
        for i in original_old_base_factor_names:
            old_base_factor_names.append(i+self.filename_appendix)
        old_base_factors = data.read_data(old_base_factor_names, original_old_base_factor_names)
        # 读取旧的因子暴露数据和因子收益数据
        old_factor_expo = pd.read_hdf('bb_factor_expo'+self.filename_appendix, '123')
        old_factor_return = data.read_data(['bb_factor_return'+self.filename_appendix]).iloc[0]
        old_specific_return = data.read_data(['bb_specific_return'+self.filename_appendix]).iloc[0]

        # 更新与否取决于原始数据和因子数据，若因子数据的时间轴早于原始数据，则进行更新
        # 这里对比的数据实际是free mv和lncap，因为barra base的计算是以这两个为基准的
        last_day = old_base_factors.major_axis[-1]
        # 如果设置了开始时间, 则开始时间是last day和开始时间最早的那个
        if isinstance(start_date, pd.Timestamp):
            start_date = min(last_day, start_date)
        else:
            start_date = last_day
        if start_date == self.base_data.stock_price.major_axis[-1]:
            print('The barra base factor data have been up-to-date.\n')
            return

        # 找因子数据的开始更新的那一天在原始数据中的对应位置
        start_loc = self.base_data.stock_price.major_axis.get_loc(start_date)
        # 将原始数据截取，截取范围从更新的第一天的（即因子数据的最后一天的）前525天到最后一天
        # 更新前525天的选取是因为t时刻的bb因子值最远需要取到525天前的原始数据，在momentum因子中用到
        new_start_loc = start_loc - 525
        self.base_data.stock_price = self.base_data.stock_price.iloc[:, new_start_loc:, :]
        self.base_data.raw_data = self.base_data.raw_data.iloc[:, new_start_loc:, :]
        self.base_data.if_tradable = self.base_data.if_tradable.iloc[:, new_start_loc:, :]
        # 还要记得截取complete_base_data中的数据
        self.complete_base_data.stock_price = self.complete_base_data.stock_price.iloc[:, new_start_loc:, :]
        self.complete_base_data.raw_data = self.complete_base_data.raw_data.iloc[:, new_start_loc:, :]
        self.complete_base_data.if_tradable = self.complete_base_data.if_tradable.iloc[:, new_start_loc:, :]

        # 开始计算新的因子值
        self.construct_factor_base()
        self.get_base_factor_return()

        # 将旧因子值的股票索引换成新的因子值的股票索引
        old_base_factors = old_base_factors.reindex(minor_axis=self.base_data.factor.minor_axis)
        old_factor_expo = old_factor_expo.reindex(minor_axis=self.base_data.factor_expo.minor_axis)
        old_specific_return = old_specific_return.reindex(columns=self.specific_return.columns)
        # 衔接新旧因子值, 注意, 衔接的时候一定要把last_day的老数据丢弃掉, 只用新的数据
        # 就像database的更新一样, 最后一天的老数据, 会因更新的当时可能数据不全, 有各种问题
        # 因此一定不能用.
        new_factor_data = pd.concat([old_base_factors.drop(last_day, axis=1), self.base_data.factor], axis=1)
        new_factor_expo = pd.concat([old_factor_expo.drop(last_day, axis=1), self.base_data.factor_expo], axis=1)
        new_factor_return = pd.concat([old_factor_return.drop(last_day, axis=0), self.base_factor_return], axis=0)
        new_specific_return = pd.concat([old_specific_return.drop(last_day, axis=0), self.specific_return], axis=0)
        # 因为更新数据需要用到以前的数据的原因, 会导致更新的时候更新数据的起头不会是last_day
        # 且这些数据都是nan, 因此只能要第一个数据, 即当时的老数据, 而丢弃掉那些因更新原因产生的nan的数据
        self.base_data.factor = new_factor_data.groupby(new_factor_data.major_axis).first()
        self.base_data.factor_expo = new_factor_expo.groupby(new_factor_expo.major_axis).first()
        self.base_factor_return = new_factor_return.groupby(new_factor_return.index).first()
        self.specific_return = new_specific_return.groupby(new_specific_return.index).first()
        # 储存因子值数据, 因为这里的因子顺序与定义的排序不一样, 因此要将items的数据转化为手动定义的顺序, 再进行储存
        self.base_data.factor = self.base_data.factor[original_old_base_factor_names]
        data.write_data(self.base_data.factor, file_name=old_base_factor_names)
        self.base_data.factor_expo.to_hdf('bb_factor_expo'+self.filename_appendix, '123')
        self.base_factor_return.to_csv('bb_factor_return_' + self.base_data.stock_pool + '.csv',
                                       index_label='datetime', na_rep='NaN', encoding='GB18030')
        self.specific_return.to_csv('bb_specific_return_' + self.base_data.stock_pool + '.csv',
                                    index_label='datetime', na_rep='NaN', encoding='GB18030')

        self.is_update = False
        self.try_to_read = True


if __name__ == '__main__':
    import time
    # pools = ['all', 'hs300', 'zz500', 'zz800', 'sz50', 'zxb', 'cyb']
    # pools = ['all', 'hs300', 'zz500', 'sz50']
    # for i in pools:
    bb = barra_base()
    bb.base_data.stock_pool = 'zz500'
    # bb.base_data.stock_pool = i
    # bb.try_to_read = False
    # bb.read_original_data()
    bb.construct_factor_base(if_save=False)
    bb.base_data.factor_expo = pd.read_hdf('bb_factor_expo_zz500', '123')
    # bb.base_data.factor_expo = bb.base_data.factor_expo[['CNE5S_SIZE', 'CNE5S_BETA', 'CNE5S_MOMENTUM',
    #     'CNE5S_RESVOL', 'CNE5S_SIZENL', 'CNE5S_BTOP', 'CNE5S_LIQUIDTY', 'CNE5S_EARNYILD', 'CNE5S_GROWTH',
    #     'CNE5S_LEVERAGE', 'CNE5S_AERODEF', 'CNE5S_AIRLINE', 'CNE5S_AUTO', 'CNE5S_BANKS', 'CNE5S_BEV',
    #     'CNE5S_BLDPROD', 'CNE5S_CHEM', 'CNE5S_CNSTENG', 'CNE5S_COMSERV', 'CNE5S_CONMAT', 'CNE5S_CONSSERV',
    #     'CNE5S_DVFININS', 'CNE5S_ELECEQP', 'CNE5S_ENERGY', 'CNE5S_FOODPROD', 'CNE5S_HDWRSEMI', 'CNE5S_HEALTH',
    #     'CNE5S_HOUSEDUR', 'CNE5S_INDCONG', 'CNE5S_LEISLUX', 'CNE5S_MACH', 'CNE5S_MARINE', 'CNE5S_MATERIAL',
    #     'CNE5S_MEDIA', 'CNE5S_MTLMIN', 'CNE5S_PERSPRD', 'CNE5S_RDRLTRAN', 'CNE5S_REALEST', 'CNE5S_RETAIL',
    #     'CNE5S_SOFTWARE', 'CNE5S_TRDDIST', 'CNE5S_UTILITIE', 'CNE5S_COUNTRY']]
    # bb.base_data.factor_expo['CNE5S_COUNTRY'] = bb.base_data.factor_expo['CNE5S_COUNTRY'].fillna(1)
    # bb.base_data.factor_expo.ix['CNE5S_AERODEF':'CNE5S_UTILITIE'].fillna(0, inplace=True)
    # bb.n_indus = 32
    bb.get_base_factor_return(if_save=True)
    # if i in ['all', 'hs300', 'zz500']:
    #     bb.base_data.factor_expo.to_hdf('bb_factorexpo_'+i, '123')
    # bb.base_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
    # bb.base_data.factor_expo = bb.base_data.factor_expo[['CNE5S_SIZE', 'CNE5S_BETA', 'CNE5S_MOMENTUM',
    #     'CNE5S_RESVOL', 'CNE5S_SIZENL', 'CNE5S_BTOP', 'CNE5S_LIQUIDTY', 'CNE5S_EARNYILD', 'CNE5S_GROWTH',
    #     'CNE5S_LEVERAGE', 'CNE5S_AERODEF', 'CNE5S_AIRLINE', 'CNE5S_AUTO', 'CNE5S_BANKS', 'CNE5S_BEV',
    #     'CNE5S_BLDPROD', 'CNE5S_CHEM', 'CNE5S_CNSTENG', 'CNE5S_COMSERV', 'CNE5S_CONMAT', 'CNE5S_CONSSERV',
    #     'CNE5S_DVFININS', 'CNE5S_ELECEQP', 'CNE5S_ENERGY', 'CNE5S_FOODPROD', 'CNE5S_HDWRSEMI', 'CNE5S_HEALTH',
    #     'CNE5S_HOUSEDUR', 'CNE5S_INDCONG', 'CNE5S_LEISLUX', 'CNE5S_MACH', 'CNE5S_MARINE', 'CNE5S_MATERIAL',
    #     'CNE5S_MEDIA', 'CNE5S_MTLMIN', 'CNE5S_PERSPRD', 'CNE5S_RDRLTRAN', 'CNE5S_REALEST', 'CNE5S_RETAIL',
    #     'CNE5S_SOFTWARE', 'CNE5S_TRDDIST', 'CNE5S_UTILITIE', 'CNE5S_COUNTRY']]
    # bb.base_data.factor_expo['CNE5S_COUNTRY'] = bb.base_data.factor_expo['CNE5S_COUNTRY'].fillna(1)
    # bb.base_data.factor_expo.ix['CNE5S_AERODEF':'CNE5S_UTILITIE'].fillna(0, inplace=True)
    # bb.base_factor_return = data.read_data(['bb_factor_return_zz500']).iloc[0]
    # bb.base_factor_return = bb.base_factor_return.reindex(columns=bb.base_data.factor_expo.items)
    # bb.specific_return = data.read_data(['bb_specific_return_zz500']).iloc[0]
    # bb.base_factor_return = pd.read_hdf('barra_real_fac_ret', '123')
    # bb.base_factor_return = bb.base_factor_return.reindex(columns=bb.base_data.factor_expo.items)
    # bb.specific_return = pd.read_hdf('barra_real_spec_ret_new', '123')
    # bb.initial_cov_mat = pd.read_hdf('bb_factor_vracovmat_all', '123')
    # bb.daily_var_forecast = pd.read_hdf('bb_factor_var_hs300', '123')
    # bb.get_initial_cov_mat()
    # bb.get_initial_cov_mat_parallel()
    # bb.get_vra_cov_mat()
    # bb.get_eigen_adjusted_cov_mat_parallel(n_of_sims=100, simed_sample_size=504, scaling_factor=3)
    # bb.get_initial_spec_vol()
    # bb.get_initial_spec_vol_parallel()
    # bb.get_vra_spec_vol()
    # bb.get_bayesian_shrinkage_spec_vol(shrinkage_parameter=0.25)
    # start_time = time.time()
    # bb.base_data.stock_price = data.read_data(['FreeMarketValue'])
    # bb.construct_risk_forecast_parallel(eigen_adj_sims=1000)
    # bb.handle_barra_data()
    # bb.risk_forecast_performance_parallel(test_type='random', bias_type=2, freq='w')
    # bb.risk_forecast_performance_parallel_spec(test_type='stock', bias_type=1, cap_weighted_bias=False)
    # bb.risk_forecast_performance_total_parallel(test_type='random', bias_type=2, freq='w')
    # bb.update_factor_base_data()
    # print("time: {0} seconds\n".format(time.time()-start_time))
    pass



























