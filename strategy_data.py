#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:06:35 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
from cvxopt import solvers, matrix
from pandas.stats.fama_macbeth import fama_macbeth # deprecated in version 0.20.2

from data import data
from position import position

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 多因子策略数据类
class strategy_data(data):
    """ This is the multi_factor strategy data class.
    
    stock_price (pd.Panel): price data of stocks
    benchmark_price (pd.Panel): price data of benchmarks
    raw_data (pd.Panel): original data get from market or financial report, or intermediate data
                         which is used for factor calculation, note the difference between stock_price
                         data and raw_data
    factor (pd.Panel): final factors calculated which is used during process of stock selection
    factor_expo(pd.Pnael): factor exposure after standardization
    stock_pool(pd.DataFrame): stock pool to select stocks from
    """
    def __init__(self):
        data.__init__(self)
        self.factor = pd.Panel()
        self.factor_expo = pd.Panel()
        # 股票池，即策略选取的股票池，或各因子数据计算时用到的股票池
        # 目前对股票池的处理方法是将其归为不可交易，用discard_untradable_data来将股票池外的数据设为nan
        self.stock_pool = 'all'

    # 新建一个dataframe储存股票是否在股票池内，再建一个dataframe和if_tradable取交集
    def handle_stock_pool(self, *, shift=False):
        # 如果未设置股票池
        if self.stock_pool == 'all':
            self.if_tradable['if_inpool'] = True
        # 设置了股票池，若已存在benchmark中的weight，则直接使用
        elif 'Weight_'+self.stock_pool in self.benchmark_price.items:
            self.if_tradable['if_inpool'] = self.benchmark_price.ix['Weight_'+self.stock_pool]>0
        # 若不在，则读取weight数据，文件名即为stock_pool
        else:
            temp_weights = data.read_data(['Weight_'+self.stock_pool],['Weight_'+self.stock_pool],
                                          shift=shift).fillna(0.0)
            if self.benchmark_price.empty:
                self.benchmark_price = temp_weights
            else:
                self.benchmark_price['Weight_'+self.stock_pool] = temp_weights['Weight_'+self.stock_pool]
            # 由于指数权重数据会跟1有一点点偏离, 因此要将其归一化
            self.benchmark_price['Weight_'+self.stock_pool] = self.benchmark_price['Weight_'+self.stock_pool]. \
                apply(position.to_percentage_func, axis=1)
            # 指数权重大于0的股票, 即为在指数内的股票
            self.if_tradable['if_inpool'] = self.benchmark_price.ix['Weight_'+self.stock_pool]>0

        # 若还没有if_tradable，则生成if_tradable
        if 'if_tradable' not in self.if_tradable.items:
            self.generate_if_tradable(shift=shift)

        # 新建一个if_inv，表明在股票池中，且可以交易
        # 在if_tradable中为true，且在if_inpool中为true，才可投资，即在if_inv中为true
        self.if_tradable['if_inv'] = np.logical_and(self.if_tradable.ix['if_tradable'], self.if_tradable.ix['if_inpool'])

    # 对数据进行winsorization
    @staticmethod
    def winsorization(raw_data, *, percentile=0.01):
        """ Winsorize the data.
        
        raw_data (pd.DataFrame): data you'd like to winsorize
        percentile: percentile on which data will be winsorized.
        """
        temp = raw_data * 1
        lower_q = raw_data.quantile(percentile, 1)[:, np.newaxis]
        upper_q = raw_data.quantile(1-percentile, 1)[:, np.newaxis]
        raw_data = np.where(np.greater(raw_data, lower_q), raw_data, lower_q)
        raw_data = np.where(np.less(raw_data, upper_q), raw_data, upper_q)
        # np.greater, np.less遇到nan时都返回false，因此要将nan的数据改为nan
        raw_data = np.where(temp.isnull(), np.nan, raw_data)
        return pd.DataFrame(raw_data, index=temp.index, columns=temp.columns)
            
    # 对数据进行zscore
    @staticmethod
    def zscore(raw_data):
        """ Get z-score of a series of data.
        
        raw_data (pd.DataFrame): data you'd like to get z-score from
        """
        mu = raw_data.mean(1)
        sigma = raw_data.std(1)
        raw_data = raw_data.sub(mu, axis=0).div(sigma, axis=0)
        return raw_data

    # 对数据进行zscore，均值用市值进行加权，标准差还是简单加权
    @staticmethod
    def cap_wgt_zscore(raw_data, mv):
        cap_wgt_mu = raw_data.mul(mv).div(mv.sum(1), axis=0).sum(1)
        sigma = raw_data.std(1)
        raw_data = raw_data.sub(cap_wgt_mu, axis=0).div(sigma, axis=0)
        return raw_data

    # 对数据进行rescale，将尾部分布进行压缩，方法参考eue3
    # 将大于3 sigma的数据部分压缩到一个限制范围内，默认为3.5
    # 注意输入数据为z score后的数据
    @staticmethod
    def compress_tail_data(raw_data, *, limit=3.5):
        # 首先计算数据的均值，进行平移，使得新数据均值为0
        # 这是因为按照barra的zscore方法做出来的数据均值并不是0，标准差一定是1
        mu = raw_data.mean(1)
        centered = raw_data.sub(mu, axis=0)
        # 按照eue3的方法进行compress
        data_max = centered.max(axis=1)
        data_min = centered.min(axis=1)
        s_plus = np.maximum(np.zeros_like(data_max), np.minimum(np.ones_like(data_max), (limit-3)/(data_max-3)))
        s_minus = np.maximum(np.zeros_like(data_min), np.minimum(np.ones_like(data_min), (3-limit)/(data_min+3)))
        centered = np.where(np.less(centered, 3), centered, (centered-3).mul(s_plus, axis=0)+3)
        centered = pd.DataFrame(centered, index=raw_data.index, columns=raw_data.columns)
        centered = np.where(np.greater(centered, -3), centered, (centered+3).mul(s_minus, axis=0)-3)
        centered = pd.DataFrame(centered, index=raw_data.index, columns=raw_data.columns)
        compressed = centered.add(mu, axis=0)
        return compressed

    # 计算因子暴露，简单加权
    @staticmethod
    def get_exposure(factor, *, percentile=0.01, compress=True, limit=3.5):
        temp_data = strategy_data.winsorization(factor, percentile = percentile)
        # 如有需要，对尾部数据进行压缩
        if compress:
            # 先标准化
            temp_data = strategy_data.zscore(temp_data)
            # 进行压缩
            temp_data = strategy_data.compress_tail_data(temp_data, limit=limit)
        # 进行标准化
        final_data = strategy_data.zscore(temp_data)
        return final_data
    
    # 计算市值加权的因子暴露
    @staticmethod
    def get_cap_wgt_exposure(factor, mv, *, percentile=0.01, compress=True, limit=3.5):
        temp_data = strategy_data.winsorization(factor, percentile = percentile)
        # 如有需要，对尾部数据进行压缩
        if compress:
            # 先标准化
            temp_data = strategy_data.cap_wgt_zscore(temp_data, mv)
            # 进行压缩
            temp_data = strategy_data.compress_tail_data(temp_data, limit=limit)
        # 进行标准化
        final_data = strategy_data.cap_wgt_zscore(temp_data, mv)
        return final_data
    
    
    # 检查在某一时间，某只股票是否处于可交易状态
    def check_if_tradable(self, time, stock):
        return self.if_tradable.ix['if_tradable', time, stock]  
    
    # 将strategy_data中的所有数据，在不可交易的时候，都设为nan，
    # 在策略中：因为在shift之后，if_tradable是一个选股时的已知信息，
    # 直接利用这个信息，过滤掉调仓日之前不可交易的股票，使得选股策略更加真实有效
    # 需注意，如果在策略中使用，一定要在if_tradable数据shift之后再使用，否则会用到未来信息
    # 如果只是单纯的计算数据（如计算因子），则不需要shift if_tradable数据，因为当天的数据是用当天所有已知信息计算后储存下来的
    def discard_untradable_data(self):
        # 如果没有可交易标记的数据，则什么数据也不丢弃
        if self.if_tradable.ix['if_tradable'].empty:
            return
        
        # 股票价格行情数据
        if not self.stock_price.empty:
            for item, df in self.stock_price.iteritems():
                self.stock_price.ix[item] = self.stock_price.ix[item].where(
                                             self.if_tradable.ix['if_tradable'], np.nan)
        
        # benchmark数据
        if not self.benchmark_price.empty:
            for item, df in self.benchmark_price.iteritems():
                self.benchmark_price.ix[item] = self.benchmark_price.ix[item].where(
                                                 self.if_tradable.ix['if_tradable'], np.nan)
        
        # 原始数据                                          
        if not self.raw_data.empty:
            for item, df in self.raw_data.iteritems():
                self.raw_data.ix[item] = self.raw_data.ix[item].where(
                                          self.if_tradable.ix['if_tradable'], np.nan)
                                               
        # 因子数据
        if not self.factor.empty:
            for item, df in self.factor.iteritems():
                self.factor.ix[item] = self.factor.ix[item].where(
                                        self.if_tradable.ix['if_tradable'], np.nan)
        
        # 因子暴露数据        
        if not self.factor_expo.empty:
            for item, df in self.factor_expo.iteritems():
                self.factor_expo.ix[item] = self.factor_expo.ix[item].where(
                                             self.if_tradable.ix['if_tradable'], np.nan)

    # 与discard_untradable_data一样，只是这里丢弃掉不可投资的数据
    def discard_uninv_data(self):
        # 如果没有可交易标记的数据，则什么数据也不丢弃
        if self.if_tradable.ix['if_inv'].empty:
            return

        # 股票价格行情数据
        if not self.stock_price.empty:
            for item, df in self.stock_price.iteritems():
                self.stock_price.ix[item] = self.stock_price.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # benchmark数据
        if not self.benchmark_price.empty:
            for item, df in self.benchmark_price.iteritems():
                self.benchmark_price.ix[item] = self.benchmark_price.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 原始数据
        if not self.raw_data.empty:
            for item, df in self.raw_data.iteritems():
                self.raw_data.ix[item] = self.raw_data.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 因子数据
        if not self.factor.empty:
            for item, df in self.factor.iteritems():
                self.factor.ix[item] = self.factor.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 因子暴露数据
        if not self.factor_expo.empty:
            for item, df in self.factor_expo.iteritems():
                self.factor_expo.ix[item] = self.factor_expo.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        
    # 对数据进行回归取残差提纯，即gram-schmidt正交化
    @staticmethod
    def simple_orth_gs(obj, base, *, weights=None, add_constant=True):
        # 定义回归函数
        # 注意，用barra base回归时，行业暴露已经包含截距项，因此不能再添加截距项
        def reg_func(y, x, *, weights=1, add_constant=True):
            if add_constant:
                x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y,x], axis=1).dropna().shape[0] <= 1:
                return 'empty'
            model = sm.WLS(y, x, weights=weights, missing='drop')
            results = model.fit()
            return results
        new_obj = obj*np.nan
        pvalues = pd.DataFrame(np.nan, index=obj.index, columns=base.items)
        rsquared = pd.DataFrame(np.nan, index=obj.index, columns=['rsquared', 'rsquared_adj'])
        if weights is None:
            for cursor, date in enumerate(obj.index):
                curr_results = reg_func(obj.ix[cursor], base.ix[:,cursor,:], add_constant=add_constant)
                if type(curr_results) != str:
                    new_obj.ix[cursor] = curr_results.resid.reindex(obj.columns)
                    pvalues.ix[cursor] = curr_results.pvalues
                    rsquared.ix[cursor, 'rsquared'] = curr_results.rsquared
                    rsquared.ix[cursor, 'rsquared_adj'] = curr_results.rsquared_adj
        else:
            for cursor, date in enumerate(obj.index):
                curr_results = reg_func(obj.ix[cursor], base.ix[:,cursor,:], weights=weights.ix[cursor],
                                        add_constant=add_constant)
                if type(curr_results) != str:
                    new_obj.ix[cursor] = curr_results.resid.reindex(obj.columns)
                    pvalues.ix[cursor] = curr_results.pvalues
                    rsquared.ix[cursor, 'rsquared'] = curr_results.rsquared
                    rsquared.ix[cursor, 'rsquared_adj'] = curr_results.rsquared_adj
            # --------------------------------------------------------------------------------------------
            # 实际上不需要做这一步的调整, 因为results.resid并不是进行了回归权重加权的那个residual,
            # 而是直接通过y - (x*b + a)算出来的residual, 因此直接满足在回归权重加权的情况下, 和x正交的性质
            # 因此,其实不需要做出任何的调整, 即计算残差的时候不需要考虑回归权重的问题.

            # # 如果提纯为加权的回归，则默认提纯是为了之后这个残差和base进行加权回归时相互正交
            # # 即：实际为残差和加权（加根号权重）后的base因子正交，那么在之后进行加权回归的时候，会再一次的进行加权
            # # 为了避免残差因子在那个时候连加两次权，这里必须进行调整，即：除以根号权重
            # # 注意除以根号权重是因为最小二乘回归的权重实际为在因子上乘以根号权重
            # new_obj = new_obj.div(np.sqrt(weights))
            # --------------------------------------------------------------------------------------------
        return [new_obj, pvalues, rsquared]
            
    # 用因子暴露数据，回归权重，进行barra模型的回归
    # 用二次规划问题求解此线性回归问题
    # 目前，基于barra的业绩归因、barra基础因子内部回归都可以用这个线性回归模型，暂不支持新增因子
    @staticmethod
    def constrained_gls_barra_base(asset_return, base_expo, *, weights=None, indus_ret_weights=None,
                                   n_style=10, n_indus=28):
        """Solving constrained gls problem using quadratic programming.
        
        asset_return: return of asset universe.
        base_expo: barra base factor exposures, including style factors and industrial factors
        weights: weights of gls, usually the sqrt of mv, default means equal weight
        indus_ret_weights: weights that put on the constraints of industry factors returns, usually as the \
                           market value, default means equal weight
        """
        if weights is None:
            weights = pd.Series(1, index=asset_return.index)
        if indus_ret_weights is None:
            indus_ret_weights = pd.Series(1, index=asset_return.index)
            
        # 设置权重
        # 回归的权重需要开根号
        sqrt_w = np.sqrt(weights)
        y = asset_return.mul(sqrt_w)
        # base_expo中股票为index，因子名字为columns
        x = base_expo.mul(sqrt_w, axis=0)
        
        # 只要有na，就drop掉这只股票
        yx = pd.concat([y,x], axis=1)
        yx = yx.dropna()
        # 如果只有小于等于1个有效数据，返回nan序列
        if yx.shape[0] <= 1:
            return [np.full(n_style + n_indus + 1, np.nan), pd.Series(np.nan, index=x.index)]
        y = yx.ix[:, 0]
        x = yx.ix[:, 1:]

        # 设置行业因子收益限制条件，行业因子暴露为x中的第 11 列到第 38 列
        # 行业因子的限制权重，循环在行业因子中求和
        final_weight = pd.Series(np.arange(n_indus) * 0, index=x.columns[n_style:(n_style+n_indus)])
        for cursor in range(n_style, (n_style+n_indus)):
            final_weight.iloc[cursor - n_style] = (indus_ret_weights * x.ix[:, cursor]).sum()
        final_weight = final_weight / (final_weight.sum())

        # 储存结果的series
        results_s = pd.Series(np.nan, index=x.columns)

        # 移除被删除的风格因子，或股票池中不包含的行业因子
        # 首先，x中暴露全为0的因子，一定是要被删除的
        x = x.replace(0, np.nan).dropna(axis=1, how='all').fillna(0.0)
        # 判断行业因子中有多少要被移除的，只需要看final_weight中有多少权重是0
        final_weight = final_weight.replace(0, np.nan).dropna()
        num_valid_all = x.shape[1]
        num_valid_indus = final_weight.shape[0]
        num_valid_style = num_valid_all - num_valid_indus - 1

        # 如果有效的观测数据个数小于有效因子数, 则优化问题无解, 直接返回nan
        if x.shape[0] < num_valid_all:
            return [np.full(n_style+n_indus+1, np.nan), pd.Series(np.nan, index=x.index)]

        # 设置行业因子收益的加权求和限制为0
        indus_cons = pd.Series(np.arange(num_valid_all) * 0)
        # 系数的第11到38项设置为行业因子收益的权重，注意如果有被移除的行业因子，则需要对应调整
        indus_cons.iloc[num_valid_style:num_valid_all-1] = final_weight.values

        # 开始设置优化
        # P = X.T dot X
        P = matrix(np.dot(x.as_matrix().transpose(), x.as_matrix()))
        # q = - (X.T dot Y)
        q = matrix(-np.dot(x.as_matrix().transpose(), y.as_matrix()))

        # 设置限制条件
        A = matrix(indus_cons.as_matrix(), (1, indus_cons.size))
        b = matrix(0.0)
        
        # 隐藏优化器输出
        solvers.options['show_progress'] = False
        # 解优化问题
        results = solvers.qp(P=P, q=q, A=A, b=b)
        # 将数据类型改为(n,)的ndarray
        results_np = np.array(results['x']).squeeze()

        # 将结果对应到相应的因子上，构成结果的series
        results_s[x.columns] = results_np
        # 不存在的因子收益，可以认为它的收益是0
        results_s = results_s.fillna(0)

        # 计算残余收益, 注意: 这里的残余收益是指没有被base模型解释的收益率部分,
        # 即, 用y - (x*b + a) 直接算出的残差, 这个残差与回归权重没有关系, 就代表的是没有被回归模型解释的部分
        # statsmodels里的results.resid返回的也是这个残差, 特别注意这样一点, 计算残差的时候与回归权重无关.
        residual_return = asset_return.reindex(index=y.index) - base_expo.reindex(index=x.index,
                            columns=x.columns).dot(results_np)
        # 将残余收益的index改为asset return的index(所有传入的股票), 而不是残余回归的有效股票index
        residual_return = residual_return.reindex(index=asset_return.index)
        
        return [results_s, residual_return]



    # 计算净利润增长率的函数, 因为净利润增长, 涉及净利润是负数的情况比较麻烦, 所以专门写一个函数来计算
    # lag为传入序列要计算的成长率, 1则是计算每隔一期的成长, 2则是每隔两期(注意这样会少一些数据)
    # annualize_term为年化增长率的参数, 若增长率为每天, 则参数为1/252, 若为2年, 则为2
    @staticmethod
    def get_ni_growth(ni_data, *, lag=1, annualize_term=0):
        former = ni_data.shift(lag)
        latter = ni_data * 1
        final_growth = ni_data * np.nan
        # 前后都大于0的情况, 直接算增长率
        growth = np.where(np.logical_and(latter>0, former>0), latter/former-1, np.nan)
        # 后为大于0, 前为小于0的情况, 这样实际增长了很多
        growth = np.where(np.logical_and(latter>0, former<0), 1-latter/former, growth)
        # 后为小于0, 前为大于0的情况, 这样实际减少了很多
        growth = np.where(np.logical_and(latter<0, former>0), latter/former-1, growth)
        # 两者都小于0的情况, 在算了增长率后要加负号才正确
        growth = np.where(np.logical_and(latter<0, former<0), 1-latter/former, growth)

        final_growth[:] = growth

        # 进行年化的调整
        if annualize_term != 0:
            final_growth = (final_growth+1) ** (1/annualize_term) - 1

        return final_growth

    # 定义进行fama-macbeth回归的函数, 因为很多论文中用到了大量的fm回归
    @staticmethod
    def fama_macbeth(y, x, *, nw_lags=6, intercept=True):
        """

        :param y: pd.DataFrame
        :param x: pd.Panel
        :param nw_lags: Newey-West adjustment lags
        :return: coefficents, t statitics, rsquared, rsquared adj
        """

        # 堆叠y和x
        stacked_y = y.stack(dropna=False)
        stacked_x = x.to_frame(filter_observations=False)

        # 移除nan的项
        valid = pd.concat([stacked_y, stacked_x], axis=1).notnull().all(1)
        valid_stacked_y = stacked_y[valid]
        valid_stacked_x = stacked_x[valid]

        if nw_lags == 0:
            results_fm = fama_macbeth(y=valid_stacked_y, x=valid_stacked_x, intercept=intercept)
        else:
            results_fm = fama_macbeth(y=valid_stacked_y, x=valid_stacked_x, intercept=intercept,
                                      nw_lags_beta=nw_lags)

        r2 = results_fm._ols_result.r2.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
        if x.shape[0] == 1:
            return results_fm.mean_beta, results_fm.t_stat, r2
        else:
            r2_adj = results_fm._ols_result.r2_adj.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            return results_fm.mean_beta, results_fm.t_stat, r2, r2_adj

    # # 此函数用于计算基准的因子暴露，以及涉及基准的因子暴露计算的超额因子暴露的计算的调整
    # # 在计算基准的因子暴露时（或基于基准因子暴露的超额组合因子暴露），会出现基准中的成分股不能交易的情况，
    # # 这会导致基准的因子暴露计算不准确，因为不能交易的成分股，其因子暴露数据已经被过滤掉了，
    # # 但是其确实在基准中还有权重，因此我们需要把这些成分股的因子暴露数据用此前的数据填充
    # # 填充这个数据的原则是，用这支股票上一个可交易日的数据来填充，无论其是否是nan
    # @staticmethod
    # def adjust_benchmark_related_expo(original_expo, holding_matrix, if_tradable):
    #     """The function of adjusting benchmark related factor exposures
    #
    #     :param original_expo: (pd.Panel) original factor exposure data
    #     :param holding_matrix: (pd.DataFrame) holding matrix of the benchmark or benchmark related portfolio.
    #         note that after returning the adjusted factor exposure, you are expected to get factor exposure of
    #         portfolio using this holding matrix, or error may come out. This parameter may has different index
    #         compared to original expo.
    #     :param if_tradable: (pd.DataFrame) marks indicate if this stock is tradabale at a time. Must have the same index
    #         as original expo.
    #     :return: (pd.Panel) the adjusted factor exposure data, which is expected to be used to get portfolio factor
    #         exposure with holding matrix parameter.
    #     """
    #     # 首先新建因子暴露数据，重索引为持仓的时间段，并将nan填为0
    #     adjusted_expo = original_expo.reindex(major_axis=holding_matrix.index).fillna(0.0)
    #
    #     # 得到那些有持仓，却不可交易的股票
    #     held_but_nontradable = np.logical_and(holding_matrix != 0.0,
    #                                           np.logical_not(if_tradable.reindex(holding_matrix.index)))
    #
    #     # 首先调整非country factor因子
    #     # 填充非country factor因子的原则，永远是用上一个可交易时的数据填充，不管那个数据是不是nan
    #     for item in original_expo.items:
    #         if item != 'country_factor':
    #             # 创建用于fillna的expo，首先将可交易，且数据为nan的地方填成0，
    #             # 这样可以保证之后需要填充的持有且不可交易的地方，会被上一个可交易时的值填充，即使上一个可交易时的值是nan
    #             # 如果不这样做，则会被上一个可交易且非nan（有数据）的填充，损失真实性
    #             tradable_and_null = np.logical_and(if_tradable, original_expo.ix[item].isnull())
    #             # 乘以1为防止传引用
    #             fillna_expo = original_expo.ix[item] * 1
    #             fillna_expo[tradable_and_null] = 0.0
    #             # 这时用于fillna的expo就可以向前填充了，这里为nan的地方都是不可以交易的地方，
    #             # 而向前填nan则意味着用可交易时的数据填充不可交易时的数据
    #             fillna_expo = fillna_expo.fillna(method='ffill').reindex(index=holding_matrix.index)
    #             # 将每个因子中，那些持有且不可交易的股票暴露重新设置为nan
    #             adjusted_expo[item][held_but_nontradable.astype(bool)] = np.nan
    #             # 然后用fillna_expo的数据去填充这些nan，这样可以做到始终用上一个可交易时的数据填充，保证：
    #             # 第一，有持仓却不可交易的地方永远是被上一个可交易的数据填充的，无论那个数据是不是nan
    #             # 第二，无持仓且不可交易的地方仍然是0，虽然其取值不会影响后面的组合暴露的计算
    #             adjusted_expo[item] = adjusted_expo[item].fillna(fillna_expo)
    #     # 对于country factor，需要用1去填充，直接用1填充所有的nan数据即可
    #     if 'country_factor' in original_expo.items:
    #         adjusted_expo['country_factor'] = original_expo.ix['country_factor', holding_matrix.index, :].fillna(
    #             1.0)
    #
    #     return adjusted_expo

    # 计算组合的超额暴露时, 有时候不能用简单的超额持仓乘以因子暴露的方法来计算
    # 如,在基准组合成分股调整的时候, 那一个持仓区间会出现持有的股票调仓时在指数内, 之后不在指数内的情况
    # 如果直接过滤, 会把持仓中的这部分股票剔除掉, 这是不对的. 应该把这些股票的暴露值用最近的有效数据进行填充
    # 计算组合的暴露, 且组合有投资域时, 一样会有这个问题, 即组合可能持有不在投资域内的股票, 不能轻易的剔除他们的数据
    # 具体的更改思路见多因子研究框架的笔记
    @staticmethod
    def get_port_expo(holding_matrix, factor_expo, if_tradable, *, show_warning=True):
        """The function of calculating portfolio (acitve) factor exposures

        :param holding_matrix: (pd.DataFrame) portfolio (active) holding matrix, it may has different
            index compared to factor_expo
        :param factor_expo: (pd.Panel) factor exposure data
        :param if_tradable: (pd.Panel) marks indicate if a stock is tradable(investable, or in pool)
            at a time. have the index as factor_expo, since they usually come from the same
            strategy_data object
        :param show_warning: (Bool) indicate if the function prints a warning message to user when
            the portfolio holds stocks that are not in stock pool
        :return: (pd.DataFrame) the (relative) factor exposure of the portfolio, with index same as
            holding matrix and columns same as factor_expo's items
        """
        # 创建调整后因子暴露的panel
        adjusted_factor_expo = factor_expo.reindex(major_axis=holding_matrix.index).fillna(0.0)

        # 标记需要填充暴露值的那些地方, 即持有的, 可交易的, 且不在投资域的
        held_tradable = np.logical_and(holding_matrix != 0.0,
                            if_tradable['if_tradable'].reindex(holding_matrix.index))
        held_tradable_notinpool = np.logical_and(held_tradable,
                            np.logical_not(if_tradable['if_inpool'].reindex(holding_matrix.index)))

        # 如果设置了警告提醒, 且有持有的, 可交易的, 且不在投资域的持仓, 则输出提示用户
        if show_warning and held_tradable_notinpool.any().any():
            print('Warning: There are some stocks held in position which are tradable and not in stock '
                  'pool. Be aware of possible mis-selection of stocks outside stock pool. Note that '
                  'index component adjustment may also cause this problem. \n')

        # 填充的原则是, 将不在投资域内的, 持有的, 且可交易的股票的因子暴露值,
        # 用上一个最近的可交易且在指数内的值填充, 无论是不是nan
        # country factor的填充原则也是一样的, 因此不做区分
        for item in factor_expo.items:
            # 首先将可交易, 且在投资域内的(即investable)数据为nan的地方填成0
            # 这样可以保证之后需要填充的可交易且不在投资域内的地方都被最近的一个数据填充, 即使这个数据是nan
            # 如果不这样做, 则会被上一个investable的非nan数据填充, 这样就不真实了
            inv_null = np.logical_and(if_tradable['if_inv'], factor_expo[item].isnull())
            # 乘以1防止传引用
            fillna_expo = factor_expo[item] * 1
            fillna_expo[inv_null] = 0.0
            # 这时, 用来fillna的expo就可以向前填充了, 这里被填充的地方都是不可交易或不在投资域的地方
            fillna_expo = fillna_expo.fillna(method='ffill').reindex(index=holding_matrix.index)
            # 将因子中, 那些持有且可交易,且不在投资域中的(即需要填充的那些)那些股票的暴露设置为nan
            adjusted_factor_expo[item][held_tradable_notinpool] = np.nan
            # 用fillna_expo中的数据去填充这些nan, 这样可以做到始终用上一个可交易时的数据填充, 保证:
            # 第一, 有持仓, 可交易, 不在投资域中的地方永远是被上一个可交易的数据填充的, 即使那个数据是nan
            # 第二, 不需要填充的, 且此前是nan的地方, 都会变成0, 而0不会影响之后组合暴露的计算
            adjusted_factor_expo[item] = adjusted_factor_expo[item].fillna(fillna_expo)

        # 因子暴露值调整完成后, 根据持仓矩阵计算组合的因子暴露
        port_expo = np.einsum('ijk,jk->ji', adjusted_factor_expo, holding_matrix)
        port_expo = pd.DataFrame(port_expo, index=holding_matrix.index, columns=factor_expo.items)

        return port_expo

    # 建立指数加权序列
    # 注意, 生成的权重序列若用于权重参数, 则可以直接使用, 若用于直接乘在或除在原始变量上, 且最后要算的量受到数据数量级的影响
    # 则注意一定要根据原始数据的数量n, 对原始数据进行调整, 因为等权相当于每个数据都乘以了1, 而这里的指数权重则是一列和为1的分数
    # 如果不做调整直接乘在原始数据上, 则原始数据的数量级会变小
    @staticmethod
    def construct_expo_weights(halflife, length):
        exponential_lambda = 0.5 ** (1 / halflife)
        exponential_weights = (1 - exponential_lambda) * exponential_lambda ** np.arange(length-1,-1,-1)
        exponential_weights = exponential_weights/np.sum(exponential_weights)
        return exponential_weights

    # 将指数权重直接乘在原始数据上后, 进行数量级调整的函数, 注意, 作为输入变量的weights, 必须和为1
    # 调整数量级的核心思想是, 一定要使得有效数据的权重之和是有效数据的数量valid_n
    # multiply power代表的意思是, 在将权重对有效数据归一化后, 用权重的几次方去乘以原始数据
    # 如算mean的时候需要乘以权重的1次方, 算std的时候需要乘以权重的0.5次方(即根号权重), 默认为1
    # 此函数针对raw_data的类型有不同版本以增加速度
    @staticmethod
    def multiply_weights(raw_data, weights, *, multiply_power=1.0):
        if isinstance(raw_data, pd.Series):
            # 提取出那些真正有作用的权重, 即并未对应nan的那些权重
            valid_weights = np.where(raw_data.notnull(), weights, 0)
            # 第一个调整的步骤, 必须使得有效的数据的权重之和为1
            # 进行这一步调整之后, 有效数据的权重之和就会是1了
            adjusted_weights = np.divide(valid_weights, np.sum(valid_weights))
            # 第二个调整的步骤, 必须使得有效数据的权重之和是有效数据的数量valid n
            valid_n = raw_data.notnull().sum()
            # 因为之前有效数据的权重之和已经是1了, 这时只需要乘以valid n即可
            # 进行这一步调整之后, 有效数据的权重之和就是valid n了
            adjusted_weights *= valid_n
        elif isinstance(raw_data, pd.DataFrame):
            # 当原始数据是dataframe时, 使用向量化的方法来计算, 而不是循环, 这样可以提高速度
            valid_weights = np.where(raw_data.notnull(), np.expand_dims(weights, 1), 0)
            adjusted_weights = np.divide(valid_weights, np.sum(valid_weights, 0))
            valid_n = raw_data.notnull().sum(0)
            adjusted_weights = np.multiply(adjusted_weights, np.expand_dims(valid_n, 0))
        else:
            adjusted_weights = weights * 1
        # 将调整后的有效权重按照指定的幂乘在原始数据上, 形成新的数据
        new_data = raw_data.mul(adjusted_weights ** (multiply_power))

        return new_data

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    