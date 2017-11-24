#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:32:36 2016

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
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from dynamic_backtest import dynamic_backtest
from barra_base import barra_base
from factor_base import factor_base


# 单因子策略的类, 包括测试单因子表现的单因子测试模块

class single_factor_strategy(strategy):
    """ Single factor test strategy.
    
    foo
    """
    def __init__(self):
        strategy.__init__(self)
        # 每个因子策略都需要用到是否可交易的数据
        self.strategy_data.generate_if_tradable(shift=True)
        # 读取市值数据以进行市值加权
        self.strategy_data.stock_price = data.read_data(['FreeMarketValue'],['FreeMarketValue'],shift = True)
        # 用来画图的pdf对象
        self.pdfs = None

    # 生成调仓日的函数
    # holding_freq为持仓频率，默认为月，这个参数将作为resample的参数
    # start_date和end_date为调仓日的范围区间，默认为取数据的所有区间断
    def generate_holding_days(self, *, holding_freq='w', start_date=None, end_date=None, loc=-1):
        # 读取free market value以其日期作为holding days的选取区间
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price.\
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq, loc=loc)
        # 根据传入参数截取需要的调仓日区间
        if isinstance(start_date, pd.Timestamp):
            holding_days = holding_days[holding_days >= start_date]
        if isinstance(end_date, pd.Timestamp):
            holding_days = holding_days[holding_days <= end_date]
        self.holding_days = holding_days
        
    # 选取股票，选股比例默认为最前的80%到100%，方向默认为因子越大越好，weight=1为市值加权，0为等权
    # weight=3 为按照因子值加权, 需注意因子是否进行了标准化
    def select_stocks(self, *, select_ratio=[0.8, 1], direction='+', weight=0,
                      use_factor_expo = True, expo_weight = 1):
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            curr_factor_data = self.strategy_data.factor.ix[0, time, :]
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = curr_factor_data.rank(ascending = True)
            elif direction is '-':
                factor_score = curr_factor_data.rank(ascending = False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            
            # 取有效的股票数
            effective_num = curr_factor_data.dropna().size
            # 无股票可选，进行下一次循环
            if effective_num == 0:
                continue
            # 选取股票的得分范围
            lower_bound = np.floor(effective_num * select_ratio[0])
            upper_bound = np.floor(effective_num * select_ratio[1])
            # 选取股票
            selected_stocks = curr_factor_data.ix[np.logical_and(factor_score>lower_bound,
                                                                 factor_score<=upper_bound)].index
            # 被选取的股票都将持仓调为1
            self.position.holding_matrix.ix[time, selected_stocks] = 1
            
        # 循环结束
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()
        # 设置为等权重        
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
        # 如果是因子加权, 则进行因子值加权
        elif weight == 2:
            # 看是否需要计算因子暴露, 用因子暴露值进行加权
            if use_factor_expo:
                if expo_weight == 1:
                    factor_weight = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                        self.strategy_data.stock_price.ix['FreeMarketValue'])
                elif expo_weight == 0:
                    factor_weight = strategy_data.get_exposure(self.strategy_data.factor.iloc[0])
            else:
                factor_weight = self.strategy_data.factor.iloc[0]
            # 进行因子值加权的权重计算
            self.position.weighted_holding(factor_weight.ix[self.position.holding_matrix.index, :])
        pass

    # 分行业选股，跟上面的选股方式一样，只是在每个行业里选固定比例的股票
    # weight等于0为等权，等于1为直接市值加权，等于2则行业内等权, 行业间按基准加权
    # 等于3为行业内市值加权, 行业间按基准加权
    def select_stocks_within_indus(self, *, select_ratio=[0.8, 1], direction='+', weight=0):
        # 读取行业数据：
        industry = data.read_data(['Industry'], ['Industry'])
        industry = industry['Industry']
        # 定义选股的函数
        def get_stks(factor_data, *, select_ratio=[0.8, 1], direction='+'):
            holding = pd.Series(0, index=factor_data.index)
            # 取有效的股票数
            effective_num = factor_data.dropna().size
            # 无股票可选，进行下一次循环
            if effective_num == 0:
                # 将该行业内的所有股票都选入, 不必担心不可交易的, 或不在投资域的,
                # 因为在选股循环的最后, 会去除这些股票
                holding.ix[:] = 1.0
                return holding
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = factor_data.rank(ascending=True)
            elif direction is '-':
                factor_score = factor_data.rank(ascending=False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            # 选取股票的得分范围
            lower_bound = np.floor(effective_num * select_ratio[0])
            upper_bound = np.floor(effective_num * select_ratio[1])
            # 选取股票
            selected_stocks = factor_score.ix[np.logical_and(factor_score > lower_bound,
                                                             factor_score <= upper_bound)].index
            # 被选取的股票都将持仓调为1
            holding.ix[selected_stocks] = 1.0
            return holding
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            # 当前数据
            curr_data = pd.DataFrame({'factor':self.strategy_data.factor.ix[0, time, :], 'industry':industry.ix[time]})
            # 根据行业分类选股
            curr_holding_matrix = curr_data.groupby('industry')['factor'].apply(get_stks, select_ratio=select_ratio,
                                                                                direction=direction).fillna(0)
            self.position.holding_matrix.ix[time] = curr_holding_matrix
            pass
        # 对不可交易的股票进行过滤
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()

        # 选择加权的方式
        self.position.to_percentage()

        # 使用strategy_data.benchmark_price里的指数权重, 指数权重是从handle_stock_pool里读入的
        # 在handle_stock_pool中, 做了归一化处理, 并且在之后用filter_uninv过滤了不可交易的成分股
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
        elif weight == 2 and self.strategy_data.stock_pool == 'all':
            pass
        elif weight == 2 and self.strategy_data.stock_pool != 'all':
            self.position.weighted_holding_indus(industry, inner_weights=0, outter_weights=self.strategy_data. \
                benchmark_price.ix['Weight_' + self.strategy_data.stock_pool, self.position.holding_matrix.index, :])
        elif weight == 3 and self.strategy_data.stock_pool == 'all':
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
        elif weight == 3 and self.strategy_data.stock_pool != 'all':
            self.position.weighted_holding_indus(industry, inner_weights=self.strategy_data.stock_price.ix \
                ['FreeMarketValue', self.position.holding_matrix.index, :], outter_weights= \
                self.strategy_data.benchmark_price.ix['Weight_'+self.strategy_data.stock_pool,
                self.position.holding_matrix.index, :])

        # benchmark的权重之和少于1的部分, 就是那些在指数中停牌的股票, 这些股票应当当做现金持有
        self.position.cash = 1 - self.strategy_data.benchmark_price.ix['Weight_'+self.strategy_data.stock_pool,
                self.position.holding_matrix.index, :].sum(1)
        pass

    # 用优化的方法构造纯因子组合，纯因子组合保证组合在该因子上有暴露（注意，并不一定是1），在其他因子上无暴露
    # 当优化方法中的方差矩阵为回归权重的逆矩阵时，优化方法和回归方法得到一样的权重（见Barra, efficient replication of factor returns），
    # 这时这里的结果和用回归计算的正交化提纯后的因子的因子收益一样，但是把它做成策略可以放到回测中进行回测，
    # 从而可以考虑交易成本和实际情况。注意：这里先做因子相对barra base的纯因子组合，之后可添加相对任何因子的
    # 这里先做一个简单的直接用解析解算出的组合
    def select_stocks_pure_factor_closed_form(self, *, base_expo, cov_matrix=None, reg_weight='Empty',
                                              direction='+', regulation_lambda=1):
        # 计算因子值的暴露
        factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                                         self.strategy_data.stock_price.ix['FreeMarketValue'])
        if direction == '-':
            factor_expo = - factor_expo
        self.strategy_data.factor_expo = pd.Panel({'factor_expo':factor_expo},
                major_axis=self.strategy_data.factor.major_axis, minor_axis=self.strategy_data.factor.minor_axis)

        # 循环调仓日
        for cursor, time in self.holding_days.iteritems():
            # 当前的因子暴露向量，为n*1
            x_alpha = self.strategy_data.factor_expo.ix['factor_expo', time, :].fillna(0)
            # 当前的其他因子暴露向量，为n*(k-1)，实际就是barra base因子的暴露
            x_sigma = base_expo.ix[:, time, :].fillna(0)

            # 有协方差矩阵，优先用协方差矩阵
            if isinstance(cov_matrix, pd.Panel):
                inv_v = np.linalg.pinv(cov_matrix.ix[time].fillna(0))
            else:
                assert isinstance(reg_weight, pd.DataFrame), 'The construction of pure factor ' \
                    'portfolio require one of following:\n Covariance matrix of factor returns ' \
                    '(priority), OR \n Regression weight when getting factor return using linear ' \
                    'regression.\n'
                # 取当期的回归权重，每只股票的权重在对角线上
                # inv_v = np.diag(reg_weight.ix[time].fillna(0))
                curr_weight = reg_weight.ix[time]
                curr_weight = (curr_weight/curr_weight.sum()).fillna(0)
                inv_v = np.diag(curr_weight)

            # 通过优化的解析解计算权重，解析解公式见barra, Efficient Replication of Factor Returns, equation (6)
            temp_1 = np.linalg.pinv(np.dot(np.dot(x_sigma.T, inv_v), x_sigma))
            temp_2 = np.dot(np.dot(x_sigma.T, inv_v), x_alpha)
            temp_3 = x_alpha - np.dot(np.dot(x_sigma, temp_1), temp_2)
            h_star = 1/regulation_lambda * np.dot(inv_v, temp_3)

            # 加权方式只能为这一种，只是需要归一化一下
            self.position.holding_matrix.ix[time] = h_star

        self.position.to_percentage()
        pass

    # 上一种选股方法的优化解法
    def select_stocks_pure_factor(self, *, base_expo, cov_matrix=None, reg_weight=None, direction='+',
                                  benchmark_weight=None, is_long_only=True, use_factor_expo=True,
                                  expo_weight=1):
        # 计算因子值的暴露
        if use_factor_expo:
            if expo_weight == 1:
                factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                self.strategy_data.stock_price.ix['FreeMarketValue'])
            elif expo_weight == 0 :
                factor_expo = strategy_data.get_exposure(self.strategy_data.factor.iloc[0])
        else:
            factor_expo = self.strategy_data.factor.iloc[0]

        if direction == '-':
            factor_expo = - factor_expo
        self.strategy_data.factor_expo = pd.Panel({'factor_expo': factor_expo},
                                                  major_axis=self.strategy_data.factor.major_axis,
                                                  minor_axis=self.strategy_data.factor.minor_axis)
        # 如果有benchmark，则计算benchmark的暴露
        if isinstance(benchmark_weight, pd.DataFrame):
            benchmark_weight = benchmark_weight.fillna(0.0)
            benchmark_base_expo = strategy_data.get_port_expo(benchmark_weight, base_expo,
                                                              self.strategy_data.if_tradable)
            benchmark_curr_factor_expo = strategy_data.get_port_expo(benchmark_weight,
                pd.Panel({'factor_expo': factor_expo}), self.strategy_data.if_tradable)
            # 注意, benchmark_curr_factor_expo是一个只有一列的dataframe, 要把它转成series
            benchmark_curr_factor_expo = benchmark_curr_factor_expo.iloc[:, 0]
            self.strategy_data.factor_expo.ix['factor_expo'] = factor_expo.sub(benchmark_curr_factor_expo, axis=0)

        # 循环调仓日
        for cursor, time in self.holding_days.iteritems():
            curr_factor_expo = self.strategy_data.factor_expo.ix['factor_expo', time, :]
            curr_base_expo = base_expo.ix[:, time, :]

            # 有协方差矩阵，优先用协方差矩阵
            if isinstance(cov_matrix, pd.Panel):
                curr_v = cov_matrix.ix[time]
                curr_v_diag = curr_v.diagonal()
                # 去除有nan的数据
                all_data = pd.concat([curr_v_diag, curr_factor_expo, curr_base_expo], axis=1)
                all_data = all_data.dropna()
                # 如果有效数据小于等于1，当期不选股票
                if all_data.shape[0] <= 1:
                    continue
                # 指数中选股可能会出现一个行业暴露全是0的情况，所以关于这个行业的限制条件会冗余，于是要进行剔除
                all_data = all_data.replace(0, np.nan).dropna(axis=1, how='all').fillna(0.0)
                curr_factor_expo = all_data.ix[:, 0]
                curr_v_diag = all_data.ix[:, 1]
                curr_base_expo = all_data.ix[:, 2:]
                curr_v = curr_v.reindex(index=curr_v_diag.index, columns=curr_v_diag.index)
            else:
                assert isinstance(reg_weight, pd.DataFrame), 'The construction of pure factor portfolio ' \
                    'require one of following:\n Covariance matrix of factor returns (priority), OR \n' \
                    'Regression weight when getting factor return using linear regression.\n'
                # 取当期的回归权重，每只股票的权重在对角线上
                curr_v_diag = reg_weight.ix[time]
                # 去除有nan的数据
                all_data = pd.concat([curr_v_diag, curr_factor_expo, curr_base_expo], axis=1)
                all_data = all_data.dropna()
                # 如果有效数据小于等于1，当期不选股票
                if all_data.shape[0] <= 1:
                    continue
                # 指数中选股可能会出现一个行业暴露全是0的情况，所以关于这个行业的限制条件会冗余，于是要进行剔除
                all_data = all_data.replace(0, np.nan).dropna(axis=1, how='all').fillna(0.0)
                curr_v_diag = all_data.ix[:, 0]
                curr_factor_expo = all_data.ix[:, 1]
                curr_base_expo = all_data.ix[:, 2:]
                # 将回归权重归一化
                curr_v_diag = curr_v_diag / curr_v_diag.sum()
                curr_v = np.linalg.pinv(np.diag(curr_v_diag))
                curr_v = pd.DataFrame(curr_v, index=curr_factor_expo.index, columns=curr_factor_expo.index)

            # 设置其他因子为0的限制条件，在有基准的时候，设置为基准的暴露
            if type(benchmark_weight) != str:
                expo_target = benchmark_base_expo.ix[time].reindex(index=curr_base_expo.columns)
            else:
                expo_target = pd.Series(0.0, index=curr_base_expo.columns)

            # 开始设置优化
            # P = V
            P = matrix(curr_v.as_matrix())
            # q = - (factor_expo.T)
            q = matrix(-curr_factor_expo.as_matrix().transpose())

            # 其他因为暴露为0，或等于基准的限制条件
            A = matrix(curr_base_expo.as_matrix().transpose())
            b = matrix(expo_target.as_matrix())

            solvers.options['show_progress'] = False

            # 如果只能做多，则每只股票的比例都必须大于等于0
            if is_long_only:
                long_only_constraint = pd.DataFrame(-1.0*np.eye(curr_factor_expo.size), index=curr_factor_expo.index,
                                                   columns=curr_factor_expo.index)
                long_only_target = pd.Series(0.0, index=curr_factor_expo.index)

                G = matrix(long_only_constraint.as_matrix())
                h = matrix(long_only_target.as_matrix())

                # 解优化问题
                results = solvers.qp(P=P, q=q, A=A, b=b, G=G,  h=h)
            else:
                results = solvers.qp(P=P, q=q, A=A, b=b)

            results_np = np.array(results['x']).squeeze()
            results_s = pd.Series(results_np, index=curr_factor_expo.index)
            # 重索引为所有股票代码
            results_s = results_s.reindex(self.strategy_data.stock_price.minor_axis, fill_value=0)

            # 股票持仓
            self.position.holding_matrix.ix[time] = results_s

        # 循环结束后，进行权重归一化
        self.position.to_percentage()
        # 如果有benchmark, 归一化后, 将benchmark中不可交易的比例当做现金持有, 即benchmark因为停牌的原因
        # 其在country factor上的暴露并不一定是1, 因此停牌的那部分, 组合要当做现金持有, 更为合理
        if isinstance(benchmark_weight, pd.DataFrame):
            self.position.cash = 1 - benchmark_weight.ix[self.holding_days, :].sum(1)

        # 因为优化后重新归一的组合, 其对目标因子暴露不一定是1,
        # (详情见barra文档或active portfolio management第一章附录),
        # 因此这里需要计算组合对当前因子的暴露, 总的或者是超额的, 并且输出平均暴露信息
        port_factor_expo = self.position.holding_matrix.reindex(index=
            self.strategy_data.factor_expo.major_axis, method='ffill').mul(
            self.strategy_data.factor_expo.iloc[0]).sum(1).replace(0.0, np.nan)
        # 注意, 如果有benchmark, 可能因为现金的存在, 是的纯因子暴露再一次被稀释
        if isinstance(benchmark_weight, pd.DataFrame):
            port_factor_expo *= 1 - self.position.cash.reindex(index=port_factor_expo.index, method='ffill')
        self.pure_factor_expo_of_port = port_factor_expo
        # 输出平均数, 给用户参考
        port_factor_expo_mean = port_factor_expo.mean()
        output_str = 'The exposure of constructed pure factor portfolio ' \
                     'on the target factor is: {0} \n'.format(port_factor_expo_mean)
        print(output_str)

        pass

    # 初始化回测对象, 为构造动态化的策略做准备
    def initialize_dynamic_backtest(self, *, bkt_obj=None, bkt_start=None, bkt_end=None):
        # 如果有外来的回测对象, 则使用这个对象
        if isinstance(bkt_obj, dynamic_backtest):
            self.dy_bkt = bkt_obj
        # 如果没有, 则自己建立这个对象
        else:
            # 如果自身的持仓矩阵还是空的, 则需要建立持仓矩阵
            if self.position.holding_matrix.empty:
                # 持仓矩阵的日期由调仓日决定, 持仓矩阵的股票油策略数据类里的股票决定
                if self.holding_days.empty:
                    self.generate_holding_days()
                self.initialize_position(self.strategy_data.stock_price.ix[0, self.holding_days, :])
            self.dy_bkt = dynamic_backtest(self.position, bkt_start=bkt_start, bkt_end=bkt_end)

    # 单因子的因子收益率计算和检验，用来判断因子有效性，
    # holding_freq为回归收益的频率，默认为月，可调整为与调仓周期一样，也可不同
    # weights为用来回归的权重，默认为等权回归
    def get_factor_return(self, *, holding_freq='m', weights=None, direction='+', plot_cum=True,
                          start=None, end=None):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算因子收益的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price.\
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 如果有指定，只取start和end之间的时间计算
        if isinstance(start, pd.Timestamp):
            holding_days = holding_days[start:]
        if isinstance(end, pd.Timestamp):
            holding_days = holding_days[:end]
        # 计算股票简单收益以及因子暴露
        # 这里虽然不涉及时间截面上的资产组合加总，但是为了和归因中的纯因子组合对比，最好用简单收益
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',holding_days,:]
        holding_day_return = holding_day_price.div(holding_day_price.shift(1)).sub(1.0)
        holding_day_factor = self.strategy_data.factor.ix[0, holding_days, :]
        holding_day_factor_expo = strategy_data.get_cap_wgt_exposure(holding_day_factor,
                                    self.strategy_data.stock_price.ix['FreeMarketValue', holding_days, :])
        # 注意因子暴露要用前一期的数据
        holding_day_factor_expo = holding_day_factor_expo.shift(1)

        # 初始化因子收益序列以及估计量的t统计量序列
        factor_return_series = np.empty(holding_days.size)*np.nan
        t_stats_series = np.empty(holding_days.size)*np.nan
        self.factor_return_series = pd.Series(factor_return_series, index=holding_days)
        self.t_stats_series = pd.Series(t_stats_series, index=holding_days)

        # 进行回归，对调仓日进行循环
        for cursor, time in holding_days.iteritems():

            y = holding_day_return.ix[time, :]
            x = holding_day_factor_expo.ix[time, :]
            if y.isnull().all() or x.isnull().all():
                continue
            x = sm.add_constant(x)
            if weights is None:
                results = sm.WLS(y, x, missing='drop').fit()
            else:
                results = sm.WLS(y, x, weights=weights.ix[time], missing='drop').fit()
            self.factor_return_series.ix[time] = results.params[1]
            self.t_stats_series.ix[time] = results.tvalues[1]

        # 如果方向为负，则将因子收益和t统计量加个负号
        if direction == '-':
            self.factor_return_series = -self.factor_return_series
            self.t_stats_series = -self.t_stats_series

        # 输出的string
        tstats_sig_ratio = self.t_stats_series[np.abs(self.t_stats_series) >= 2].size / self.t_stats_series.size
        target_str = 'The average return of this factor: {0:.4f}%\n' \
                     'Note that the return of factor is not annualized but corresponding to the holding days interval\n' \
                     'The average t-statistics value: {1:.4f}\n' \
                     'Ratio of t_stats whose absolute value >= 2: {2:.2f}%\n'.format(
            self.factor_return_series.mean()*100, self.t_stats_series.mean(), tstats_sig_ratio*100
        )

        # 循环结束，输出结果
        print(target_str)
        with open(str(os.path.abspath('.'))+'/'+self.strategy_data.stock_pool+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)

        # 画图，默认画因子收益的累计收益图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        zero_series = pd.Series(np.zeros(self.factor_return_series.shape), index=self.factor_return_series.index)
        if plot_cum:
            plt.plot(self.factor_return_series.add(1).cumprod().sub(1)*100, 'b-')
        else:
            plt.plot(self.factor_return_series*100, 'b-')
            plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return of The Factor (%)')
        ax.set_title('The Return Series of The Factor')
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorReturn.png', dpi=1200)
        if isinstance(self.pdfs, PdfPages):
            plt.savefig(self.pdfs, format='pdf')

        fx = plt.figure()
        ax = fx.add_subplot(1, 1, 1)
        plt.plot(self.t_stats_series, 'b-')
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('T-Stats of The Factor Return')
        ax.set_title('The T-Stats Series of The Factor Return')
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorReturnTStats.png', dpi=1200)
        if isinstance(self.pdfs, PdfPages):
            plt.savefig(self.pdfs, format='pdf')

    # 计算因子的IC，股票收益率是以holding_freq为频率的的收益率，默认为月
    def get_factor_ic(self, *, holding_freq='m', direction = '+', start=None, end=None):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算ic的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price. \
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 如果有指定，只取start和end之间的时间计算
        if isinstance(start, pd.Timestamp):
            holding_days = holding_days[start:]
        if isinstance(end, pd.Timestamp):
            holding_days = holding_days[:end]
        # 初始化ic矩阵
        ic_series = np.empty(holding_days.size)*np.nan
        self.ic_series = pd.Series(ic_series, index = holding_days)
        # 计算股票简单收益，提取因子值，同样的，因子值要用前一期的因子值
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',holding_days,:]
        holding_day_return = holding_day_price.div(holding_day_price.shift(1)).sub(1.0)
        holding_day_factor = self.strategy_data.factor.ix[0, holding_days, :]
        holding_day_factor = holding_day_factor.shift(1)
        # 对调仓日进行循环
        for cursor, time in holding_days.iteritems():
            # 计算因子值的排序
            curr_factor_data = holding_day_factor.ix[time, :]
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = curr_factor_data.rank(ascending = True)
            elif direction is '-':
                factor_score = curr_factor_data.rank(ascending = False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            
            # 对因子实现的简单收益率进行排序，升序排列，因此同样，秩类似于得分
            return_score = holding_day_return.ix[time, :].rank(ascending = True)
            
            # 计算得分（秩）之间的线性相关系数，就是秩相关系数
            self.ic_series.ix[time] = factor_score.corr(return_score, method = 'pearson')
            
        # 循环结束
        # 输出结果
        target_str = 'The average IC of this factor: {0:.4f}\n'.format(self.ic_series.mean())
        print(target_str)
        with open(str(os.path.abspath('.'))+'/'+self.strategy_data.stock_pool+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)
        
        # 画图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        plt.plot(self.ic_series, 'b-')
        # 画一条一直为0的图，以方便观察IC的走势是否显著不为0
        zero_series = pd.Series(np.zeros(self.ic_series.shape), index = self.ic_series.index)
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('IC of The Factor')
        ax.set_title('The IC Time Series of The Factor')
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorIC.png', dpi=1200)
        if isinstance(self.pdfs, PdfPages):
            plt.savefig(self.pdfs, format='pdf')
        
    # 根据分位数分组选股，用来画同一因子不同分位数分组之间的收益率对比，以此判断因子的有效性
    def select_qgroup(self, no_of_groups, group, *, direction = '+', weight = 0):
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            curr_factor_data = self.strategy_data.factor.ix[0, time, :]
            # 无股票可选，则直接进行下一次循环
            if curr_factor_data.dropna().empty:
                continue
            # 对因子值进行调整，使得其在qcut后，分组标签越小的总是在最有利的方向上
            if direction is '+':
                curr_factor_data = -curr_factor_data
            elif direction is '-':
                pass
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
                
            # # 进行qcut
            # labeled_factor = pd.qcut(curr_factor_data, no_of_groups, labels = False)

            # This is a temporary solution to pandas.qcut's unique bin edge error.
            # It will be removed when pandas 0.20.0 releases, which gives an additional parameter to handle this problem
            def pct_rank_qcut(series, n):
                edges = pd.Series([float(i) / n for i in range(n + 1)])
                f = lambda x: (edges >= x).argmax()-1
                return series.rank(pct=1).apply(f).reindex(series.index)
            labeled_factor = pct_rank_qcut(curr_factor_data.dropna(), no_of_groups).reindex(curr_factor_data.index)

            # 选取指定组的股票，注意标签是0开始，传入参数是1开始，因此需要减1
            selected_stocks = curr_factor_data.ix[labeled_factor == group-1].index
            # 被选取股票的持仓调为1
            self.position.holding_matrix.ix[time, selected_stocks] = 1

        # 循环结束
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()
        # 设置为等权重
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
    
    # 循环画分位数图与long short图的函数
    # 定义按因子分位数选股的函数，将不同分位数收益率画到一张图上，同时还会画long-short的图
    # value=1为画净值曲线图，value=2为画对数收益率图，weight=0为等权，=1为市值加权
    def plot_qgroup(self, bkt, no_of_groups, *, direction='+', value=1, weight=0):
        # 默认画净值曲线图
        if value == 1:
            # 先初始化图片
            f1 = plt.figure()
            ax1 = f1.add_subplot(1, 1, 1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Net Account Value')
            ax1.set_title('Net Account Value Comparison of Different Quantile Groups of The Factor')

            # 开始循环选股、画图
            for group in range(no_of_groups):
                # 选股
                self.reset_position()
                self.select_qgroup(no_of_groups, group + 1, direction=direction, weight=weight)

                # 回测
                bkt.show_warning = False
                bkt.reset_bkt_position(self.position)
                bkt.execute_backtest()
                bkt.initialize_performance()

                # 画图，注意，这里画净值曲线图，差异很小时，净值曲线图的差异更明显
                plt.plot(bkt.bkt_performance.net_account_value, label='Group %s' % str(group + 1))

                # 储存第一组和最后一组以画long-short收益图
                if group == 0:
                    long_series = bkt.bkt_performance.net_account_value
                elif group == no_of_groups - 1:
                    short_series = bkt.bkt_performance.net_account_value

            ax1.legend(loc='best')
            plt.xticks(rotation=30)
            plt.grid()
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'QGroupsNetValue.png', dpi=1200)
            if isinstance(self.pdfs, PdfPages):
                plt.savefig(self.pdfs, format='pdf')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Net Account Value')
            ax2.set_title('Net Account Value of Long-Short Portfolio of The Factor')
            plt.plot(long_series - short_series)
            plt.xticks(rotation=30)
            plt.grid()
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'LongShortNetValue.png', dpi=1200)
            if isinstance(self.pdfs, PdfPages):
                plt.savefig(self.pdfs, format='pdf')

        elif value == 2:
            # 先初始化图片
            f1 = plt.figure()
            ax1 = f1.add_subplot(1, 1, 1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Cumulative Log Return (%)')
            ax1.set_title('Cumulative Log Return Comparison of Different Quantile Groups of The Factor')

            # 开始循环选股、画图
            for group in range(no_of_groups):
                # 选股
                self.reset_position()
                self.select_qgroup(no_of_groups, group + 1, direction=direction, weight=weight)

                # 回测
                bkt.show_warning = False
                bkt.reset_bkt_position(self.position)
                bkt.execute_backtest()
                bkt.initialize_performance()

                # 画图，注意，这里画累积对数收益图，当差异很大时，累积对数收益图看起来更容易
                plt.plot(bkt.bkt_performance.cum_log_return * 100, label='Group %s' % str(group + 1))

                # 储存第一组和最后一组以画long-short收益图
                if group == 0:
                    long_series = bkt.bkt_performance.cum_log_return
                elif group == no_of_groups - 1:
                    short_series = bkt.bkt_performance.cum_log_return

            ax1.legend(loc='best')
            plt.xticks(rotation=30)
            plt.grid()
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'QGroupsCumLog.png', dpi=1200)
            if isinstance(self.pdfs, PdfPages):
                plt.savefig(self.pdfs, format='pdf')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('Cumulative Log Return of Long-Short Portfolio of The Factor')
            plt.plot((long_series - short_series) * 100)
            plt.xticks(rotation=30)
            plt.grid()
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'LongShortCumLog.png', dpi=1200)
            if isinstance(self.pdfs, PdfPages):
                plt.savefig(self.pdfs, format='pdf')

    # 用回归取残差的方法（即gram-schmidt正交法）取因子相对一基准的纯因子暴露
    # 之后可以用这个因子暴露当作因子进行选股，以及回归得纯因子组合收益率（主要用途），或者算ic等
    # reg_weight=1为默认值, 即barra中的以市值的平方根为权重, weight=0为默认权重
    # add_constant为是否要对线性回归加上截距项, 如果有行业这种dummy variable, 则不需要加入
    # use_factor_expo为是否要使用当前因子的暴露进行回归, 设为true会根据当前因子计算因子暴露
    # 如果当前因子已经是计算过的因子暴露了, 则使用False, 否则会winsorize两次
    # expo weight为计算暴露时计算barra风格的市值加权暴露还是简单的暴露, 1为市值加权暴露, 0为简单暴露
    # get_expo_again为指示对提纯后的残差因子是否要再次计算暴露, 计算暴露的话, 不能再winsorize
    # 再次计算暴露后的因子将不再与base中的因子正交, 但是却拥有了和其他因子暴露之间的可比性
    # 如不再次计算暴露, 则保留了正交性, 不再与其他因子暴露之间可比, 但是用回归的方法算因子收益的时候
    # 可以得到相对base因子的纯因子收益, 在这个角度上更有参考性, 因此是否重新计算暴露, 两者都会经常用到
    def get_pure_factor_gs_orth(self, base_expo, *, reg_weight=1, add_constant=False,
                                use_factor_expo=True, expo_weight=1, get_expo_again=True):
        # 计算当前因子的暴露，注意策略里的数据都已经lag过了
        if use_factor_expo:
            if expo_weight == 1:
                factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                self.strategy_data.stock_price.ix['FreeMarketValue'])
            elif expo_weight == 0:
                factor_expo = strategy_data.get_exposure(self.strategy_data.factor.iloc[0])
        else:
            factor_expo = self.strategy_data.factor.iloc[0]

        # 在base expo里去掉国家因子，去掉是为了保证有唯一解，而且去掉后残差值不变，不影响结果
        # 因为国家因子向量已经能表示成行业暴露的线性组合了
        if 'country_factor' in base_expo.items:
            base_expo_no_cf = base_expo.drop('country_factor', axis=0)
        else:
            base_expo_no_cf = base_expo
        # 利用多元线性回归进行提纯
        if reg_weight == 1:
            pure_factor_expo = strategy_data.simple_orth_gs(factor_expo, base_expo_no_cf, weights=
                np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']), add_constant=add_constant)[0]
        elif reg_weight == 0:
            pure_factor_expo = strategy_data.simple_orth_gs(factor_expo, base_expo_no_cf, add_constant=add_constant)[0]

        # 得到的纯化后的因子, 要对其重新计算暴露, 要注意的是, 一旦再次计算了暴露, 这个因子将不再与base中的因子正交了
        # 而且这里的因子暴露的计算不能再winsorize, 因为回归用到的因子暴露已经去过极值了
        if get_expo_again:
            if expo_weight == 1:
                pure_factor_expo = strategy_data.get_cap_wgt_exposure(pure_factor_expo,
                        self.strategy_data.stock_price.ix['FreeMarketValue'], percentile=0)
            elif expo_weight == 0:
                pure_factor_expo = strategy_data.get_exposure(pure_factor_expo, percentile=0)
        # 无论如何都将当前因子是否已经是暴露值的标签设置为True.
        # 这样做的原因是, 如果没有选择重新计算暴露, 则目的是用残差进行回归, 以得到相对base因子的春因子组合收益率
        # 这是如果不将标签改为True, 则可能会在后面重新计算暴露, 导致达不到希望的目的
        self.is_curr_factor_already_expo = True

        # 将得到的纯化因子放入因子值中储存
        self.strategy_data.factor.iloc[0] = pure_factor_expo

    # 提取纯因子的外函数, 主要用作根据不同的策略, 选取不同的base
    # 默认的单因子测试中的纯因子为相对barra base的纯因子
    def get_pure_factor(self, base_obj, *, reg_weight=1, add_constant=False, use_factor_expo=True,
                        expo_weight=1, get_expo_again=True):
        # 注意，因为这里是用base对因子进行提纯，而不是用base归因，因此base需要lag一期，才不会用到未来信息
        # 否则就是用未来的base信息来对上一期的已知的因子进行提纯，而这里因子暴露的计算lag不会影响归因时候的计算
        # 因为归因时候的计算会用没有lag的因子值和其他base数据重新计算暴露
        lag_base_expo = base_obj.base_data.factor_expo.shift(1).reindex(major_axis=base_obj.base_data.factor_expo.major_axis)

        # 进行提纯
        self.get_pure_factor_gs_orth(lag_base_expo, reg_weight=reg_weight, add_constant=add_constant,
                                     use_factor_expo=use_factor_expo, expo_weight=expo_weight,
                                     get_expo_again=get_expo_again)

    # 检验因子间相关性的内函数, 与计算纯因子的内函数基本相同
    # reg_weight为回归的权重, 1为barra的市值平方根, 0为等权回归
    # add constant为是否要在线性回归中加截距项
    # use_factor_expo为是否要对当前因子的暴露进行回归, 若当前因子已经是暴露值, 则设为false
    # expo_weight为计算暴露时的权重,1为市值加权, 0为等权
    def get_factor_corr_gs_orth(self, base_expo, *, reg_weight=1, add_constant=False, use_factor_expo=True,
                                expo_weight=1):
        # 计算当前因子的暴露
        if use_factor_expo:
            if expo_weight == 1:
                factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                self.strategy_data.stock_price.ix['FreeMarketValue'])
            elif expo_weight == 0:
                factor_expo = strategy_data.get_exposure(self.strategy_data.factor.iloc[0])
        else:
            factor_expo = self.strategy_data.factor.iloc[0]

        # 在base expo里去掉国家因子，去掉是为了保证有唯一解，而且去掉后残差值不变，不影响结果
        # 因为国家因子向量已经能表示成行业暴露的线性组合了
        if 'country_factor' in base_expo.items:
            base_expo_no_cf = base_expo.drop('country_factor', axis=0)
        else:
            base_expo_no_cf = base_expo
        # 利用多元线性回归的结果进行因子相关性检验
        if reg_weight == 1:
            residual, pvalues, rsquared = \
                strategy_data.simple_orth_gs(factor_expo, base_expo_no_cf,
                weights=np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']),
                add_constant=add_constant)
            # 注意, 加权回归的输出变量为回归残差除以根号权重后得到的, 而这里只需要回归残差, 因此要将根号权重乘回来
            residual = residual.mul(np.sqrt(np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue'])))
            # 同理, 原始因子值也要乘以根号权重, 以获得其在回归时的那个向量
            y = factor_expo.mul(np.sqrt(np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue'])))
        elif reg_weight == 0:
            residual, pvalues, rsquared = \
                strategy_data.simple_orth_gs(factor_expo, base_expo_no_cf, add_constant=add_constant)
            # 等权回归就不需要乘了
            y = factor_expo * 1

        # 通过回归结果, 计算检验相关性的指标
        # 判断相关性的其中一个指标, 计算残差向量和原始向量的夹角的cos值
        # 注意, 并不是所有的原始因子值都用于了回归, 因此将residual中为nan的, 原始因子中也要改为nan
        y = y.where(residual.notnull(), np.nan)
        # 计算cosine的值
        residual_cosine = (residual * y).sum(1).div(np.sqrt((residual ** 2).sum(1))). \
            div(np.sqrt((y ** 2).sum(1)))
        # 通过反函数, 由cosine值求夹角值
        residual_angle = np.arccos(residual_cosine)
        # 详见维基百科中cosine similarity中的angular distance and similarity
        # distance为1为最远, 等于0(代表残差和原因子同一方向)或等于2(代表残差和原因子相反方向)为距离最近
        angular_distance = 2 * residual_angle / np.pi
        # similarity, 等于0时代表不相似, 等于1或-1时代表非常相关, 分别代表同一方向的相似和相反反向的相似
        # 如果残差与原因子相似度低(值趋近于0), 则说明原因子被基础因子大量解释
        # 如果残差与原因子相似度高(值趋近于1或-1), 则说明原因子几乎没有被基础因子解释
        angular_similarity = 1 - angular_distance
        # 另外一种检验相关性的方法, 直接计算原因子与残差项的线性相关系数
        # 注意, 当因子均值为0时(即使用等权回归时), cos值与线性相关系数一模一样
        residual_corr = y.corrwith(residual, axis=1)
        # 另外一种检验相关性的方法, 使用vif, 即用这个回归的rsquared来计算vif,
        # 根据这个回归的拟合程度来判断因子间的相关性
        vif = 1 / (1 - rsquared['rsquared'])

        # 统计能够展示因子相关性的指标
        # 首先是残差的内积, 然后是p值和rsquared_adj的平均值
        residual_cosine_mean = residual_cosine.mean()
        angular_similarity_mean = angular_similarity.mean()
        residual_corr_mean = residual_corr.mean()
        vif_mean = vif.mean()
        pvalues_mean = pvalues.mean()
        rsquared_adj_mean = rsquared['rsquared_adj'].mean()
        # 输出这些信息
        output_str = 'Factor correlation test outcome: \n' \
                     'Average cosine between residual and y: {0} \n' \
                     'Average angular similarity between residual and y: {1} \n' \
                     'Average corr between residual and y: {2} \n' \
                     'Average VIF: {3}' \
                     'Average p values: {4} \n' \
                     'Average r squared adjusted: {5} \n'.format(
            residual_cosine_mean, angular_similarity_mean, residual_corr_mean, vif_mean,
            pvalues_mean, rsquared_adj_mean)
        print(output_str)

        # 将序列信息储存起来
        self.factor_corr_test_outcome = [residual_cosine, angular_similarity, residual_corr,
                                         vif, pvalues, rsquared]

    # 检验因子间相关性的外函数, 主要用作根据不同的策略, 选取不同的base
    # 默认为barra base因子
    # 使用回归的方法检验因子间的相关性, 提出残差, 根据残差大小来检验因子间的相关性
    def get_factor_corr_test(self, base_obj, *, reg_weight=1, add_constant=False, use_factor_expo=True,
                                expo_weight=1):
        # 要进行lag
        lag_base_expo = base_obj.base_data.factor_expo.shift(1).reindex(major_axis=base_obj.base_data.factor_expo.major_axis)

        self.get_factor_corr_gs_orth(lag_base_expo, reg_weight=reg_weight, add_constant=add_constant,
                                     use_factor_expo=use_factor_expo, expo_weight=expo_weight)


    # 根据一个股票池进行一次完整的单因子测试的函数
    # select method为单因子测试策略的选股方式，0为按比例选股，1为分行业按比例选股
    def single_factor_test(self, *, loc=-1, factor=None, direction='+', bkt_obj=None, base_obj=None,
                           discard_factor=[], bkt_start=None, bkt_end=None, stock_pool='all',
                           select_method=0, do_pa=True, do_active_pa=False, do_base_pure_factor=False,
                           holding_freq='w', do_data_description=False, do_factor_corr_test=False):
        ###################################################################################################
        # 第一部分是生成调仓日, 股票池, 及可投资标记
        # 生成调仓日和生成可投资标记是第一件事, 因为之后包括因子构建的函数都要用到它
        self.sft_part_1(loc=loc, stock_pool=stock_pool, holding_freq=holding_freq)

        ###################################################################################################
        # 第二部分是读取或生成要研究的因子
        self.sft_part_2(factor=factor)

        ###################################################################################################
        # 第三部分是除去不可投资的数据, 初始化或者重置策略持仓,
        # 处理barra base类, backtest类, 以及建立文件夹等零碎的事情
        self.sft_part_3(base_obj=base_obj)

        ###################################################################################################
        # 第四部分为, 1. 若各策略类有对原始因子数据的计算等, 可以在data description中进行
        # 2. 根据选择的一组因子base(可以是barra的, 也可以不是), 进行与当前因子的相关性检验
        # 3. 根据选择的一组因子base, 对当前因子进行提纯, 注意尽管与2差不多, 但是这里使得2, 3可以选择不同的base以及提纯方法
        # 4. 未来还可添加: 如果是基础财务因子, 还可以添加对净利润增长的预测情况
        self.sft_part_4(do_data_description=do_data_description, do_factor_corr_test=do_factor_corr_test,
                        do_base_pure_factor=do_base_pure_factor)

        ###################################################################################################
        # 第五部分为, 根据不同的单因子选股策略, 进行选股
        self.sft_part_5(select_method=select_method, direction=direction)

        ###################################################################################################
        # 一个介于第五部分和第六部分之间的额外部分
        # 这里可以直接插入一个外来的持仓矩阵, 测试这个持仓矩阵的回测结果, 和归因结果
        # 这个函数默认是一个空函数
        self.sft_test_outside_position()

        ###################################################################################################
        # 第六部分为, 1. 对策略选出的股票进行回测, 画图
        # 2. 如果有归因, 则对策略选出的股票进行归因
        self.sft_part_6(bkt_obj=bkt_obj, bkt_start=bkt_start, bkt_end=bkt_end, do_pa=do_pa,
                        do_active_pa=do_active_pa, discard_factor=discard_factor)

        ###################################################################################################
        # 第七部分为, 1. 根据回归算单因子的纯因子组合收益率
        # 2. 计算单因子的ic序列
        # 3. 单因子选股策略的n分位图
        # 4. 画单因子策略n分位图的long-short图
        self.sft_part_7(direction=direction, bkt_start=bkt_start, bkt_end=bkt_end)

        ###################################################################################################
        # 第八部分, 最后的收尾工作
        self.pdfs.close()

        ###################################################################################################
        # 单因子测试函数结束
        ###################################################################################################

    def sft_part_1(self, *, loc=-1, stock_pool='all', holding_freq='w'):
        # 第一部分是生成调仓日, 股票池, 及可投资标记
        # 生成调仓日和生成可投资标记是第一件事, 因为之后包括因子构建的函数都要用到它

        # 生成调仓日
        if self.holding_days.empty:
            self.generate_holding_days(holding_freq=holding_freq, loc=loc)

        # 将策略的股票池设置为当前股票池
        self.strategy_data.stock_pool = stock_pool
        # 根据股票池生成标记
        self.strategy_data.handle_stock_pool(shift=True)

    def sft_part_2(self, *, factor=None):
        # 第二部分是读取或生成要研究的因子

        # 首先有一个指示因子是否是原始因子, 或者已经是因子暴露值的量, 如果因子已经是暴露值,
        # 则在之后的的很多情况里, 就不能再计算暴露了, 特别是不能再winsorize
        # 首先初始化为false, 然后在这里的因子计算中, 如果将因子值变成了因子暴露, 要记得改为True
        self.is_curr_factor_already_expo = False

        # 如果传入的是str，则读取同名文件，如果是dataframe，则直接传入因子, 如果是None,
        # 则使用内部的构建因子函数construct_factor
        if type(factor) == str:
            self.strategy_data.factor = data.read_data([factor], shift=True)
        elif factor is None:
            # 如果传入的是None, 那么看策略类本身是否有内部构建其因子的函数
            self.construct_factor()
            # 检测是否已经有因子存在,1. 因子是由策略类内部自身构建的
            # 2. 有可能该实例化的类已经有了计算好的要测试的因子
            if self.strategy_data.factor.shape[0] >= 1:
                print('The factor has been set to be the FIRST one in strategy_data.factor\n')
            elif self.strategy_data.factor_expo.shape[0] >= 1:
                self.strategy_data.factor = self.strategy_data.factor_expo
                print('The factor data has been copied from factor_expo data, and the factor will be'
                      'the FIRST one in strategy_data.factor_expo\n')
            else:
                print('Error: No factor data, please try to specify a factor!\n')
        elif isinstance(factor, pd.DataFrame):
            if self.strategy_data.factor.empty:
                self.strategy_data.factor = pd.Panel({'factor_one': factor})
            else:
                self.strategy_data.factor.iloc[0] = factor * 1

    def sft_part_3(self, *, base_obj=None):
        # 第三部分是除去不可投资的数据, 初始化或者重置策略持仓,
        # 处理barra base类, backtest类, 以及建立文件夹等零碎的事情

        # 除去不可交易或不可投资的数据
        # 注意，对策略数据的修改是永久性的，无法恢复，因此在使用过某个股票池的单因子测试后，对策略数据的再使用要谨慎
        if self.strategy_data.stock_pool == 'all':
            self.strategy_data.discard_untradable_data()
        else:
            self.strategy_data.discard_uninv_data()

        # 初始化持仓或重置策略持仓
        if self.position.holding_matrix.empty:
            self.initialize_position(self.strategy_data.factor.ix[0, self.holding_days, :])
        else:
            self.reset_position()

        # 如果有传入的base对象
        if isinstance(base_obj, factor_base):
            # 外来的base对象, 如果股票池和当前股票池一样, 且有因子暴露值, 就不用再次计算了
            if base_obj.base_data.stock_pool == self.strategy_data.stock_pool and \
                    not base_obj.base_data.factor_expo.empty:
                pass
            # 否则还是需要重设股票池, 进行计算
            else:
                # 将base的股票池改为当前股票池
                base_obj.base_data.stock_pool = self.strategy_data.stock_pool
                # 根据股票池, 生成因子值, 同时生成暴露值
                base_obj.construct_factor_base()
        # 没有外来传入的base对象, 则自己建立base对象
        else:
            base_obj = barra_base()
            base_obj.base_data.stock_pool = self.strategy_data.stock_pool
            base_obj.construct_factor_base()

        # 将base对象进行深拷贝, 然后将其赋给自己
        # 注意, 非常重要的一点是, 这里的base_obj的所有数据都是没有shift过的!!
        # 因为这里的base_obj的首要目的是归因, 归因不需要做shift, 因此之后的策略中要用到base_obj的地方,
        # 都一定要对base_obj中的数据进行lag才能使用
        self.base_obj = copy.deepcopy(base_obj)

        # 如果没有文件夹，则建立一个文件夹
        if not os.path.exists(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/'):
            os.makedirs(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/')
        # 建立画pdf的对象
        self.pdfs = PdfPages(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/allfigs.pdf')

    def sft_part_4(self, *, do_data_description=False, do_factor_corr_test=False, do_base_pure_factor=False):
        # 第四部分为, 1. 若各策略类有对原始因子数据的计算等, 可以在data description中进行
        # 2. 根据选择的一组因子base(可以是barra的, 也可以不是), 进行与当前因子的相关性检验
        # 3. 根据选择的一组因子base, 对当前因子进行提纯, 注意尽管与2差不多, 但是这里使得2, 3可以选择不同的base以及提纯方法
        # 4. 未来还可添加: 如果是基础财务因子, 还可以添加对净利润增长的预测情况

        # 如果有对原始数据的表述,则进行原始数据表述
        if do_data_description:
            self.data_description()
            print('Data description completed...\n')

        # 根据某一base, 做当前因子与其他因子的相关性检验
        if do_factor_corr_test:
            self.get_factor_corr_test(self.base_obj, use_factor_expo=not self.is_curr_factor_already_expo)

        # 如果要做基于barra base的纯因子组合，则要对因子进行提纯
        if do_base_pure_factor:
            self.get_pure_factor(self.base_obj, use_factor_expo=not self.is_curr_factor_already_expo)

    def sft_part_5(self, *, select_method=0, direction='+', ):
        # 第五部分为, 根据不同的单因子选股策略, 进行选股

        # 按策略进行选股
        if select_method == 0:
            # 简单分位数选股
            self.select_stocks(weight=1, direction=direction, select_ratio=[0.8, 1])
        elif select_method == 1:
            # 分行业选股
            self.select_stocks_within_indus(weight=3, direction=direction, select_ratio=[0.8, 1])
        elif select_method == 2 or select_method == 3:
            # 用构造纯因子组合的方法选股，2为组合自己是纯因子组合，3为组合相对基准是纯因子组合
            # 首先和计算纯因子一样，要base因子的暴露, 且同样需要lag
            lag_base_expo = self.base_obj.base_data.factor_expo.shift(1).reindex(
                major_axis=self.base_obj.base_data.factor_expo.major_axis)
            # 同样不能有country factor
            lag_base_expo_no_cf = lag_base_expo.drop('country_factor', axis=0)
            # # 构造纯因子组合，权重使用回归权重，即市值的根号
            # 初始化temp weight为'Empty'，即如果选股方法是2，则传入默认的benchmark weight
            temp_weight = 'Empty'
            if select_method == 2:
                # self.select_stocks_pure_factor_base(base_expo=lag_base_expo_no_cf, reg_weight=np.sqrt(
                #     self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction)
                self.select_stocks_pure_factor(base_expo=lag_base_expo_no_cf, reg_weight=np.sqrt(
                    self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction,
                                               benchmark_weight=temp_weight, is_long_only=False)
            if select_method == 3:
                if self.strategy_data.stock_pool == 'all':
                    temp_weight = data.read_data(['Weight_zz500'], ['Weight_zz500'], shift=True)
                    temp_weight = temp_weight['Weight_zz500']
                else:
                    # 注意股票池为非全市场时，基准的权重数据已经shift过了
                    temp_weight = self.strategy_data.benchmark_price.ix['Weight_' + self.strategy_data.stock_pool]

                self.select_stocks_pure_factor(base_expo=lag_base_expo_no_cf, reg_weight=np.sqrt(
                    self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction,
                                               benchmark_weight=temp_weight, is_long_only=True,
                                               use_factor_expo=not self.is_curr_factor_already_expo)

    def sft_part_6(self, *, bkt_obj=None, bkt_start=None, bkt_end=None, do_pa=True, do_active_pa=False,
                   discard_factor=[]):
        # 第六部分为, 1. 对策略选出的股票进行回测, 画图
        # 2. 如果有归因, 则对策略选出的股票进行归因

        # 如果有外来的backtest对象，则使用这个backtest对象，如果没有，则需要自己建立，同时传入最新持仓
        if isinstance(bkt_obj, backtest):
            bkt_obj.reset_bkt_position(self.position)
        else:
            bkt_obj = backtest(self.position, bkt_start=bkt_start, bkt_end=bkt_end)
        # 将回测的基准改为当前的股票池，若为all，则用默认的基准值
        if self.strategy_data.stock_pool != 'all':
            bkt_obj.reset_bkt_benchmark(['ClosePrice_adj_' + self.strategy_data.stock_pool])

        # 深拷贝一份回测对象, 赋给策略对象自己, 以方便查询回测, 画图, 归因等数据
        self.bkt_obj = copy.deepcopy(bkt_obj)

        import time
        start_time = time.time()
        # 回测、画图、归因
        self.bkt_obj.execute_backtest()
        print("exec time: {0} seconds\n".format(time.time() - start_time))
        self.bkt_obj.get_performance(foldername=self.strategy_data.stock_pool, pdfs=self.pdfs)
        print("plot time: {0} seconds\n".format(time.time() - start_time))

        # 如果要进行归因的话
        if do_pa:
            # 如果指定了要做超额收益的归因，且有股票池，则用相对基准的持仓来归因
            # 而股票池为全市场时的超额归因默认基准为中证500
            pa_benchmark_weight = None
            if do_active_pa:
                if self.strategy_data.stock_pool != 'all':
                    # 注意：策略里的strategy_data里的数据都是shift过后的，
                    # 而进行归因的数据和回测一样，不能用shift数据，要用当天最新数据
                    # 因此用于超额归因的benchmark weight数据需要重新读取
                    pa_benchmark_weight = data.read_data(['Weight_' + self.strategy_data.stock_pool],
                                                         ['Weight_' + self.strategy_data.stock_pool])
                    pa_benchmark_weight = pa_benchmark_weight['Weight_' + self.strategy_data.stock_pool]
                else:
                    temp_weight = data.read_data(['Weight_zz500'], ['Weight_zz500'])
                    pa_benchmark_weight = temp_weight['Weight_zz500']

            # base_obj在这里不用进行深拷贝, 因为归因的股票池=base_obj中base_data的股票池=策略的股票池,
            # 只要是一个股票池之间的计算, 就不会出现数据的丢失. 注意base_obj中的数据是没有shift过的, 专供归因所用
            self.bkt_obj.get_performance_attribution(outside_base=self.base_obj, benchmark_weight=pa_benchmark_weight,
                discard_factor=discard_factor, show_warning=False, pdfs=self.pdfs, is_real_world=True,
                foldername=self.strategy_data.stock_pool, real_world_type=2,
                enable_reading_pa_return=True)
        
    def sft_part_7(self, *, direction='+', bkt_start=None, bkt_end=None):
        # 第七部分为, 1. 根据回归算单因子的纯因子组合收益率
        # 2. 计算单因子的ic序列
        # 3. 单因子选股策略的n分位图
        # 4. 画单因子策略n分位图的long-short图

        # 画单因子组合收益率
        self.get_factor_return(weights=np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']),
                               holding_freq='w', direction=direction, start=bkt_start, end=bkt_end)
        # 画ic的走势图
        self.get_factor_ic(direction=direction, holding_freq='w', start=bkt_start, end=bkt_end)
        # 画分位数图和long short图
        self.plot_qgroup(self.bkt_obj, 5, direction=direction, value=1, weight=1)

    def sft_part_8(self):
        # 第八部分, 最后的收尾工作
        self.pdfs.close()
        
    def sft_test_outside_position(self):
        # holding = pd.read_csv('sue_holding500.csv', index_col=0, parse_dates=[2])
        # holding = holding.pivot_table(index='TradingDay', columns='SecuCode', values='WEIGHT')
        # new_col = []
        # for curr_stock in holding.columns:
        #     new_col.append(str(curr_stock).zfill(6))
        # holding.columns = new_col
        # holding = pd.read_csv('tarholding.csv', index_col=0, parse_dates=True)
        # self.position.holding_matrix = holding.reindex(self.position.holding_matrix.index,
        #                             self.position.holding_matrix.columns, method='ffill').fillna(0.0)
        # pass
        # holding = pd.read_hdf('opt_holding_tar_hs300', '123')
        # self.position.holding_matrix = holding.reindex(self.position.holding_matrix.index,
        #                                                method='ffill').fillna(0.0)
        pass

            
            
                
                
            
            
            
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            