#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:06:16 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages

from data import data
from backtest_data import backtest_data
from position import position

# 表现类，即根据账户的时间序列，计算各种业绩指标，以及进行画图
class performance(object):
    """ The class for performance calculation and plot.
    
    foo
    """
    
    def __init__(self, account_value, *, benchmark = None, holding_days=None, info_series=None,
                 tradedays_one_year=252, risk_free_rate=None, cash_ratio=None):
        """ Initialize performance object.
        
        foo
        """
        self.account_value = account_value
        self.benchmark = benchmark
        self.tradedays_one_year = tradedays_one_year
        self.risk_free_rate = risk_free_rate
        # 储存一下第一行的时间点，这个点为虚拟的所用动作开始前的点，资金为原始资金
        self.base_timestamp = self.account_value.index[0]
        # 简单收益率，这时去除第一项，这种序列用来计算统计量
        self.simple_return = self.account_value.pct_change().ix[1:]
        # 对数收益率，同上
        self.log_return = (np.log(self.account_value/self.account_value.shift(1))).ix[1:]
        # 根据simple return的index来初始化cash ratio, 因为cash ratio
        if isinstance(risk_free_rate, pd.Series):
            self.risk_free_rate = risk_free_rate
        else:
            self.risk_free_rate = pd.Series(0.0, index=self.simple_return.index)
        if isinstance(cash_ratio, pd.Series):
            self.cash_ratio = cash_ratio
        else:
            self.cash_ratio = pd.Series(0.0, index=self.simple_return.index)
        # 累积简单收益率，这种收益率用来画图，以及计算最大回撤等，注意这个收益率序列有起始项
        self.cum_simple_return = (self.simple_return+1).cumprod()-1
        # 累积对数收益
        self.cum_log_return = self.log_return.cumsum()
        # 拼接起始项
        base_series = pd.Series(0, index = [self.base_timestamp])
        self.cum_simple_return = pd.concat([base_series, self.cum_simple_return])
        self.cum_log_return = pd.concat([base_series, self.cum_log_return])
        
        # 策略账户净值序列
        self.net_account_value = self.account_value / self.account_value.ix[0]
        # 策略的调仓日，在有benchmark，计算策略的超额净值的时候会用到
        self.holding_days = holding_days
        # 其他信息，包括换手率，持股数等
        self.info_series = info_series
    
        # 对benchmark进行同样的计算，暂时只管一个benchmark数据
        if isinstance(benchmark, pd.Series):
            self.simple_return_bench = self.benchmark.pct_change().ix[1:]
            self.log_return_bench = (np.log(self.benchmark/self.benchmark.shift(1))).ix[1:]
            self.cum_simple_return_bench = (self.simple_return_bench+1).cumprod()-1
            self.cum_log_return_bench = self.log_return_bench.cumsum()
            self.cum_simple_return_bench = pd.concat([base_series, self.cum_simple_return_bench])
            self.cum_log_return_bench = pd.concat([base_series,self.cum_log_return_bench])

            # 下面开始计算超额净值及超额收益, 首先要先计算股票资产的超额收益
            # 然后才能结合现金资产和股票资产的比例来计算组合整体的超额收益
            # 注意, 有现金资产的情况下, benchmark的比例只会和股票资产比例一致,
            # 因此active的部分其实只有股票资产部分, 即股票的active return + 现金的return

            # 计算股票资产的收益
            self.get_stock_asset_return()
            # 计算组合的超额收益
            self.get_active_nav_and_return()

    def get_stock_asset_return(self):
        # 首先计算股票资产的账户净值, 用股票部分的净值, 减去benchmark乘以股票资产比例的净值
        # 一个问题是, 如果第一天就是调仓日, 则为了计算第一天的股票收益率, 但初始时刻的股票资产价值为0
        # 解决办法是, 设置初始时刻的现金比例和第一天一样, 这样股票资产的初始价值就可比了
        adjusted_cash_ratio = pd.concat([pd.Series(self.cash_ratio.iloc[0], index=[self.base_timestamp]),
                                         self.cash_ratio], axis=0)
        # 拥有股票资产的净值序列, 如何计算它的收益呢, 因为每次调仓可能有现金资产和股票资产的流动
        # 因此不能直接用股票资产的净值序列来计算, 由于调仓时间在一天中不确定, 需要区别对待
        # 在非调仓日, 股票资产比例直接使用昨天的比例就可以了
        # 但是在调仓日,股票资产的收益, 在昨天结算到今天换仓期间, 是由昨天的比例带来的
        # 在今天换仓到今天结算期间是由今天最新的比例带来的, 因此需要分别对待
        # 从info series中取得换仓时刻前的持仓价值(注意是用换仓价格算出的)
        assert isinstance(self.info_series, pd.DataFrame), 'Error: Please enter appropriate info series!\n'
        holding_value = self.info_series['holding_value']
        # holding_value为0的地方说明这一天没有换仓, 因此holding value按照今天的account value计算
        # 即这一天所有的收益都是用昨天的股票资产持仓带来的
        # 注意没有换仓的天数, 其股票收益就是直接按照这个算出来的
        holding_value = pd.concat([pd.Series(self.account_value.iloc[0], index=[self.base_timestamp]),
                                   holding_value], axis=0)
        holding_value = holding_value.where(holding_value!=0, self.account_value)
        holding_value = holding_value.div(self.account_value.iloc[0])

        # 第一步计算昨天到今天换仓时间的收益, 以昨天的资产比例为基础
        base_cash_value1 = self.net_account_value.mul(adjusted_cash_ratio).shift(1)
        base_stock_value1 = self.net_account_value.mul(1 - adjusted_cash_ratio).shift(1)
        # 计算方法为, 首先用昨天的现金比例和今天的现金收益算出现金带来的资产增值
        # 注意, 默认现金收益是隔夜带来的
        value_added_cash1 = base_cash_value1 * (np.exp(self.risk_free_rate) - 1)
        # 用总的资产增值减去现金资产带来的增值, 则等于股票资产带来的增值
        value_added_stock1 = holding_value.sub(self.net_account_value.shift(1)).sub(value_added_cash1)
        # 昨天到今天调仓时的收益, 即股票带来的价值, 除以股票资产的基础值
        self.log_return_equity1 = np.log(value_added_stock1.div(base_stock_value1) + 1).ix[1:]

        # 第二步计算今天换仓到今天结算的收益, 以今天换仓后的资产比例为基础
        # 这一部分没有现金收益, 且换仓的手续费也被算在了这里
        # 需要或取换仓完成后的股票资产比例, 具体方法是, 因为换仓后到结束的现金资产无收益
        # 因此今天结算时的现金数目等于今天换仓后的现金数目
        base_cash_value2 = self.net_account_value*adjusted_cash_ratio
        base_stock_value2 = holding_value - base_cash_value2
        # 因为没有现金增值, 因此所有增值都是股票带来的增值
        value_added_stock2 = self.net_account_value.sub(holding_value)
        # 今天调仓到今天结算的收益, 即股票带来的价值, 除以股票资产的基础值
        self.log_return_equity2 = np.log(value_added_stock2.div(base_stock_value2) + 1).ix[1:]

        # 将需要的数据储存下来, 这些数据在之后的计算中会用到, 如计算超额净值和收益
        self.return_data = pd.DataFrame({'log_return_equity1': self.log_return_equity1,
            'log_return_equity2': self.log_return_equity2, 'log_return_bench': self.log_return_bench,
            'risk_free_rate': self.risk_free_rate, 'base_cash_value1': base_cash_value1,
            'base_stock_value1': base_stock_value1, 'base_cash_value2': base_cash_value2,
            'base_stock_value2': base_stock_value2})
        # 不要第一行的那个基础起始点
        self.return_data = self.return_data.iloc[1:]

    # 计算组合的超额收益和超额净值的函数
    # 注意, 有现金资产的情况下, benchmark的比例只会和股票资产比例一致,
    # 因此active的部分其实只有股票资产部分, 即股票的active return + 现金的return
    # 超额净值，注意超额净值并不是账户净值减去基准净值，因为超额净值要考虑到策略在调仓日对基准份额的调整
    # 超额净值的算法为，每个调仓周期之内的股票超额净值序列为exp（策略累计收益序列）- exp（基准累计收益序列）
    # 周期内整体超额收益为: 股票资产超额收益净值与现金净值按照对应比例相加
    # 不同调仓周期之间的净值为：这个调仓周期内的超额净值序列加上上一个调仓周期的最后一天的净值
    def get_active_nav_and_return(self):
        # benchmark的账户净值
        # 第一个不为nan的数
        bench_base_value = self.benchmark[self.benchmark.notnull()].ix[0]
        self.net_benchmark = self.benchmark / bench_base_value

        # 有一个问题是, benchmark的价格并无vwap, 只有收盘价价格, 因此benchmark的调仓价格只能是收盘价
        # 所以现在统一把benchmark都设定在用收盘价换仓, 因此, 其第二部分的return是0,
        # 且股票资产基数按照昨天的股票资产比例来计算(参照第一部分的收益率计算)

        # shift一天, 即一个周期第一天为上个调仓日后的第一天, 一直到下一个调仓日的当天
        self.return_data['mark'] = self.holding_days.asof(self.return_data.index).shift(1).\
            replace(pd.tslib.NaT, self.account_value.index[0])
        grouped = self.return_data.groupby('mark')

        # 定义每个调仓周期内的收益序列计算方法
        # 每个调仓周期内用周期内股票资产的净值增长加上现金部分的净值增长
        # 每个调仓周期内的资产配置比例起始点, 选为上一个调仓周期的最后一天的比例
        def func_intra_nav(x, *, get_deviation=False, get_node_value=False):
            # 股票资产的收益率序列, 分为调仓期前的部分和调仓后的部分
            stock_return_before = x['log_return_equity1'] + x['log_return_equity2']
            # 最后一天, 即下一个调仓日的
            stock_return_before.iloc[-1] -= x.ix[-1, 'log_return_equity2']
            # 先计算从第一天, 到最后一天(即调仓日那天)换仓前的那部分净值增值序列
            stock_nav_change_before = np.exp(stock_return_before.cumsum()) - 1
            # benchmark的净值增长序列
            bench_nav_change_before = np.exp(x['log_return_bench'].cumsum()) - 1
            # 如果需要计算的是intra_holding_deviation(即股票多空头的偏差), 则这个时候的信息已经足够了
            if get_deviation:
                intra_deviation = stock_nav_change_before - bench_nav_change_before
                # 最后一天, 因为是换仓日, 且假设了benchmark在收盘换仓, 因此deviation一定是0
                # 注意, 在整个回测的最后一个调仓周期, 调仓周期的最后一天不一定是调仓日
                if intra_deviation.index[-1] in self.holding_days:
                    intra_deviation.iloc[-1] = 0.0
                    return intra_deviation
            # 如果不是计算intra deviation, 则继续我们的计算
            # 同时计算现金部分, 现金部分的收益全部在这一部分实现
            cash_nav_change = np.exp(x['risk_free_rate'].cumsum()) - 1
            # 这部分净值序列的起始资产比例为对应的base1, 即上一个调仓日结算时的比例
            base_cash_ratio1 = x.ix[0, 'base_cash_value1'] / (x.ix[0, 'base_cash_value1'] +
                x.ix[0, 'base_stock_value1'])
            # 于是净值增长的序列
            active_nav_change_before = (stock_nav_change_before - bench_nav_change_before) * \
                (1 - base_cash_ratio1) + cash_nav_change * base_cash_ratio1

            # 现在来计算调仓后的部分
            stock_return_after = x.ix[-1, 'log_return_equity2']
            stock_nav_change_after = np.exp(stock_return_after) - 1
            # 由于假设, 1. benchmark均在收盘价换仓, 即benchmark的所有收益分配到第一部分
            # 2. 现金的无风险收益算在隔夜上, 同样也全部分配到了第一部分
            # 因此第二部分的收益就只有换仓后的股票资产的收益
            # 而其股票资产的基准比例为对应的base2, 即换仓时价值对应的那个比例
            base_cash_ratio2 = x.ix[-1, 'base_cash_value2'] / (x.ix[-1, 'base_cash_value2'] +
                x.ix[-1, 'base_stock_value2'])
            active_nav_change_after = stock_nav_change_after * (1 - base_cash_ratio2)

            # 最后总的nav序列为前后两个nav之和
            intra_active_nav_change = active_nav_change_before + active_nav_change_after

            # 如果只是为了得到周期最后一天的净值, 则只返回最后一个, 否则返回一个序列
            if get_node_value:
                return intra_active_nav_change.iloc[-1]
            else:
                return intra_active_nav_change


            # active_stock_nav = np.exp(x['log_return_equity'].cumsum()) - np.exp(x['log_return_bench'].cumsum())
            # cash_nav = np.exp(x['risk_free_rate'].cumsum())
            # # 如果是为了计算调仓期内的股票多头空头偏差, 则直接返回股票资产的超额收益
            # if get_deviation:
            #     return active_stock_nav
            # # 找到距离这一期第一天最近的一个调仓日
            # latest_holidng_day = self.holding_days.asof(x.index[0])
            # # 根据这个调仓日寻找比例
            # base_cash_ratio = self.cash_ratio.ix[latest_holidng_day]
            # intra_holding_nav = active_stock_nav * (1 - base_cash_ratio) + cash_nav * base_cash_ratio
            #
            # return intra_holding_nav

        intra_holding = grouped.apply(func_intra_nav).reset_index(0, drop=True)

        # # 算每个调仓周期的最后一天的周期内净值
        # def func_holding_node_nav(x):
        #     active_stock_nav = np.exp(x['log_return_equity'].sum()) - np.exp(x['log_return_bench'].sum())
        #     cash_nav = np.exp(x['risk_free_rate'].sum())
        #     # 找到距离这一期第一天最近的一个调仓日
        #     latest_holidng_day = self.holding_days.asof(x.index[0])
        #     # 根据这个调仓日寻找比例
        #     base_cash_ratio = self.cash_ratio.ix[latest_holidng_day]
        #     holding_node_nav = active_stock_nav * (1 - base_cash_ratio) + cash_nav * base_cash_ratio
        #
        #     return holding_node_nav

        holding_node_value = grouped.apply(func_intra_nav, get_node_value=True)
        # 此后的每个周期内的净值，都需要加上此前所有周期的最后一天的净值，注意首先需要shift一个调仓周期
        # 因为每个周期结束的净值, 其label是上一个周期的最后一天(即上一个调仓日)
        # 在将index设置为每天后(而非每个调仓周期), 需要再次shift一天, 因为当前周期的最后一天
        # 即当前周期的那个调仓日, 其不需要加上这个周期结束时的数据, 而是在下一天才开始加入
        holding_node_value_cum = holding_node_value.shift(1).cumsum().fillna(0.0). \
            reindex(intra_holding.index, method='ffill').shift(1).fillna(0.0)
        active_nav_change = holding_node_value_cum + intra_holding
        self.active_nav = active_nav_change + 1.0
        self.active_nav = pd.concat([pd.Series(1.0, index=[self.base_timestamp]),
                                                   self.active_nav], axis=0)
        # 计算用超额净值得到的超额收益序列，用这个序列来计算超额收益的统计量，更符合实际
        self.active_log_return = np.log(self.active_nav.
                                        div(self.active_nav.shift(1))).ix[1:]
        self.cum_active_log_return = self.active_log_return.cumsum()
        self.cum_active_log_return = pd.concat([pd.Series(0.0, index=[self.base_timestamp]),
                                                self.cum_active_log_return], axis=0)

        # 每个调仓周期内的累计超额收益, 这是一个在其他地方可能会用到的数据,
        # 因为这代表着在调仓期内, 因为超额收益带来的多头组合和空头基准的偏离度
        # 因此把这个量提取出来, 不用shift, 因为归因里会自己shift
        self.intra_holding_deviation = grouped.apply(func_intra_nav, get_deviation=True). \
            reset_index(0, drop=True)
        pass


            
    # 定义各种计算指标的函数，这里都用对数收益来计算
    # 年化收益
    @staticmethod
    def annual_return(cum_return_series, tradedays_one_year):
        return cum_return_series.ix[-1] / (cum_return_series.size-1) * tradedays_one_year
        
    # 年化波动率
    @staticmethod
    def annual_std(return_series, tradedays_one_year):
        return return_series.std() * np.sqrt(tradedays_one_year)
        
    # 年化夏普比率
    @staticmethod
    def annual_sharpe(annual_return, annual_std, risk_free_rate):
        return (annual_return - risk_free_rate) / annual_std
        
    # 最大回撤率，返回值为最大回撤率，以及发生的时间点的位置
    @staticmethod
    def max_drawdown(account_value_series):
        past_peak = account_value_series.ix[0]
        max_dd = 0
        past_peak_loc = 0
        low_loc = 0
        temp_past_peak_loc = 0
        for i, curr_account_value in enumerate(account_value_series):
            if curr_account_value >= past_peak:
                past_peak = curr_account_value
                temp_past_peak_loc = i
            elif (curr_account_value - past_peak) / past_peak < max_dd:
                max_dd = (curr_account_value - past_peak) / past_peak
                low_loc = i
                past_peak_loc = temp_past_peak_loc
        return max_dd, past_peak_loc, low_loc

    # 计算年化calmar比率
    @staticmethod
    def annual_calmar_ratio(annual_return, max_drawdown):
        return annual_return / max_drawdown

    # 计算年化sortino比率
    @staticmethod
    def annual_sortino_ratio(return_series, annual_return, *, return_target=0.0,
                             tradedays_one_year=252, risk_free_rate=0.0):
        under_performance_return = return_series - return_target
        under_performance_return = under_performance_return.where(under_performance_return<0, 0.0)
        sortino = (annual_return - risk_free_rate) / (under_performance_return.std() * np.sqrt(tradedays_one_year))
        return sortino
        
    # 接下来计算与benchmark相关的指标
    
    # 年化超额收益
    def annual_active_return(self):
        return self.cum_active_log_return.ix[-1] * self.tradedays_one_year / \
               (self.cum_active_log_return.size - 1)
               
    # 年化超额收益波动率
    def annual_active_std(self):
        return self.active_log_return.std() * np.sqrt(self.tradedays_one_year)
        
    # 年化信息比
    def info_ratio(self, annual_active_return, annual_active_std):
        return annual_active_return / annual_active_std
        
    # 胜率
    def win_ratio(self):
        return self.active_log_return.ix[self.active_log_return>0].size / \
               self.active_log_return.size
        
    # 计算并输出各个指标
    def get_performance(self, *, foldername=''):
        
        annual_r = performance.annual_return(self.cum_log_return, self.tradedays_one_year)
        annual_std = performance.annual_std(self.log_return, self.tradedays_one_year)
        # 无风险收益这里暂时用平均值替代一下, 之后需要修改
        annual_sharpe = performance.annual_sharpe(annual_r, annual_std, self.risk_free_rate.mean())
        max_dd, peak_loc, low_loc = performance.max_drawdown(self.account_value)
        annual_calmar = performance.annual_calmar_ratio(annual_r, max_dd)
        # 无风险收益这里暂时用平均值替代一下, 之后需要修改
        annual_sortino = performance.annual_sortino_ratio(self.log_return, annual_r, return_target=0.0,
            tradedays_one_year=self.tradedays_one_year, risk_free_rate=self.risk_free_rate.mean())
        if isinstance(self.benchmark, pd.Series):
            annual_ac_r = self.annual_active_return()
            annual_ac_std = self.annual_active_std()
            annual_info_ratio = self.info_ratio(annual_ac_r, annual_ac_std)
            max_dd_ac, peak_loc_ac, low_loc_ac = performance.max_drawdown(self.active_nav)
            annual_ac_calmar = performance.annual_calmar_ratio(annual_ac_r, max_dd_ac)
            win_ratio = self.win_ratio()
        else:
            annual_ac_r = np.nan
            annual_ac_std = np.nan
            annual_info_ratio = np.nan
            max_dd_ac = np.nan
            peak_loc_ac = 0
            low_loc_ac = 0
            annual_ac_calmar = np.nan
            win_ratio = np.nan


        # 输出指标
        target_str = 'Stats START ------------------------------------------------------------------------\n' \
                     'The stats of the strategy (and its performance against benchmark) is as follows:\n' \
                     'Annual log return: {0:.2f}%\n' \
                     'Annual standard deviation of log return: {1:.2f}%\n' \
                     'Annual Sharpe ratio: {2:.2f}\n' \
                     'Max drawdown: {3:.2f}%\n' \
                     'Max drawdown happened between {4} and {5}\n' \
                     'Annual Calmar ratio: {6:.2f}\n' \
                     'Annual Sortino ratio: {7:.2f}\n' \
                     'Averge cash ratio: {8:.2f}%\n '.format(
            annual_r*100, annual_std*100, annual_sharpe, max_dd*100, self.cum_log_return.index[peak_loc],
            self.cum_log_return.index[low_loc], annual_calmar, annual_sortino, self.cash_ratio.mean()*100
            )

        if isinstance(self.benchmark, pd.Series):
            target_str = target_str + \
                         'Annual active log return: {0:.2f}%\n' \
                         'Annual standard deviation of active log return: {1:.2f}%\n' \
                         'Annual information ratio: {2:.2f}\n' \
                         'Max drawdown of active account value: {3:.2f}%\n' \
                         'Max drawdown happened between {4} and {5}\n' \
                         'Annual active Calmar ratio: {6:.2f}\n' \
                         'Winning ratio: {7:.2f}%\n'.format(
            annual_ac_r * 100, annual_ac_std * 100, annual_info_ratio, max_dd_ac * 100,
            self.cum_log_return.index[peak_loc_ac], self.cum_log_return.index[low_loc_ac],
            annual_ac_calmar, win_ratio * 100,
            )

        if isinstance(self.info_series, pd.DataFrame):
            target_str = target_str + \
                         'Average turnover ratio: {0:.2f}%\n'\
                         'Average number of stocks holding: {1:.2f}\n' \
                         'Average of untradable stocks weight in target holding: {2:.2f}% \n' \
                         'Average of untradable stocks weight in current holding: {3:.2f}% \n' \
                         'Average of buy-but-cap stocks weight to total holding: {4:.2f}% \n' \
                         'Average of sell-but-bottom stocks weight to total holding: {5:.2f}% \n' \
                         'Average holding difference between target position and real position: {6:.2f}% \n' \
                         ''.format(
            self.info_series.ix[self.holding_days, 'turnover_ratio'].mean() * 100,
            self.info_series.ix[:, 'holding_num'].mean(),
            self.info_series.ix[self.holding_days, 'target_sus'].mean() * 100,
            self.info_series.ix[self.holding_days, 'holding_sus'].mean() * 100,
            self.info_series.ix[self.holding_days, 'buy_cap'].mean() * 100,
            self.info_series.ix[self.holding_days, 'sell_bottom'].mean() * 100,
            self.info_series.ix[self.holding_days, 'holding_diff'].mean() * 100
            )

        target_str = target_str + \
            'Stats END --------------------------------------------------------------------------\n'

        print(target_str)

        # 将输出写到txt中
        with open(str(os.path.abspath('.'))+'/'+foldername+'/performance.txt',
                  'w', encoding='GB18030') as text_file:
            text_file.write(target_str)


    # 画图
    def plot_performance(self, *, foldername='', pdfs=None):
        
        # 第一张图为策略自身累积收益曲线
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.plot(self.cum_log_return*100, 'b-', label = 'Strategy')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Log Return (%)')
        ax1.set_title('The Cumulative Log Return of The Strategy (and The Benchmark)')
        plt.xticks(rotation=30)
        plt.grid()
        
        # 如有benchmark，则加入benchmark的图
        if isinstance(self.benchmark, pd.Series):
            plt.plot(self.cum_log_return_bench*100, 'r-', label = 'Benchmark')
            
        ax1.legend(loc = 'best')
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/CumLog.png', dpi=1200)
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf')
        
        # 第二张图为策略超额收益曲线，只有在有benchmark的时候才画
        if isinstance(self.benchmark, pd.Series):
            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            plt.plot(self.cum_active_log_return*100, 'b-')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('The Cumulative Active Log Return of The Strategy')
            plt.xticks(rotation=30)
            plt.grid()

            plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/ActiveCumLog.png', dpi=1200)
            if isinstance(pdfs, PdfPages):
                plt.savefig(pdfs, format='pdf')
            
        # 第三张图为策略账户净值曲线
        f3 = plt.figure()
        ax3 = f3.add_subplot(1,1,1)
        plt.plot(self.net_account_value, 'b-', label = 'Strategy')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Account Value')
        ax3.set_title('The Net Account Value of The Strategy (and The Benchmark)')
        plt.xticks(rotation=30)
        plt.grid()
        
        # 如有benchmark，则加入benchmark的图
        if isinstance(self.benchmark, pd.Series):
            plt.plot(self.net_benchmark, 'r-', label = 'Benchmark')
            
        ax3.legend(loc = 'best')
        plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/NetValue.png', dpi=1200)
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf')
        
        # 第四张图为策略超额收益净值，只有在有benchmark的时候才画
        if isinstance(self.benchmark, pd.Series):
            f4 = plt.figure()
            ax4 = f4.add_subplot(1,1,1)
            plt.plot(self.active_nav, 'b-')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Active Net Value')
            ax4.set_title('The Active Net Value of The Strategy')
            plt.xticks(rotation=30)
            plt.grid()

            plt.savefig(str(os.path.abspath('.')) + '/'+foldername+'/ActiveNetValue.png', dpi=1200)
            if isinstance(pdfs, PdfPages):
                plt.savefig(pdfs, format='pdf')

        # 第五张图画策略的持股数曲线
        if isinstance(self.info_series, pd.DataFrame):
            f5 = plt.figure()
            ax5 = f5.add_subplot(1,1,1)
            plt.plot(self.info_series.ix[:, 'holding_num'], 'b-')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Number of Stocks')
            ax5.set_title('The Number of Stocks holding of The Strategy')
            plt.xticks(rotation=30)
            plt.grid()

            plt.savefig(str(os.path.abspath('.')) + '/'+foldername+'/NumStocksHolding.png', dpi=1200)
            plt.savefig(pdfs, format='pdf')

        # 第六张图画策略的现金比例图
        if (self.cash_ratio >= 1e-6).any():
            f6 = plt.figure()
            ax6 = f6.add_subplot(1, 1, 1)
            plt.plot(self.cash_ratio, 'b-')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Ratio of Cash Asset')
            ax6.set_title('The Ratio of Cash Asset in Total Portfolio')
            plt.xticks(rotation=30)
            plt.grid()

            plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/CashRatio.png', dpi=1200)
            plt.savefig(pdfs, format='pdf')

            
            
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
