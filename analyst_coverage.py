#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:51:44 2017

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
from pandas.tools.plotting import table
from pandas.stats.fama_macbeth import fama_macbeth
from statsmodels.discrete.discrete_model import Poisson
import pathos.multiprocessing as mp
from linearmodels import FamaMacBeth

from single_factor_strategy import single_factor_strategy
from database import database
from data import data
from strategy_data import strategy_data
from strategy import strategy

# 分析师预测覆盖因子的单因子策略

class analyst_coverage(single_factor_strategy):
    """Analyst coverage single factor strategy class.
    
    foo
    """
    def __init__(self, start_date=pd.Timestamp('2007-01-01'), end_date=pd.Timestamp('2018-01-17')):
        single_factor_strategy.__init__(self)
        # 该策略用于取数据的database类
        self.db = database(start_date=start_date, end_date=end_date)
        self.db.get_trading_days()
        self.db.get_labels()

    # 取计算分析师覆盖因子的原始数据
    def get_coverage_number(self):
        # 取每天，每只股票的净利润预测数据
        sql_query = "select create_date, code, count(forecast_profit) as num_fore from " \
                    "((select id, code, organ_id, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) "\
                    "group by create_date, code " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        coverage = original_data.pivot_table(index='create_date', columns='code', values='num_fore')
        # 因为发现gg数据库里的数据，每天分不同时间点有不同的数据，因此要再resample一下
        coverage = coverage.resample('d').sum().dropna(axis=0, how='all')
        # 对过去90天进行rolling求和，注意，coverage的日期索引包含非交易日
        rolling_coverage = coverage.rolling(90, min_periods=0).apply(lambda x:np.nansum(x))
        # 策略数据，注意shift
        rolling_coverage = rolling_coverage.reindex(self.strategy_data.stock_price.major_axis).shift(1)
        # 将其储存到raw_data中，顺便将stock code重索引为标准的stock code
        self.strategy_data.raw_data = pd.Panel({'coverage':rolling_coverage}, major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

    # 取每支股票的每期报表的第一次发布的数据, 以用来选择分析师数据的rolling window
    def get_fra_pub_date(self):
        sql_query = 'select b.InfoPublDate, b.EndDate, a.SecuCode from (' \
                    'select distinct CompanyCode, SecuCode from SmartQuant.dbo.ReturnDaily) a ' \
                    'left join (select InfoPublDate, EndDate, CompanyCode from LC_BalanceSheetAll ' \
                    'where IfMerged=1 and IfAdjusted=2 and IfComplete!=3) b on a.CompanyCode=b.CompanyCode ' \
                    'order by SecuCode, InfoPublDate, EndDate'
        fra_pub_date = self.db.jydb_engine.get_original_data(sql_query)
        # 有一些重复出现的报告数据, 要把这些数据删除掉, 这些数据信息都是一样的, 并未调整, 但是在不同的年份重复出现
        fra_pub_date = fra_pub_date.drop_duplicates(subset=['SecuCode', 'EndDate'], keep='first')

        self.fra_pub_date = fra_pub_date
        pass


    # 寻找起始时间的函数
    @staticmethod
    def get_start_time(s, *, end_time, fra_interval=2):
        # 在给定日期之前的所有财报发布日
        past_dates = s[s <= end_time]
        # 开始日为给定日期前2个财报发布的时间段, 可以更改2这个数字
        if past_dates.shape[0] >= fra_interval:
            start_time = past_dates.iloc[-fra_interval]
        else:
            start_time = pd.Timestamp('1990-01-01')

        # 如果前两个财报和当前时间间隔过短(一般是因为一季度报表和年度报表时间太近, 有时还在同一天)
        # 如果短于60天, 则寻找前3个财报的时间段
        if (end_time-start_time).days < 60:
            if past_dates.shape[0] >= fra_interval+1:
                start_time = past_dates.iloc[-(fra_interval+1)]
            else:
                start_time = pd.Timestamp('1990-01-01')

        return start_time

    # 定义在grouped数据(对同一股票的过去一段时间的分析师预测数据)里, 取最近一次预测的最近一个财年的预测值的函数
    @staticmethod
    def get_recent_report_recent_fy(x):
        # 删除重复报告, 只得到每个机构的每个分析师对每个财年做出的最近的那个报告
        recent_report = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'],
                                          keep='last')
        # 对这个报告的机构,分析师进行分组,取里面的第一个预测的年份
        # 要对这个年份进行排序, 取最大的那一个, 因为有可能出现, 过去一段时间(如90天), 机构预测的第一个财年不一样的情况
        # 这种多出现于4月份这种发财报的时候, 排序后取最大的那一个年份, 就是最新报告中的最近财年.
        # 注意:这里的假设是, 分析师不会跳着发布预测, 即不会出现2016年没发布2017年的预测, 直接发布2018年的
        recent_fy = recent_report.groupby(['organ_id', 'author'])['Time_year']. \
            apply(lambda x:x.dropna().iloc[0] if x.dropna().size>0 else np.nan).sort_values(ascending=True)
        # 如果当前股票的预测财年都是nan, 则返回nan, 这种情况很少见, 否则则返回第一个值,即最大值
        if recent_fy.dropna().size == 0:
            return np.nan
        else:
            return recent_fy.iloc[0]

    # 定义计算ni的std与均值的函数
    @staticmethod
    def ni_std(x):
        # 首先去除重复项
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')[
            ['Time_year', 'forecast_profit']]
        # 将每一只股票的唯一预测数据, 按照机构和作者进行分组,并取最近的那一年
        # recent_forecast = unique_x.groupby(['organ_id', 'author'])['forecast_profit'].nth(0)
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        # 取算出的财年的数据
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算std
        recent_std = recent_forecast.std()
        # 必须有3个以上的独立分析师数据, 否则数据为nan
        if recent_forecast.dropna().shape[0] >= 3:
            return recent_std
        else:
            return np.nan

    @staticmethod
    def ni_mean(x):
        # 首先去除重复项
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')
        # 将每一只股票的唯一预测数据, 按照机构和作者进行分组,并取最近的那一年
        # recent_forecast = unique_x.groupby(['organ_id', 'author'])['forecast_profit'].nth(0)
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算mean
        recent_mean = recent_forecast.mean()
        # 必须有3个以上的独立分析师数据, 否则数据为nan
        if recent_forecast.dropna().shape[0] >= 3:
            return recent_mean
        else:
            return np.nan


    @staticmethod
    def ni_count(x):
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算count
        recent_count = recent_forecast.count()
        return recent_count

    @staticmethod
    def simple_count(x):
        unique_x = x.drop_duplicates(subset=['organ_id', 'author'], keep='last')
        recent_count = unique_x.shape[0]
        return recent_count

    # 计算滚动期内的唯一分析师分析数据
    def get_unique_coverage_number(self, *, rolling_days=90):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 先将时间期内的所有数据都取出来
        sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        # 先构造一个pivot table,主要目的是为了取时间
        date_mark = original_data.pivot_table(index='create_date', columns='code', values='forecast_profit')
        # 因为数据有每天不同时点的数据,因此要resample
        date_mark = date_mark.resample('d').mean().dropna(axis=0, how='all')
        # 建立储存数据的dataframe
        coverage = date_mark * np.nan
        # # 建立储存disp与ep的dataframe, disp为有效公司的下一年的预测ni的std/price
        # # ep为有效公司的下一年的预测ni/price, 有效公司为rolling days内至少有3个独立预测值的公司
        # coverage_disp = coverage * np.nan
        # coverage_ep = coverage * np.nan

        # # 所有数据的透视表
        # original_table = original_data.pivot_table(index='create_date', columns='code',
        #         values=['forecast_profit', 'organ_id', 'author', 'Time_year'])
        # original_table = original_table.T.to_panel().transpose(1, 0, 2)

        # 根据得到的时间索引进行循环
        for cursor, time in enumerate(date_mark.index):
            # 取最近的90天
            end_time = time
            # start_time = end_time - pd.DateOffset(days=rolling_days-1)
            # # 满足最近90天条件的项
            # condition = np.logical_and(original_data['create_date'] >= start_time,
            #                            original_data['create_date'] <= end_time)
            # recent_data = original_data[condition]

            # # 对股票分组
            # grouped = recent_data.groupby('code')
            # 分组汇总, 若机构id,作者,预测年份都一样,则删除重复的项,然后再汇总一共有多少预测ni的报告(ni值非nan)
            # curr_coverage = grouped.apply(lambda x:x.drop_duplicates(subset=['organ_id', 'author',
            #                               'Time_year'])['forecast_profit'].count())

            # curr_coverage = grouped.apply(analyst_coverage.simple_count)
            # coverage_std = grouped.apply(analyst_coverage.ni_std)
            # coverage_mean = grouped.apply(analyst_coverage.ni_mean)

            start_time = self.fra_pub_date.groupby('SecuCode')['InfoPublDate'].apply(
                analyst_coverage.get_start_time, end_time=end_time)
            # 循环股票
            for stock_id, curr_start_time in start_time.iteritems():
                time_condition = np.logical_and(original_data['create_date']>=curr_start_time,
                                                original_data['create_date']<=end_time)
                stock_condition = (original_data['code'] == stock_id)
                curr_data = original_data[np.logical_and(time_condition, stock_condition)]
                curr_coverage = analyst_coverage.simple_count(curr_data)
                coverage.ix[time, stock_id] = curr_coverage


            # # 储存此数据
            # coverage.ix[time, :] = curr_coverage
            # coverage_disp.ix[time, :] = coverage_std
            # coverage_ep.ix[time, :] = coverage_mean

            print(time)
        pass


        # 策略数据需要shift, 先shift再重索引为交易日,这样星期一早上能知道上周末的信息,而不是上周5的信息
        coverage = coverage.shift(1)
        # 将其储存到raw_data中,对交易日和股票代码的重索引均在这里完成
        self.strategy_data.raw_data = pd.Panel({'coverage':coverage}, major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

        # # 读取市值数据
        # mv = data.read_data(['FreeMarketValue'], shift=False)
        # mv = mv['FreeMarketValue']
        # # 重索引
        # coverage_disp = coverage_disp.reindex(index=mv.index, columns=mv.columns)
        # coverage_ep = coverage_ep.reindex(index=mv.index, columns=mv.columns)
        # coverage_disp = coverage_disp/mv
        # coverage_ep = coverage_ep/mv
        # # 储存数据
        # self.strategy_data.raw_data['coverage_disp'] = coverage_disp.shift(1)
        # self.strategy_data.raw_data['coverage_ep'] = coverage_ep.shift(1)
        pass

    # 计算滚动期内的唯一分析师分析数据
    def get_unique_coverage_number_parallel(self, *, rolling_days=120):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 先将时间期内的所有数据都取出来
        sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, type_id, author, create_date from CMB_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from CMB_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) where type_id != 28" \
                    "order by create_date, code "

        sql_query1 = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, type_id, author, create_date from CMB_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from CMB_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) where type_id in (21, 22, 25)" \
                    "order by create_date, code "

        sql_query2 = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, type_id, author, create_date from CMB_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from CMB_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) where type_id in (23, 24, 26, 98) " \
                    "order by create_date, code "

        original_data = self.db.gg_engine.get_original_data(sql_query)
        original_data1 = self.db.gg_engine.get_original_data(sql_query1)
        original_data2 = self.db.gg_engine.get_original_data(sql_query2)
        # 先构造一个pivot table,主要目的是为了取时间
        date_mark = original_data.pivot_table(index='create_date', columns='code', values='forecast_profit')
        # 因为数据有每天不同时点的数据,因此要resample
        date_mark = date_mark.resample('d').mean().dropna(axis=0, how='all')
        # 建立储存数据的dataframe
        coverage = date_mark * np.nan
        coverage2 = date_mark * np.nan
        # # 建立储存disp与ep的dataframe, disp为有效公司的下一年的预测ni的std/price
        # # ep为有效公司的下一年的预测ni/price, 有效公司为rolling days内至少有3个独立预测值的公司
        # coverage_disp = coverage * np.nan
        # coverage_ep = coverage * np.nan
        fra_pub_date = self.fra_pub_date * 1
        # 暂时只算周末的值
        self.generate_holding_days(holding_freq='w', loc=-1, start_date=self.db.start_date,
                                   end_date=self.db.end_date)
        holding_days = self.holding_days

        # 计算每期的coverage的函数
        def one_time_coverage(cursor):
            end_time = holding_days.index[cursor]
            # # 得到对应位置的时间索引,取为截至时间
            # # end_time = date_mark.index[cursor]
            # start_time = end_time - pd.DateOffset(days=rolling_days - 1)
            # # 满足最近90天条件的项
            # condition = np.logical_and(original_data['create_date'] >= start_time,
            #                            original_data['create_date'] <= end_time)
            # recent_data = original_data[condition]
            # # 对股票分组
            # grouped = recent_data.groupby('code')
            # 分组汇总, 若机构id,作者,预测年份都一样,则删除重复的项,然后再汇总一共有多少预测ni的报告(ni值非nan)
            # curr_coverage = grouped.apply(lambda x: x.drop_duplicates(subset=['organ_id', 'author',
            #                               'Time_year'])['forecast_profit'].count())

            # curr_coverage = grouped.apply(analyst_coverage.simple_count)
            # curr_coverage = grouped.apply(analyst_coverage.ni_count)
            # coverage_std = grouped.apply(analyst_coverage.ni_std)
            # coverage_mean = grouped.apply(analyst_coverage.ni_mean)
            # df = pd.DataFrame({'cov':curr_coverage, 'std':coverage_std, 'mean':coverage_mean})

            start_time = fra_pub_date.groupby('SecuCode')['InfoPublDate'].apply(
                analyst_coverage.get_start_time, end_time=end_time)
            curr_coverage1 = pd.Series(np.nan, index=start_time.index)
            curr_coverage2 = pd.Series(np.nan, index=start_time.index)
            # 循环股票
            for stock_id, curr_start_time in start_time.iteritems():
                time_condition1 = np.logical_and(original_data1['create_date'] >= curr_start_time,
                                                original_data1['create_date'] <= end_time)
                stock_condition1 = (original_data1['code'] == stock_id)
                curr_data1 = original_data1[np.logical_and(time_condition1, stock_condition1)]
                curr_coverage1.ix[stock_id] = analyst_coverage.simple_count(curr_data1)

                time_condition2 = np.logical_and(original_data2['create_date'] >= curr_start_time,
                                                original_data2['create_date'] <= end_time)
                stock_condition2 = (original_data2['code'] == stock_id)
                curr_data2 = original_data2[np.logical_and(time_condition2, stock_condition2)]
                curr_coverage2.ix[stock_id] = analyst_coverage.simple_count(curr_data2)


            df = pd.DataFrame({'cov':curr_coverage1, 'cov2':curr_coverage2})
            print(end_time)
            return df
        # 进行并行计算
        if __name__ == '__main__':
            ncpus = 16
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(holding_days.shape[0])
            chunksize=int(len(data_size)/ncpus)
            results = p.map(one_time_coverage, data_size, chunksize=chunksize)
            temp_coverage = pd.concat([i['cov'] for i in results], axis=1).T
            temp_coverage2 = pd.concat([i['cov2'] for i in results], axis=1).T
            # temp_disp = pd.concat([i['std'] for i in results], axis=1).T
            # temp_ep = pd.concat([i['mean'] for i in results], axis=1).T
            temp_coverage = temp_coverage.set_index(self.holding_days).reindex(index=coverage.index, method='ffill')
            coverage[:] = temp_coverage.reindex(columns=coverage.columns).values
            temp_coverage2 = temp_coverage2.set_index(self.holding_days).reindex(index=coverage2.index, method='ffill')
            coverage2[:] = temp_coverage2.reindex(columns=coverage2.columns).values
            # coverage_disp[:] = temp_disp.values
            # coverage_ep[:] = temp_ep.values
        pass

        # # 策略数据需要shift, 先shift再重索引为交易日,这样星期一早上能知道上周末的信息,而不是上周5的信息
        # coverage = coverage.shift(1)
        # 将其储存到raw_data中,对交易日和股票代码的重索引均在这里完成
        self.strategy_data.raw_data = pd.Panel({'coverage': coverage, 'coverage2':coverage2}, major_axis=
        self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

        # # 读取市值数据
        # mv = data.read_data(['FreeMarketValue'], shift=False)
        # mv = mv['FreeMarketValue']
        # # 重索引
        # coverage_disp = coverage_disp.reindex(index=mv.index, columns=mv.columns)
        # coverage_ep = coverage_ep.reindex(index=mv.index, columns=mv.columns)
        # coverage_disp = coverage_disp / mv
        # coverage_ep = coverage_ep / mv

        # # 策略数据需要shift
        # coverage_disp = coverage_disp.shift(1)
        # coverage_ep = coverage_ep.shift(1)
        # 储存数据
        # self.strategy_data.raw_data['coverage_disp'] = coverage_disp
        # self.strategy_data.raw_data['coverage_ep'] = coverage_ep

        # data.write_data(self.strategy_data.raw_data, file_name=['unique_coverage', 'coverage_disp', 'coverage_ep'])
        data.write_data(self.strategy_data.raw_data,
                        file_name=['unique_coverage garbage', 'unique_coverage gold'])

    # 计算因子值
    def get_abn_coverage(self):
        if os.path.isfile('unique_coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['unique_coverage'], item_name=['coverage'],
                                                         shift=True)
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

        # 计算ln(1+coverage)得到回归的y项
        self.strategy_data.raw_data['ln_coverage'] = np.log(self.strategy_data.raw_data.ix['coverage'] + 1)
        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume']/data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj']/data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        # 生成调仓日
        self.generate_holding_days(holding_freq='m', start_date='2007-01-01')

        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                        major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])
        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['ln_coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3:
                continue
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            abn_coverage.ix[time] = results.resid
            self.reg_stats.ix['coef', time, :] = results.params.values
            self.reg_stats.ix['t_stats', time, :] = results.tvalues.values
            self.reg_stats.ix['rsquare', time, 0] = results.rsquared
            self.reg_stats.ix['rsquare', time, 1] = results.rsquared_adj

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        self.strategy_data.factor = pd.Panel({'abn_coverage':abn_coverage}, major_axis=
                    self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)
        self.strategy_data.stock_price.ix['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                           percentile=0, compress=False)

    # 利用泊松回归来计算abn coverage
    def get_abn_coverage_poisson(self):
        if os.path.isfile('unique_coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['unique_coverage'], ['coverage'], shift=True)
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume'] / data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj'] / data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        # 生成调仓日
        self.generate_holding_days(holding_freq='m', start_date='2007-01-01')
        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                                  major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])

        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3 or not (y>0).any():
                continue
            P = Poisson(y, x, missing='drop')
            results = P.fit(full_output=True)
            abn_coverage.ix[time] = results.resid

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        self.strategy_data.factor = pd.Panel({'abn_coverage': abn_coverage}, major_axis=
        self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)
        self.strategy_data.factor.ix['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                                       percentile=0, compress=False)


    # 定义进行fama-macbeth回归的函数, 因为论文中用到了大量的fm回归
    @staticmethod
    def fama_macbeth(y, x, *, nw_lags=0, intercept=True):
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
        r2_adj = results_fm._ols_result.r2_adj.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()

        return results_fm.mean_beta, results_fm.t_stat, r2, r2_adj

    def get_table1a(self):
        if os.path.isfile('unique_coverage.csv'):
            # self.strategy_data.raw_data = data.read_data(['unique_coverage weighted'], ['coverage'], shift=True)
            # 将深度报告赋予更多的权重
            real_report = data.read_data(['unique_coverage gold'], shift=True)
            all_report = data.read_data(['unique_coverage garbage'], shift=True)
            real_report = real_report.fillna(0)
            all_report = all_report.fillna(0)
            self.strategy_data.raw_data = pd.Panel({'coverage': real_report['unique_coverage gold']*10 +
                                                                all_report['unique_coverage garbage']*1})
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

        # 计算ln(1+coverage)得到回归的y项
        self.strategy_data.raw_data['ln_coverage'] = np.log(self.strategy_data.raw_data.ix['coverage'] + 1)
        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume'] / data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj'] / data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                                  major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])
        from statsmodels.discrete.discrete_model import Poisson
        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['ln_coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3:
                continue
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            # P = Poisson(y, x, missing='drop')
            # results = P.fit(full_output=True)
            abn_coverage.ix[time] = results.resid
            self.reg_stats.ix['coef', time, :] = results.params.values
            self.reg_stats.ix['t_stats', time, :] = results.tvalues.values
            # self.reg_stats.ix['rsquare', time, 0] = results.rsquared
            # self.reg_stats.ix['rsquare', time, 1] = results.rsquared_adj

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        # 再次对abn coverage计算暴露, 但是不再winsorize
        self.strategy_data.stock_price['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                                    percentile=0, compress=False)

        # 应当根据月份,对数据进行fm回归
        y_fm = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :]
        x_fm = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], self.holding_days, :]
        # 进行fm回归
        coef, t_stat, r2, r2_adj = analyst_coverage.fama_macbeth(y_fm, x_fm)
        self.table1a = pd.DataFrame({'coef':coef, 't_stat':t_stat})
        self.table1a['r_square'] = np.nan
        self.table1a.ix[0, 'r_square'] = r2
        self.table1a.ix[1, 'r_square'] = r2_adj

        # 用csv储存结果
        # self.table1a.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #              '/' + 'Table1a.csv', na_rep='N/A', encoding='GB18030')
        self.reg_stats.mean(1).to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                      '/' + 'Table1a.csv', na_rep='N/A', encoding='GB18030')

        # # 对比用clustered se算出的t stats
        # stacked_y = self.strategy_data.raw_data.ix['ln_coverage'].stack(dropna=False)
        # stacked_x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum']].to_frame(filter_observations=False)
        # stacked_x = sm.add_constant(stacked_x)
        # valid = pd.concat([stacked_y, stacked_x], axis=1).notnull().all(1)
        # stacked_y = stacked_y[valid]
        # stacked_x = stacked_x[valid]
        # groups_stock = stacked_y.index.get_level_values(1).values
        # model = sm.OLS(stacked_y, stacked_x)
        # # results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups':groups_stock})
        # results_cluster = model.fit()
        # self.table1a_cluster = self.table1a * np.nan
        # self.table1a_cluster['coef'] = results_cluster.params.values
        # self.table1a_cluster['t_stats'] = results_cluster.tvalues.values
        # self.table1a_cluster.ix[0, 'rsquare'] = results_cluster.rsquared
        # self.table1a_cluster.ix[1, 'rsquare'] = results_cluster.rsquared_adj
        #
        # # 储存结果
        # self.table1a_cluster.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #                     '/' + 'Table1a_cluster.csv', na_rep='N/A', encoding='GB18030')

        # 首先计算各种要用到的数据
        self.strategy_data.stock_price['vlty'] = self.strategy_data.stock_price.ix['daily_return'].rolling(252).std()
        # relative spread 定义不明，暂时不做
        bp = data.read_data(['bp'], shift=True)
        bp = bp.ix['bp']
        self.strategy_data.stock_price['lbm'] = np.log(1 + bp)
        roa_data = data.read_data(['TotalAssets', 'NetIncome_ttm'], shift=True)
        self.strategy_data.stock_price['roa'] = roa_data['NetIncome_ttm'] / roa_data['TotalAssets']

        # 读取净利润
        ni_ttm = data.read_data(['NetIncome_ttm'])
        self.strategy_data.raw_data['ni_ttm'] = ni_ttm.ix['NetIncome_ttm']

        # 过滤数据
        self.strategy_data.discard_uninv_data()

        # 计算因子暴露
        for item in ['vlty', 'lbm', 'roa']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        base = pd.concat([self.strategy_data.raw_data.ix['coverage'], self.strategy_data.stock_price.ix['abn_coverage'],
                          self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum', 'vlty',
                                                             'lbm', 'roa']]], axis=0)
        self.base = pd.Panel(base.values, items=['coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                                 'vlty', 'lbm', 'roa'], major_axis=base.major_axis,
                             minor_axis=base.minor_axis)

    def get_table1b(self):

        stats = pd.Panel(np.nan, items=['obs', 'coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                        'vlty', 'lbm', 'roa'], major_axis=self.holding_days, minor_axis=np.arange(10))
        # 循环调仓日，建立分位数统计量
        for cursor, time in enumerate(self.holding_days):
            curr_data = self.base.ix[:, time, :]
            # 如果abn coverage数据全是0，则继续循环
            if curr_data['abn_coverage'].isnull().all():
                continue
            #
            group_label = pd.qcut(curr_data['abn_coverage'], 10, labels=False)
            stats.ix['obs', time, :] = curr_data['coverage'].groupby(group_label).size()
            stats.ix['coverage', time, :] = curr_data['coverage'].groupby(group_label).apply(lambda x:x.sum()/x.size)
            stats.ix[2:, time, :] = curr_data.iloc[:, 1:].groupby(group_label).mean().T.values

            if stats.ix['roa', time, 2] >=40:
                pass

        # 循环结束后,对时间序列上的值取均值
        # self.table1b = stats.mean(axis=1)
        self.table1b = stats.median(axis=1)
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table1b, loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table1b.png', dpi=1200)
        self.table1b.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Table1b.csv', na_rep='N/A', encoding='GB18030')

    # 不断加入因子回归, 看rsquare adj的路径长什么样
    def get_fig2(self):
        # 从fig2开始, coverage数据全部用的是lncoverage, 因此将coverage改为lncoverage
        self.base.ix['coverage'] = np.log(self.base.ix['coverage'] + 1)
        # 不要abn coverage因子
        y_data = self.base.ix['coverage']
        x_data = self.base.iloc[2:]
        # 储存累计r square以及最终t_stats
        self.figure2 = pd.DataFrame(np.nan, index=['r_square_adj', 't_stats'], columns=x_data.items)
        # 有多少x维度
        dim_x = x_data.shape[0]
        # 循环递增自变量
        k=1
        while k <= dim_x:
            # 储存回归结果
            reg_results = pd.DataFrame(np.nan, index=self.holding_days,
                                       columns=['r_square']+[i for i in x_data.items])
            # # 循环调仓日进行回归
            # for cursor, time in enumerate(self.holding_days):
            #     y = y_data.ix[time, :]
            #     x = x_data.ix[0:k, time, :]
            #     x = sm.add_constant(x)
            #     # 如果只有小于等于1个有效数据，则返回nan序列
            #     if pd.concat([y, x], axis=1).dropna().shape[0] <= k:
            #         continue
            #     model = sm.OLS(y, x, missing='drop')
            #     results = model.fit()
            #     reg_results.ix[time, 0] = results.rsquared_adj
            #     reg_results.ix[time, 1:(k+1)] = results.tvalues[1:].values
            # # 循环结束, 储存这轮回归的平均rsquared adj
            # self.figure2.ix['r_square', k-1] = reg_results.ix[:, 0].mean()

            # 在每次循环中使用fm回归
            y = y_data.ix[self.holding_days, :]
            x = x_data.ix[0:k, self.holding_days, :]
            coef, t_stat, r2, r2_adj = analyst_coverage.fama_macbeth(y, x)
            self.figure2.ix['r_square_adj', k-1] = r2_adj
            k += 1
        # 当结束最后一次循环的时候, 储存各回归系数的t stats
        # self.figure2.ix['t_stats', :] = reg_results.ix[:, 1:].replace(np.inf, np.nan).replace(-np.inf, np.nan).mean().values
        self.figure2.ix['t_stats', :] = t_stat
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.figure2, loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Figure2.png', dpi=1200)
        self.figure2.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Figure2.csv', na_rep='N/A', encoding='GB18030')
        pass

    # 做标准化后, 看abn cov对之后的财务数据的预测能力, 这里选取净利润, 而且是在控制其他因子的情况下
    def get_table3(self):

        # 取用来预测的解释变量
        raw_x_data = self.base.ix[['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm']]
        # 被解释变量为净利润 或净利润增长率
        raw_ya_data = self.strategy_data.raw_data.ix['ni_ttm']
        # raw_yb_data = np.log(self.strategy_data.raw_data.ix['ni_ttm']/self.strategy_data.raw_data.ix['ni_ttm'].shift(63))
        raw_yb_data = strategy_data.get_ni_growth(self.strategy_data.raw_data.ix['ni_ttm'], lag=63)
        # 标准化
        ya_data = strategy_data.get_exposure(raw_ya_data)
        yb_data = strategy_data.get_exposure(raw_yb_data)
        x_data = raw_x_data * np.nan
        for cursor, item in enumerate(raw_x_data.items):
            x_data[item] = strategy_data.get_exposure(raw_x_data.ix[item])
        # 储存数据
        self.table3a = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in raw_x_data.items]+\
                                ['r_square'], minor_axis=['q+1', 'q+2', 'q+3', 'q+4'])
        self.table3b = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in raw_x_data.items]+\
                                ['r_square'], minor_axis=['q+1', 'q+2', 'q+3', 'q+4'])
        # 循环季度长度,从下一个季度的预测,预测之后第4个季度
        for next_q in np.arange(4):
            # 解释变量会被移动
            curr_ya_data = ya_data.shift(-63 * (next_q + 1))
            curr_yb_data = yb_data.shift(-63 * (next_q + 1))
            # # 接下来建立数据储存, 以及循环回归
            # reg_results_a = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                          minor_axis=['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm', 'r_square'])
            # reg_results_b = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                          minor_axis=['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm', 'r_square'])
            # for cursor, time in enumerate(self.holding_days):
            #     ya = curr_ya_data.ix[time, :]
            #     yb = curr_yb_data.ix[time, :]
            #     x = x_data.ix[:, time, :]
            #     x = sm.add_constant(x)
            #     # 对于2个回归,只有在有一个以上有效数据的情况下才回归
            #     if pd.concat([ya, x], axis=1).dropna().shape[0] > 5:
            #         modela = sm.OLS(ya, x, missing='drop')
            #         resultsa = modela.fit()
            #         reg_results_a.ix['coef', time, 0:5] = resultsa.params[1:].values
            #         reg_results_a.ix['t_stats', time, 0:5] = resultsa.tvalues[1:].values
            #         reg_results_a.ix['coef', time, 5] = resultsa.rsquared
            #         reg_results_a.ix['t_stats', time, 5] = resultsa.rsquared_adj
            #     if pd.concat([yb, x], axis=1).dropna().shape[0] > 5:
            #         modelb = sm.OLS(yb, x, missing='drop')
            #         resultsb = modelb.fit()
            #         reg_results_b.ix['coef', time, 0:5] = resultsb.params[1:].values
            #         reg_results_b.ix['t_stats', time, 0:5] = resultsb.tvalues[1:].values
            #         reg_results_b.ix['coef', time, 5] = resultsb.rsquared
            #         reg_results_b.ix['t_stats', time, 5] = resultsb.rsquared_adj

            # 进行fm回归
            ya = curr_ya_data.ix[self.holding_days, :]
            yb = curr_yb_data.ix[self.holding_days, :]
            x = x_data.ix[:, self.holding_days, :]
            coef_a, t_a, r2_a, r2_adj_a = analyst_coverage.fama_macbeth(ya, x)
            coef_b, t_b, r2_b, r2_adj_b = analyst_coverage.fama_macbeth(yb, x)
            self.table3a.ix['coef', :, next_q] = coef_a
            self.table3a.ix['t_stats', :, next_q] = t_a
            self.table3a.ix['coef', -1, next_q] = r2_a
            self.table3a.ix['t_stats', -1, next_q] = r2_adj_a

            self.table3b.ix['coef', :, next_q] = coef_b
            self.table3b.ix['t_stats', :, next_q] = t_b
            self.table3b.ix['coef', -1, next_q] = r2_b
            self.table3b.ix['t_stats', -1, next_q] = r2_adj_b

            # # 循环结束,计算回归结果平均数并储存
            # self.table3a.ix['coef', :, next_q] = reg_results_a.ix['coef', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3a.ix['t_stats', :, next_q] = reg_results_a.ix['t_stats', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3b.ix['coef', :, next_q] = reg_results_b.ix['coef', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3b.ix['t_stats', :, next_q] = reg_results_b.ix['t_stats', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
        pass
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3a.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3a_coef.png', dpi=1200)
        self.table3a.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Table3a_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3a.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3a_t_stats.png', dpi=1200)
        self.table3a.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                       '/' + 'Table3a_t_stats.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3b.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3b_coef.png', dpi=1200)
        self.table3b.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                       '/' + 'Table3b_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3b.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3b_t_stats.png', dpi=1200)
        self.table3b.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table3b_t_stats.csv', na_rep='N/A', encoding='GB18030')

    # 用一众因子对收益进行逐步回归,其实就是算各因子的回归纯因子收益
    def get_table6(self):
        # 首先算月收益,将收益按调仓月分配到各个组中取聚合求和
        term_label = pd.cut(self.strategy_data.stock_price.major_axis.map(lambda x:x.value),
                            bins=self.holding_days.map(lambda x:x.value), labels=False)
        term_label[np.isnan(term_label)] = -1.0
        self.monthly_return = self.strategy_data.stock_price.ix['daily_return'].groupby(term_label).sum()
        # 因为是用之后的收益来回归,因此相当于预测了收益
        monthly_return = self.monthly_return.shift(-1)
        ## 选取解释变量,将base中丢掉abn coverage即可
        # 用其他回归后, 用coverage不如用abn coverage, 因此丢弃coverage
        raw_x_data = self.base.drop('coverage')
        # 计算暴露
        y_data = pd.DataFrame(monthly_return.values, index=self.holding_days, columns=monthly_return.columns)
        x_data = raw_x_data * np.nan
        x_dim = x_data.shape[0]
        for cursor, item in enumerate(raw_x_data.items):
            x_data[item] = strategy_data.get_exposure(raw_x_data.ix[item])
        # 初始化储存数据的矩阵
        self.table6 = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in x_data.items] + \
                               ['intercept', 'r_square'], minor_axis=np.arange(x_dim))
        # 开始进行循环,每一步多添加一个因子
        k = 1
        while k <= x_dim:
            # # 建立储存数据的panel
            # reg_results = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                        minor_axis=self.table6.major_axis)
            # # 循环进行回归
            # for cursor, time in enumerate(self.holding_days):
            #     y = y_data.iloc[cursor, :]
            #     x = x_data.ix[:k, time, :]
            #     x = sm.add_constant(x)
            #     # 对于2个回归,只有在有一个以上有效数据的情况下才回归
            #     if pd.concat([y, x], axis=1).dropna().shape[0] <= k:
            #         continue
            #     model = sm.OLS(y, x, missing='drop')
            #     results = model.fit()
            #     # 储存结果
            #     reg_results.ix['coef', time, :k] = results.params[1:k+1]
            #     reg_results.ix['coef', time, -2] = results.params[0]
            #     reg_results.ix['coef', time, -1] = results.rsquared
            #     reg_results.ix['t_stats', time, :k] = results.tvalues[1:k+1]
            #     reg_results.ix['t_stats', time, -2] = results.tvalues[0]
            #     reg_results.ix['t_stats', time, -1] = results.rsquared_adj
            # # 循环结束,求平均值储存
            # self.table6.ix[:, :, k-1] = reg_results.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean(axis=1)

            # 进行fm回归
            y = y_data
            x = x_data.ix[:k, self.holding_days, :]
            coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x, nw_lags=10)
            self.table6.ix['coef', :k, k-1] = coef.values[:k]
            self.table6.ix['coef', 'intercept', k-1] = coef.values[-1]
            self.table6.ix['t_stats', :k, k-1] = t_stats.values[:k]
            self.table6.ix['t_stats', 'intercept', k-1] = t_stats.values[-1]
            self.table6.ix['coef', 'r_square', k-1] = r2
            self.table6.ix['t_stats', 'r_square', k-1] = r2_adj

            k += 1
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table6.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table6_coef.png', dpi=1200)
        self.table6.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table6_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table6.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table6_t_stats.png', dpi=1200)
        self.table6.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table6_t_stats.csv', na_rep='N/A', encoding='GB18030')
        pass

    # 画论文中的table7, 研究abn coverage因子与分析师预测值本身的std与mean的关系
    def get_table7(self):
        # 读取disp和ep的数据
        disp_ep = data.read_data(['coverage_disp', 'coverage_ep'], shift=True)
        disp = disp_ep['coverage_disp']
        ep = disp_ep['coverage_ep']
        # abn coverage数据
        abn_coverage = self.base['abn_coverage']

        def pct_rank_qcut(x, *, n):
            if x.dropna().size <= 3:
                return pd.Series(np.nan, index=x.index)
            q_labels = pd.qcut(x, q=n, labels=['low', 'mid', 'high'])
            return q_labels

        # 将disp, ep, abn coverage分为3个分位点
        disp_tercile = disp.apply(pct_rank_qcut, axis=1, n=3)
        ep_tercile = ep.apply(pct_rank_qcut, axis=1, n=3)
        abn_coverage_tercile = abn_coverage.apply(pct_rank_qcut, axis=1, n=3)
        # 循环取得dummy变量
        disp_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['disp_low', 'disp_mid', 'disp_high'])
        ep_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['ep_low', 'ep_mid', 'ep_high'])
        abn_coverage_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['abn_coverage_low',
                                        'abn_coverage_mid', 'abn_coverage_high'])
        for cursor, time in enumerate(self.holding_days):
            if disp_tercile.ix[time, :].dropna().size >= 3:
                disp_dummies[time] = pd.get_dummies(disp_tercile.ix[time, :], prefix='disp')
            if ep_tercile.ix[time, :].dropna().size >= 3:
                ep_dummies[time] = pd.get_dummies(ep_tercile.ix[time, :], prefix='ep')
            if abn_coverage_tercile.ix[time, :].dropna().size >= 3:
                abn_coverage_dummies[time] = pd.get_dummies(abn_coverage_tercile.ix[time, :], prefix='abn_coverage')

        # 循环结束后进行转置
        disp_dummies = disp_dummies.transpose(2, 0, 1)
        ep_dummies = ep_dummies.transpose(2, 0, 1)
        abn_coverage_dummies = abn_coverage_dummies.transpose(2, 0, 1)
        # 将所有的dummy变量链接成一个大的panel, 从中选取解释变量,并首先进行数据过滤
        dummy_base = pd.concat([abn_coverage_dummies, disp_dummies, ep_dummies], axis=0)
        for item, df in dummy_base.iteritems():
            dummy_base[item] = df.where(self.strategy_data.if_tradable['if_inv'], np.nan)

        # 储存回归结果的表
        self.table7 = pd.Panel(np.nan, items=['coef' 't_stats'], major_axis=['high ATOT', 'high ATOT & low signal',
                               'high ATOT & mid signal', 'mid ATOT', 'mid ATOT & low signal', 'mid ATOT & high signal',
                               'low ATOT', 'low ATOT & mid signal', 'low ATOT & high signal', 'ATOT', 'disp', 'ep',
                               'r_square'], minor_axis = np.arange(4))

        # 回归的被解释变量
        # 月度收益,但是注意是预测未来收益,因此shift了-1
        y = self.monthly_return.shift(-1)
        y = pd.DataFrame(y.values, index=self.holding_days, columns=y.columns)

        # 第一次回归的回归变量为atot的3个分位数
        x1 = pd.Panel({'high ATOT':dummy_base['abn_coverage_high'], 'mid ATOT':dummy_base['abn_coverage_mid'],
                       'low ATOT':dummy_base['abn_coverage_low']})
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x1, nw_lags=10, intercept=False)
        self.table7.ix['coef', ['high ATOT', 'mid ATOT', 'low ATOT'], 0] = coef
        self.table7.ix['t_stats', ['high ATOT', 'mid ATOT', 'low ATOT'], 0] = t_stats
        self.table7.ix['coef', 'r_square', 0] = r2
        self.table7.ix['t_stats', 'r_square', 0] = r2_adj

        # 计算交叉回归项的x
        def get_intersect_x(signal):
            hl = dummy_base['abn_coverage_high']*dummy_base[signal+'low']
            hm = dummy_base['abn_coverage_high']*dummy_base[signal+'mid']
            ml = dummy_base['abn_coverage_mid'] * dummy_base[signal + 'low']
            mh = dummy_base['abn_coverage_mid']*dummy_base[signal+'high']
            lm = dummy_base['abn_coverage_low']*dummy_base[signal+'mid']
            lh = dummy_base['abn_coverage_low']*dummy_base[signal+'high']\

            intersect_base = pd.Panel({'high ATOT':dummy_base['abn_coverage_high'], 'high ATOT & low signal':hl,
                                       'high ATOT & mid signal':hm, 'mid ATOT':dummy_base['abn_coverage_mid'],
                                       'mid ATOT & low signal':ml, 'mid ATOT & high signal':mh,
                                       'low ATOT':dummy_base['abn_coverage_low'], 'low ATOT & mid signal':lm,
                                       'low ATOT & high signal':lh})
            return intersect_base

        # 第二次回归为使用disp的交叉项
        x2 = get_intersect_x('disp_')
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x2, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'high ATOT':'low ATOT & high signal', 1] = coef
        self.table7.ix['t_stats', 'high ATOT':'low ATOT & high signal', 1] = t_stats
        self.table7.ix['coef', 'r_square', 1] = r2
        self.table7.ix['t_stats', 'r_square', 1] = r2_adj

        # 第三次回归为使用ep的交叉项
        x3 = get_intersect_x('ep_')
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x3, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'high ATOT':'low ATOT & high signal', 2] = coef
        self.table7.ix['t_stats', 'high ATOT':'low ATOT & high signal', 2] = t_stats
        self.table7.ix['coef', 'r_square', 2] = r2
        self.table7.ix['t_stats', 'r_square', 2] = r2_adj

        # 第三次回归使用abn coverage, disp的暴露, ep的暴露回归
        # 过滤数据
        disp = disp.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        ep = ep.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        # 计算暴露
        disp_expo = strategy_data.get_exposure(disp)
        ep_expo = strategy_data.get_exposure(ep)
        x4 = pd.Panel({'ATOT':abn_coverage, 'disp':disp_expo, 'ep':ep_expo})
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x4, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'ATOT':'ep', 3] = coef
        self.table7.ix['t_stats', 'ATOT':'ep', 3] = t_stats
        self.table7.ix['coef', 'r_square', 3] = r2
        self.table7.ix['t_stats', 'r_square', 3] = r2_adj
        pass

        # 储存数据
        self.table7.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                      '/' + 'Table7_coef.csv', na_rep='N/A', encoding='GB18030')
        self.table7.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                         '/' + 'Table7_t_stats.csv', na_rep='N/A', encoding='GB18030')

    # 根据原始的coverage值, 画面板数据的kde值
    def draw_kde(self):
        unique_coverage = self.strategy_data.raw_data.ix['coverage', self.holding_days, :]
        stacked_uc = unique_coverage.stack(dropna=True)

        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.hist(stacked_uc.values)
        plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) + '/kde.png', dpi=1200)

        uc_old = data.read_data(['unique_coverage cmb'], item_name=['coverage_old'])
        uc_old = uc_old['coverage_old'].fillna(0).where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        uc_old = uc_old.ix[self.holding_days, :]
        stacked_uc_old = uc_old.stack(dropna=True)

        f2 = plt.figure()
        ax2 = f2.add_subplot(1, 1, 1)
        plt.hist(stacked_uc_old.values)
        plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) + '/kde_old.png', dpi=1200)

        def pct_rank_qcut(x, *, n):
            if x.dropna().size <= 3:
                return pd.Series(np.nan, index=x.index)
            q_labels = pd.qcut(x, q=n, labels=False)
            return q_labels

        # 按照市值分组画图
        lncap = self.base.ix['lncap', self.holding_days, :]
        # 将市值分成3组
        lncap_labels = lncap.apply(pct_rank_qcut, axis=1, n=3)

        # 根据每组市值进行画图
        for i, mv in enumerate(['s', 'm', 'l']):
            # 根据市值标签所分的组
            curr_stacked_uc = unique_coverage.where(lncap_labels==i, np.nan).stack(dropna=True)
            curr_stacked_uc_old = uc_old.where(lncap_labels==i, np.nan).stack(dropna=True)

            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            plt.hist(curr_stacked_uc)
            plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) +
                        '/kde_' + mv + '.png', dpi=1200)

            f_old = plt.figure()
            ax_old = f_old.add_subplot(1, 1, 1)
            plt.hist(curr_stacked_uc_old)
            plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) +
                        '/kde_old_' + mv + '.png', dpi=1200)


    # 了解分析师报告中, 各项预测指标所占的比例
    def get_analyst_report_structure(self):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        sql_query = "select a.id, create_date, code, organ_id, author, Time_year, forecast_profit, " \
                    "forecast_income as revenue, forecast_income_share as eps, forecast_return, " \
                    "forecast_return_cash_share, forecast_return_capital_share from " \
                    "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select * from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.report_search_id) " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        # pivot_data = original_data.pivot_table(index='create_date', columns='code', values=['forecast_profit',
        #                                         'revenue', 'eps', 'forecast_return', 'forecast_return_cash_share',
        #                                         'forecast_return_capital_share'])

        # 去除重复项
        unique_data = original_data.drop_duplicates(subset=['code', 'organ_id', 'author', 'Time_year'],
                                                    keep='last')
        # 各项指标的预测数量
        chara_fy_count = unique_data[['forecast_profit', 'revenue', 'eps', 'forecast_return',
                                'forecast_return_cash_share', 'forecast_return_capital_share']].count()
        report_fy_count = unique_data.shape[0]
        report_structure_unique = chara_fy_count / report_fy_count

        # 不去除重复项的版本
        report_structure = original_data.count()/original_data.shape[0]

        # 预测未来3个财年的报告, 占报告的比例
        # 这里的唯一报告是3个财年只算一个报告
        unique_report = original_data.drop_duplicates(subset=['code', 'organ_id', 'author'],
                                                      keep='last').shape[0]
        # 根据报告进行分组, 看每个报告中, 有预测3个财年的比例

        pass

    # 根据每个公司财报发布的时间, 以及他们的分析师报告的时间, 来了解分析师报告的发布时间的情况
    def time_about_analyst_report(self):
        # 首先还是要取原始的分析师报告数据
        # 先将时间期内的所有数据都取出来
        # sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
        #             "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
        #             "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
        #             str(self.db.trading_days.iloc[-1]) + "') a " \
        #             "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
        #             "on a.id=b.id) " \
        #             "order by create_date, code "

        # sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
        #             "((select id, code, organ_id, type_id, author, create_date from CMB_REPORT_RESEARCH where " \
        #             "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
        #             str(self.db.trading_days.iloc[-1]) + "') a " \
        #             "left join (select report_search_id as id, Time_year, forecast_profit from CMB_REPORT_SUBTABLE) b " \
        #             "on a.id=b.id) where type_id != 28" \
        #             "order by create_date, code "

        # sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
        #             "((select id, code, organ_id, type_id, author, create_date from CMB_REPORT_RESEARCH where " \
        #             "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
        #             str(self.db.trading_days.iloc[-1]) + "') a " \
        #             "left join (select report_search_id as id, Time_year, forecast_profit from CMB_REPORT_SUBTABLE) b " \
        #             "on a.id=b.id) where type_id in (21, 22, 25) " \
        #             "order by create_date, code "

        # 需要研究的报告类型
        type_id = "98"
        folder_name = 'analyst_coverage_new/analyst_report_time/id' + type_id
        sql_query = "select stock_code, report_type, organ_id, author_name, create_date, report_year, " \
                    "forecast_eps, entrytime from rpt_forecast_stk where create_date>='" + \
                    str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "' and report_type in (" + str(type_id) + \
                    ") " + "order by create_date, stock_code"

        original_data = self.db.gg_new_engine.get_original_data(sql_query)

        # 去除重复的报告数据, 一份报告只能算一次
        original_data['create_date'] = pd.to_datetime(original_data['create_date'])
        unique_data = original_data.drop_duplicates(subset=['create_date', 'stock_code',
                                                            'organ_id', 'author_name'], keep='first')
        month_distribution = unique_data['create_date'].map(lambda x: x.month)

        if not os.path.exists(str(os.path.abspath('.')) + '/' + str(folder_name) + '/'):
            os.makedirs(str(os.path.abspath('.')) + '/' + str(folder_name) + '/')
        # 画分析师报告的发布时间的月份分布, 可以看到大部分的分析师报告都是在财报发布的月份发布的
        f1 = plt.figure()
        ax1 = f1.add_subplot(1, 1, 1)
        plt.hist(month_distribution.values, np.arange(1, 14))
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + str(folder_name) + '/month_dist.png', dpi=1200)

        # 尝试研究每份分析师报告是在公司财报发布后或发布前的什么时候发布的
        # 储存数据
        time_diff_dist = pd.Series(np.nan, index=np.arange(unique_data.shape[0]))

        fra_pub_date = self.fra_pub_date * 1
        # 计算每期的函数
        def one_time_diff(cursor):
            curr_pub_date = unique_data.iloc[cursor, 4]
            curr_stock = unique_data.iloc[cursor, 0]
            curr_stock_fra = fra_pub_date[fra_pub_date['SecuCode'] == curr_stock]
            if curr_stock_fra.empty:
                return np.nan
            # 计算财报发布日期和分析师报告发布日期的差值
            date_diff_series = curr_stock_fra['InfoPublDate'] - curr_pub_date
            # 距离分析师发布最近的那一期财报
            latest_fra_index = date_diff_series.abs().idxmin()
            # 获取最近的一期财报和这期分析师报告的发布时间差距
            recent_diff = date_diff_series[latest_fra_index].days
            return recent_diff

        # # 循环唯一的报告
        # for cursor, curr_report in unique_data.iterrows():
        #     # 当前报告的发布时间和针对的股票
        #     curr_pub_date = curr_report['create_date']
        #     curr_stock = curr_report['code']
        #     # 在财报发布日期中, 筛出当前股票的那一个
        #     curr_stock_fra = self.fra_pub_date[self.fra_pub_date['SecuCode'] == curr_stock]
        #     if curr_stock_fra.empty:
        #         continue
        #     # 计算财报发布日期和分析师报告发布日期的差值
        #     date_diff_series = curr_stock_fra['InfoPublDate'] - curr_pub_date
        #     # 距离分析师发布最近的那一期财报
        #     latest_fra_index = date_diff_series.abs().idxmin()
        #     # 获取最近的一期财报和这期分析师报告的发布时间差距
        #     recent_diff = date_diff_series[latest_fra_index].days
        #     # 储存数据
        #     time_diff_dist.ix[cursor] = recent_diff
        #     print(cursor)
        #     pass

        import pathos.multiprocessing as mp
        if __name__ == '__main__':
            ncpus = 20
            p = mp._ProcessPool(ncpus)
            data_size = np.arange(unique_data.shape[0])
            # data_size = np.arange(5000)
            chunksize = int(len(data_size)/ncpus)
            results = p.map(one_time_diff, data_size, chunksize=chunksize)
            time_diff_dist[:] = results
            pass

        time_diff_dist = time_diff_dist.dropna()

        # 画直方图
        f2 = plt.figure()
        ax2 = f2.add_subplot(1, 1, 1)
        plt.hist(time_diff_dist.values, [-300, -31, -15, -7, 0, 7, 15, 31, 300])
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + str(folder_name) + '/time_diff_dist.png', dpi=1200)

        # 画饼图
        labels = ['>1m before', '1m before', '0.5m before', '1w before', '1w after', '0.5m after',
                  '1m after', '>1m after']
        time_diff_pie = pd.cut(time_diff_dist, [-300, -31, -15, -7, -0.01, 7, 15, 31, 300], labels=labels)
        time_diff_pie_ratio = time_diff_dist.groupby(time_diff_pie).count()/time_diff_dist.shape[0]
        f3 = plt.figure()
        ax3 = f3.add_subplot(1, 1, 1)
        plt.pie(time_diff_pie_ratio, labels=labels, autopct='%1.1f%%')
        plt.savefig(str(os.path.abspath('.')) + '/' + str(folder_name) + '/time_diff_pie.png',
                    dpi=1200)


        pass

    # 构建因子
    def construct_factor(self):
        # self.get_abn_coverage_poisson()

        # self.strategy_data.factor = data.read_data(['unique_coverage der'], ['unique_coverage'], shift=True)

        # 将深度报告赋予更多的权重
        real_report = data.read_data(['unique_coverage gold'], shift=True)
        all_report = data.read_data(['unique_coverage garbage'], shift=True)
        real_report = real_report.fillna(0)
        all_report = all_report.fillna(0)
        self.strategy_data.factor = pd.Panel({'unique_coverage' : real_report['unique_coverage gold']*10 +
                                             all_report['unique_coverage garbage']*1})
        # data.write_data(self.strategy_data.factor, file_name=['unique_coverage weighted'])

        # self.strategy_data.factor = data.read_data(['growth'], shift=True)

        self.strategy_data.factor = self.strategy_data.factor.reindex(major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)
        self.strategy_data.factor.ix['unique_coverage'] = self.strategy_data.factor.ix['unique_coverage'].fillna(0)

        # # 考虑计算分析师数量的增长率
        # temp_growth = self.strategy_data.factor.ix['unique_coverage']. \
        #     div(self.strategy_data.factor.ix['unique_coverage'].shift(252) ) - 1
        # # 一年前分析师报告数量是0, 而现在拥有分析师报告的, 默认其增长了100%
        # condition_1 = np.logical_and(self.strategy_data.factor.ix['unique_coverage'] != 0,
        #     self.strategy_data.factor.ix['unique_coverage'].shift(252)==0)
        # temp_growth[condition_1] = 1
        # # 一年前和现在都是0的, 则认为是0
        # condition_2 = np.logical_and(self.strategy_data.factor.ix['unique_coverage'] == 0,
        #     self.strategy_data.factor.ix['unique_coverage'].shift(252) == 0)
        # temp_growth[condition_2] = 0
        # self.strategy_data.factor.ix['unique_coverage'] = temp_growth * 1


        self.strategy_data.factor.ix['unique_coverage'] = np.log(self.strategy_data.factor.ix['unique_coverage'] + 1)
        pass

    # # 做对barra base的泊松回归, 用这些因子进行提纯
    # def get_pure_factor(self, bb_obj, *, do_active_bb_pure_factor=False, reg_weight=1,
    #                     add_constant=False, use_factor_expo=True, expo_weight=1):
    #     bb_obj.just_get_factor_expo()
    #     lag_bb_expo = bb_obj.bb_data.factor_expo.shift(1).reindex(major_axis=bb_obj.bb_data.factor_expo.major_axis)
    #     lag_bb_expo_nocf = lag_bb_expo.drop('country_factor', axis=0)
    #
    #     lag_bb_expo_nocf = lag_bb_expo_nocf.iloc[0:10, :, :]
    #
    #     abn_coverage = self.strategy_data.factor.iloc[0] * np.nan
    #     t_stats = pd.DataFrame(np.nan, index=self.holding_days, columns=lag_bb_expo_nocf.items)
    #     p_values = t_stats * np.nan
    #     zero_y = t_stats * np.nan
    #     # 进行poisson回归
    #     for cursor, time in enumerate(self.holding_days):
    #         y = self.strategy_data.factor.ix[0, time, :]
    #         x = lag_bb_expo_nocf.ix[:, time, :]
    #         # 如果只有小于等于1个有效数据，则返回nan序列
    #         if pd.concat([y, x], axis=1).dropna().shape[0] <= 30 or not (y>0).any():
    #             continue
    #         P = Poisson(y, x, missing='drop')
    #         results = P.fit()
    #         abn_coverage.ix[time] = results.resid
    #         t_stats.ix[time] = results.tvalues
    #         p_values.ix[time] = results.pvalues
    #         zero_y.ix[time] = (y==0).sum()/(y.dropna().shape[0])
    #
    #     abn_coverage_expo = strategy_data.get_cap_wgt_exposure(abn_coverage, self.strategy_data.stock_price.
    #                                                            ix['FreeMarketValue'])
    #     # abn_coverage_expo = strategy_data.get_exposure(abn_coverage)
    #     self.strategy_data.factor.iloc[0] = abn_coverage_expo


    # 对原始数据的描述，即对论文的图表进行复制
    def data_description(self):
        self.get_table1a()
        self.get_table1b()
        self.get_fig2()
        self.get_table3()
        self.get_table6()
        # self.get_table7()
        self.draw_kde()
        pass



# 重新审视analyst coverage这个因子, 2018年2月26日
class analyst_coverage_new(analyst_coverage):
    def __init__(self, start_date=pd.Timestamp('2007-01-01'), end_date=pd.Timestamp('2018-01-17')):
        analyst_coverage.__init__(self, start_date=start_date, end_date=end_date)

    # 取新财富分析师数据, 指示其是否为新财富分析师
    def new_fortune_author(self):
        sql_query = "select author_id, author_name, time_year, pia, entrydate from T_GREAT_AUTHOR "
        self.great_author = self.db.gg_engine.get_original_data(sql_query)
        pass

    # 取分析师信息数据
    def get_author_info(self):
        # 去掉那些入职时间在1990年前的明显错误数据
        sql_query = "select author_id, organ_id, is_incumbent, y1, m1, y2, m2, entrytime " \
                    "from rpt_author_information where y1 >= 1990 "
        self.author_info = self.db.gg_new_engine.get_original_data(sql_query)
        # 取每个分析师的最早入职年限
        def get_career_start(df):
            new_df = df.sort_values('y1')
            career_start = pd.Timestamp(year=new_df.ix[new_df.index[0], 'y1'],
                                        month=new_df.ix[new_df.index[0], 'm1'], day=1)
            df['career'] = career_start
            return df
        temp = self.author_info.groupby('author_id').apply(get_career_start)
        self.author_info['career'] = temp['career']
        pass

    # 取分析师报告数据
    def get_analyst_report_count(self):
        # 只取21, 23, 24, 26这4种报告
        sql_query = "select * from ( " \
                    "(select report_id, stock_code, report_type, organ_id, author_name, create_date, " \
                    "report_year, forecast_eps, entrytime from rpt_forecast_stk where create_date>='" + \
                    str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "' and report_type in (21, 23, 24, 26, 22, 25)) a " + \
                    "left join (select report_id, author_id from rpt_report_author) b " \
                    "on a.report_id = b.report_id) " \
                    "order by create_date, stock_code"
        self.rpt_count = self.db.gg_new_engine.get_original_data(sql_query)
        self.rpt_count['create_date'] = pd.to_datetime(self.rpt_count['create_date'])
        pass

    # 根据设置的参数, 计算analyst coverage
    def get_analyst_coverage_parallel(self, *, rolling_days=90, great_rpt_extra_score=1,
                                      great_author_extra_score=1):
        """
        :param rolling_days: 滚动窗口
        :param great_rpt_extra_score: 每份深度报告的额外得分
        :param great_author_extra_score: 新财富分析师所写报告的额外得分
        :return:
        """
        # # 先构造一个pivot table,主要目的是为了取时间
        # date_mark = self.rpt_count.pivot_table(index='create_date', columns='stock_code',
        #                                       values='forecast_profit')
        # # 因为数据有每天不同时点的数据,因此要resample
        # date_mark = date_mark.resample('d').mean().dropna(axis=0, how='all')
        # 暂时只算周末的值
        self.generate_holding_days(holding_freq='d', loc=-1, start_date=self.db.start_date,
                                   end_date=self.db.end_date)
        holding_days = self.holding_days
        # 将新财富分析师数据对分析师去重, 即, 如果一个人多次得到新财富, 则只取录入数据库最早的那一次
        # 多次得到新财富并不会再次加分, 因此只取最早的那一次即可, 注意是录入最早的那一次, 而不是得奖年份最早
        unique_great_author = self.great_author.sort_values(['author_id', 'entrydate']). \
            drop_duplicates(['author_id'], keep='first')
        # 将新财富分析师数据left join到rpt_count上去
        rpt_count = self.rpt_count.merge(unique_great_author[['author_id', 'entrydate']],
                                         how='left', on='author_id')
        # 将分析师从业开始时间left join到rpt count上去
        rpt_count = rpt_count.merge(self.author_info[['author_id', 'career']],
                                    how='left', on='author_id')

        global rpt_count_g
        rpt_count_g = rpt_count
        # 计算每期的coverage的函数count
        def one_time_coverage(cursor):
            end_time = holding_days.index[cursor]
            start_time = end_time - pd.DateOffset(days=rolling_days - 1)
            # 满足最近rolling days天的条件的项
            # 使用的数据日期为create date, 如果从写报告到入库时间差显著的话,
            # 也可以考虑使用entry_time, 此时entry_time更为真实
            condition = np.logical_and(rpt_count_g['create_date'] >= start_time,
                                       rpt_count_g['create_date'] <= end_time)
            valid_rpt = rpt_count_g[condition] * 1
            # 增加得分这一列, 所有报告的默认得分是1
            valid_rpt['score'] = 0
            # 如果报告属于深度报告, 则加上额外的分数, 深度报告的分类是23, 24, 26
            valid_rpt['score'] = valid_rpt['score'].mask(valid_rpt['report_type'].isin((23, 24, 26)),
                                                         valid_rpt['score'] + great_rpt_extra_score)
            # 如果报告为新财富分析师所写, 且录入时间在当前时间之前, 则加上额外的分数
            valid_rpt['score'] = valid_rpt['score'].mask(valid_rpt['entrydate'] <= end_time,
                                                         valid_rpt['score'] + great_author_extra_score)
            # 如果为新财富分析师所写, 且是深度报告, 再额外加分
            valid_rpt['score'] = valid_rpt['score'].mask(np.logical_and(valid_rpt['report_type']. \
                isin((23, 24, 26)), valid_rpt['entrydate'] <= end_time),
                valid_rpt['score'] + 2)
            # # 根据分析师从业年限给不同的分数
            # career_days = (end_time - valid_rpt['career']).map(lambda x: x.days if type(x) !=
            #                                                    pd.tslib.NaTType else 0)
            # # 分析师从业年限在2年以下的, 直接不给分数
            # valid_rpt['score'] = valid_rpt['score'].mask(career_days < 365*2, 0)
            # # 分析师从业年限在3年以下的不给额外分数, 3-5年给2分, 5-10年给4分, 10年以上6分
            # valid_rpt['score'] = valid_rpt['score'].mask(np.logical_and(
            #     career_days < 365*5, career_days>=365*3), valid_rpt['score'] + 2)
            # valid_rpt['score'] = valid_rpt['score'].mask(np.logical_and(
            #     career_days < 365*10, career_days>=365*5), valid_rpt['score'] + 4)
            # valid_rpt['score'] = valid_rpt['score'].mask(career_days>=365*10, valid_rpt['score'] + 6)
            # 将报告按照所预测股票, 机构id, 作者, 以及得分进行排序, 前3项作为判定独立报告的条件
            # 最后的得分则是由于一份报告可能由多个作者所写, 因此得分不同, 此时只取最高的得分, 即
            # 只要作者中有一个是新财富分析师, 则报告就会享受新财富分析师的额外加分
            valid_rpt = valid_rpt.sort_values(['stock_code', 'organ_id', 'author_name', 'score'])
            # 根据预测股票, 机构, 作者进行去重, 选取最后一个(因为刚刚得分是按照升序排列的)
            unique_valid_rpt = valid_rpt.drop_duplicates(['stock_code', 'organ_id', 'author_name'],
                                                         keep='last')
            # 根据股票进行得分加总
            coverage = unique_valid_rpt.groupby('stock_code').apply(lambda x: x['score'].sum())
            return coverage

        # 进行并行计算
        ncpus = 20
        p = mp.ProcessPool(ncpus)
        data_size = np.arange(holding_days.shape[0])
        chunksize = int(len(data_size)/ncpus)
        results = p.map(one_time_coverage, data_size, chunksize=chunksize)
        analyst_coverage = pd.concat([i for i in results], axis=1)
        self.analyst_coverage = analyst_coverage.T.set_index(self.holding_days).reindex(index=
            self.strategy_data.stock_price.major_axis, columns=self.strategy_data.stock_price.minor_axis)

        # 要将holding days给清空, 否则测试的时候策略会变成每日换仓
        self.holding_days = pd.Series()

    def construct_factor(self):
        self.new_fortune_author()
        self.get_author_info()
        self.get_analyst_report_count()
        self.get_analyst_coverage_parallel()
        # self.strategy_data.factor = pd.Panel({'analyst_coverage': self.analyst_coverage.shift(1)})
        self.prepare_data()
        self.get_abn_coverage()
        self.strategy_data.factor = pd.Panel({'abn_coverage': self.abn_coverage.shift(1)})

    # 准备计算abnormal coverage的数据
    def prepare_data(self):
        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.analyst_coverage = self.analyst_coverage.fillna(0)

        # 计算ln(1+coverage)得到回归的y项
        self.strategy_data.raw_data = pd.Panel({'ln_coverage': np.log(self.analyst_coverage + 1)})
        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'])
        turnover = (data_to_be_used.ix['Volume']/data_to_be_used.ix['FreeShares']).rolling(63).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj']/data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(63).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        self.strategy_data.stock_price['vlty'] = self.strategy_data.stock_price.ix['daily_return'].rolling(252).std()
        pb = data.read_data('PB')
        self.strategy_data.stock_price['lbm'] = np.log(1 + 1/pb)
        roa_data = data.read_data(['TotalAssets', 'NetIncome_ttm'])
        self.strategy_data.stock_price['roa'] = roa_data['NetIncome_ttm'] / roa_data['TotalAssets']

        # 进行标准化
        for item in ('lncap', 'turnover', 'momentum', 'vlty', 'lbm', 'roa'):
            self.strategy_data.raw_data[item] = strategy_data.get_cap_wgt_exposure(
                self.strategy_data.stock_price[item], self.strategy_data.stock_price['FreeMarketValue'])
        self.strategy_data.raw_data['ln_coverage_expo'] = strategy_data.get_cap_wgt_exposure(
            self.strategy_data.raw_data['ln_coverage'], self.strategy_data.stock_price['FreeMarketValue'])

        self.strategy_data.raw_data['const'] = 1.0

    # 用不同的方法计算各种abnormal coverage
    def get_abn_coverage(self):
        # 取回归掉论文中的6个因子后的残差作为abn coverage
        weights = np.sqrt(self.strategy_data.stock_price['FreeMarketValue'])

        # outcome = strategy_data.simple_orth_gs(self.strategy_data.raw_data['ln_coverage_expo'],
        #     self.strategy_data.raw_data.ix[['lncap', 'turnover', 'momentum']],
        #     weights=weights)
        # self.abn_coverage = outcome[0]

        # 对barra风格因子进行回归
        bb_factor_expo = data.read_data('bb_factor_expo_all')
        bb_style_expo = bb_factor_expo.iloc[0:10, :, :]
        outcome = strategy_data.simple_orth_gs(self.strategy_data.raw_data['ln_coverage_expo'],
            bb_factor_expo, weights=weights)
        self.outcome = outcome
        self.abn_coverage = outcome[0]

        # bb_reg_base = pd.concat([self.strategy_data.raw_data.ix[['ln_coverage_expo']], bb_factor_expo. \
        #     drop('country_factor', axis=0)], axis=0)
        # fm_result8 = FamaMacBeth(self.r, bb_reg_base.ix[:, self.holding_days, :]).fit()
        pass


    # 按照论文的因子, 对coverage进行控制其他变量后的fm回归检验
    def abn_coverage_test(self, *, freq='w', startdate=None, enddate=None):
        # 首先需要按照频率生成holdingdays
        self.generate_holding_days(holding_freq=freq, loc=-1, start_date=startdate,
                                   end_date=enddate)
        # 按照频率算收益率, 和holdingdays同步, 论文用月, 我们一般用w
        r = self.strategy_data.stock_price['daily_return', startdate:enddate, :].\
            resample(freq).sum()
        # 注意, 回归的左边是未来一期的收益率, 因此要shift(-1), 即用到未来数据
        r = r.shift(-1).dropna(how='all')
        # 因为r的index为月末, 但是月末不一定是交易日, 因此将r的index重置为holding days
        self.r = r.set_index(self.holding_days)

        # # 进行fm回归
        # fm_result1 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'const'],
        #     self.holding_days, :]).fit()
        # fm_result2 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'const'], self.holding_days, :]).fit()
        # fm_result3 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'turnover', 'const'], self.holding_days, :]).fit()
        # fm_result4 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'turnover', 'momentum', 'const'], self.holding_days, :]).fit()
        # fm_result5 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'turnover', 'momentum', 'vlty', 'const'], self.holding_days, :]).fit()
        # fm_result6 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'turnover', 'momentum', 'vlty', 'lbm', 'const'], self.holding_days, :]).fit()
        # fm_result7 = FamaMacBeth(self.r, self.strategy_data.raw_data.ix[['ln_coverage_expo', 'lncap',
        #     'turnover', 'momentum', 'vlty', 'lbm', 'roa', 'const'], self.holding_days, :]).fit()

        pass

    # 根据原始的coverage值, 画面板数据的kde值
    def draw_kde(self):
        unique_coverage = self.strategy_data.factor.ix['analyst_coverage', '2009-05-04':, :].fillna(0.0)
        self.strategy_data.stock_pool = 'hs300'
        self.strategy_data.handle_stock_pool()
        unique_coverage = unique_coverage.where(self.strategy_data.if_tradable['if_inv',
                                                '2009-05-04':, :])
        stacked_uc = unique_coverage.stack(dropna=True)

        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.hist(stacked_uc.values, bins=int(stacked_uc.max()))
        plt.savefig(str(os.path.abspath('.')) + '/analyst_coverage_new/Great_300_kde.png', dpi=1200)

    # 画财报发布前后的公司的分析师报告数的变化情况, 用来分析分析师的研报数量是否和公司财报好坏有关
    def compare_coverage_around_fra(self, *, rolling_days=90):
        # 从LC_MainIndexNew中取basicEPS_YOY数据, 用来衡量财报的好坏
        sql_query = "select a.EndDate, a.eps_yoy, b.InfoPublDate, c.SecuCode from " \
                    "(SELECT CompanyCode, BasicEPSYOY as eps_yoy, EndDate " \
                    "FROM [JYDB].[dbo].[LC_MainIndexNew]) a " \
                    "left join " \
                    "(select CompanyCode, EndDate, InfoPublDate " \
                    "from jydb.dbo.LC_IncomeStatementAll where IfMerged=1) b " \
                    "on a.CompanyCode = b.CompanyCode and a.EndDate=b.EndDate " \
                    "left join " \
                    "(select distinct CompanyCode, SecuCode from SmartQuant.dbo.ReturnDaily) c " \
                    "on a.CompanyCode = c.CompanyCode " \
                    "order by SecuCode "
        eps_yoy_original = self.db.jydb_engine.get_original_data(sql_query)
        # 针对同一期的财报, 只取最初发布的那一次, 即重述的不要
        eps_yoy_original = eps_yoy_original.sort_values(['SecuCode', 'EndDate', 'InfoPublDate']). \
            groupby(['SecuCode', 'EndDate']).first().reset_index()
        # 去除2007年以前的数据, 并去掉nan
        eps_yoy_original = eps_yoy_original[eps_yoy_original['InfoPublDate'] >=
                                            pd.Timestamp('20070101')].dropna()
        # 对eps yoy进行打分
        eps_yoy_original['eps_yoy_rank'] = eps_yoy_original['eps_yoy'].rank()
        # 取每个财报发布日的前一天, 取这一天的分析师覆盖数据
        eps_yoy_original['previous_date'] = eps_yoy_original['InfoPublDate'] - pd.Timedelta(days=1)
        # 取每个财报发布日后的第rolling_days - 1天
        eps_yoy_original['following_date'] = eps_yoy_original['InfoPublDate'] + pd.Timedelta(days=rolling_days-1)
        # 将数据中存在的股票, coverage中不存在的股票去除, 这样不会出现索引错误
        eps_yoy_original['SecuCode'] = eps_yoy_original['SecuCode'].where(eps_yoy_original['SecuCode']. \
            isin(self.analyst_coverage.columns), np.nan)
        eps_yoy_original = eps_yoy_original.dropna()
        # 因为前一天和后rolling_days - 1天都有可能不是交易日, 因此取最近的一天的覆盖值
        eps_yoy_original['previous_coverage'] = eps_yoy_original.apply(lambda x:
            self.analyst_coverage[x['SecuCode']].asof(x['previous_date']), axis=1)
        eps_yoy_original['following_coverage'] = eps_yoy_original.apply(lambda x:
            self.analyst_coverage[x['SecuCode']].asof(x['following_date']), axis=1)


        pass


if __name__ == '__main__':
    ac = analyst_coverage_new()
    # ac.get_fra_pub_date()
    # ac.data_description()
    # ac.get_unique_coverage_number_parallel()
    # ac.get_unique_coverage_number()
    # ac.time_about_analyst_report()
    # ac.get_analyst_report_structure()
    # ac.new_fortune_author()
    # ac.get_author_info()
    # ac.get_analyst_report_count()
    # ac.get_analyst_coverage_parallel()
    # ac.compare_coverage_around_fra()
    # ac.prepare_data()
    # ac.abn_coverage_test(startdate=pd.Timestamp('2009-05-04'), enddate=pd.Timestamp('2018-01-16'))
    # ac.get_abn_coverage()
    # ac.construct_factor()
    # ac.draw_kde()

    # import alphalens as alphl
    # alphl.utils.get_clean_factor_and_forward_returns()
    # alphl.tears.create_full_tear_sheet()

































































