import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy
import pathos.multiprocessing as mp
import statsmodels as sm

from data import data
from strategy_data import strategy_data
from factor_base import factor_base
from barra_base import barra_base
from db_engine import db_engine

# 生产中的barra base类, 与barra base的主要不同是读取储存数据不是在本地, 而是在数据库

class barra_base_prod(barra_base):
    """ Barra base class in production system.

    foo
    """
    def __init__(self, *, stock_pool='all'):
        barra_base.__init__(self, stock_pool=stock_pool)
        # 初始化该类的risk model数据库引擎, 因为读取, 存入数据都是在这个数据库中完成
        self.initialize_rm()
        # 永远不允许从本地读取数据
        self.try_to_read = False

        # 读取数据的时候需要用到的开始和结束时间, 一般在更新数据的时候用到.
        # 开始日期默认为2007年1月1日, 结束时间默认为今天
        self.start_date = pd.Timestamp('2007-01-01')
        self.end_date = pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d'))

    # 初始化链接到RiskModel数据库的引擎
    def initialize_rm(self):
        self.rm_engine = db_engine(server_type='mssql', driver='pymssql', username='rmreader', password='OP#567890',
                                     server_ip='192.168.66.12', port='1433', db_name='RiskModel', add_info='')

    # 设置数据起始和结束时间的函数
    def set_start_end_date(self, *, start_date=None, end_date=None):
        if isinstance(start_date, pd.Timestamp):
            self.start_date = start_date
        if isinstance(end_date, pd.Timestamp):
            self.end_date = end_date
            assert self.end_date >= self.start_date, 'Please make sure that end date is not earlier than ' \
                                                     'start date!\n'

    # 从数据库读取原始数据的函数, 这个函数是生产类区别于研究类的其中一个关键点
    def read_original_data(self):
        # 需要读取的数据的tuple
        data_needed = ('FreeMarketValue', 'MarketValue', 'ClosePrice_adj', 'Volume', 'FreeShares', 'PB',
                       'NetIncome_fy1', 'NetIncome_fy2', 'CashEarnings_ttm', 'PE_ttm', 'NetIncome_ttm',
                       'NetIncome_ttm_growth_8q', 'Revenue_ttm_growth_8q', 'TotalAssets', 'TotalLiability',
                       'is_enlisted', 'is_delisted', 'is_suspended')
        # 根据投资域读取基准权重数据, base类里没有基准的概念, 因为base类不像策略, 没有对比和参考的意义,
        # 投资域就可以决定一个base, 这里暂时不支持传入一个dataframe作为投资域的情况
        if self.base_data.stock_pool != 'all':
            data_needed = data_needed + ('Weight_' + self.base_data.stock_pool, )
        sql_data_needed = "select * from RawData where DataDate >= '" + str(self.start_date) + "' and " \
                          "DataDate <= '" + str(self.end_date) + "' and DataName in " + str(data_needed)
        original_RawData = self.rm_engine.get_original_data(sql_data_needed)
        original_RawData = original_RawData.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel().transpose(2, 1, 0)
        # 将数据分配到base data的各个部分中去
        self.base_data.stock_price = original_RawData.ix[['FreeMarketValue', 'MarketValue', 'ClosePrice_adj',
                                                          'Volume', 'FreeShares']]
        self.base_data.raw_data = original_RawData.ix[['PB', 'NetIncome_fy1', 'NetIncome_fy2', 'CashEarnings_ttm',
                                                       'PE_ttm', 'NetIncome_ttm', 'NetIncome_ttm_growth_8q',
                                                       'Revenue_ttm_growth_8q', 'TotalAssets', 'TotalLiability']]
        self.base_data.if_tradable = original_RawData.ix[['is_enlisted', 'is_delisted', 'is_suspended']].fillna(0)
        if self.base_data.stock_pool != 'all':
            self.base_data.benchmark_price = original_RawData.ix[['Weight_'+self.base_data.stock_pool]].fillna(0.0)

        # 从const data表中取无风险利率数据
        sql_const_data = "select DataDate, RiskFreeRate from ConstData where DataDate >= '" + \
                         str(self.start_date) + "' and DataDate <= '" + str(self.end_date) + "' "
        risk_free_rate = self.rm_engine.get_original_data(sql_const_data)
        if not risk_free_rate.empty:
            self.base_data.const_data = risk_free_rate.set_index('DataDate')
        else:
            self.base_data.const_data = pd.DataFrame(0, index=self.base_data.stock_price.major_axis,
                                                     columns=['RiskFreeRate'])
        # 计算每只股票的日对数收益率
        self.base_data.stock_price['daily_log_return'] = np.log(self.base_data.stock_price.ix['ClosePrice_adj'].div(
            self.base_data.stock_price.ix['ClosePrice_adj'].shift(1)))
        # 计算每只股票的日超额对数收益
        self.base_data.stock_price['daily_excess_log_return'] = self.base_data.stock_price.ix[
            'daily_log_return'].sub(self.base_data.const_data.ix[:, 'RiskFreeRate'], axis=0)
        # 计算每只股票的日简单收益，注意是按日复利的日化收益，即代表净值增值
        self.base_data.stock_price['daily_simple_return'] = self.base_data.stock_price.ix['ClosePrice_adj'].div(
            self.base_data.stock_price.ix['ClosePrice_adj'].shift(1)).sub(1.0)
        # 计算每只股票的日超额简单收益，注意是按日复利的日化收益，
        # 另外注意RiskFreeRate默认是连续复利，要将其转化成对应的简单收益
        self.base_data.const_data['RiskFreeRate_simple'] = np.exp(self.base_data.const_data['RiskFreeRate']) - 1
        self.base_data.stock_price['daily_excess_simple_return'] = self.base_data.stock_price. \
            ix['daily_simple_return'].sub(self.base_data.const_data['RiskFreeRate_simple'], axis=0)

        # 读取行业数据
        sql_industry = "select * from IndustryMark where DataDate >= '" + str(self.start_date) + "' and " \
                       "DataDate <= '" + str(self.end_date) + "' "
        industry = self.rm_engine.get_original_data(sql_industry)
        industry = industry.pivot_table(index='DataDate', columns='SecuCode', values='Value', aggfunc='first')
        self.industry = industry

        # 生成可交易及可投资数据, 此时由于数据已经存在在if_tradable和benchmark_price中, 因此不函数不会尝试从本地读取数据
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        # 和研究的版本一样的数据处理
        self.base_data.discard_untradable_data()
        self.complete_base_data = copy.deepcopy(self.base_data)
        # 和研究版本一样, 过滤不可投资数据
        self.base_data.discard_uninv_data()

    # 建立barra风格因子数据库
    def create_base_factor_table(self):
        sql = "drop table BarraBaseFactor; " \
              "create table BarraBaseFactor ( " \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "FactorName varchar(255), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立barra风格因子暴露数据库
    def create_base_factor_expo_table(self):
        sql = "drop table BarraBaseFactorExpo; " \
              "create table BarraBaseFactorExpo ( " \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "FactorName varchar(255), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立barra的因子收益表
    def create_base_factor_return_table(self):
        sql = "drop table BarraBaseFactorReturn; " \
              "create table BarraBaseFactorReturn ( " \
              "DataDate smalldatetime, " \
              "FactorName varchar(255), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立barra base下的股票的specific return的表
    def create_base_specific_return_table(self):
        sql = "drop table BarraBaseSpecificReturn; " \
              "create table BarraBaseSpecificReturn ( " \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 储存风格因子数据
    def save_factor_to_sql(self):
        self.base_data.factor.major_axis.rename('DataDate', inplace='True')
        for item, df in self.base_data.factor.iteritems():
            sql_df = pd.melt(df.reset_index(level=0), id_vars='DataDate', var_name='SecuCode',
                             value_name='Value').dropna()
            sql_df['FactorName'] = item
            sql_df['StockPool'] = self.base_data.stock_pool

            # 储存到数据库中
            sql_df.to_sql('BarraBaseFactor', self.rm_engine.engine, if_exists='append', index=False)
            # 打印储存成功的提示
            print('Factor: {0} has been successfully saved into sql database!\n'.format(item))

    # 储存因子暴露数据
    # 这里不储存行业暴露, 因为行业暴露简单, 又很多,很占地空间, 因此如果需要直接提取暴露数据,
    # 则需要从IndustryMark中提取原始数据, 然后像get_industry_expo函数这样自己做成dummy variable
    def save_factor_expo_to_sql(self):
        self.base_data.factor_expo.major_axis.rename('DataDate', inplace=True)
        style_factor_expo = self.base_data.factor_expo.iloc[:self.n_style]
        for item, df in style_factor_expo.iteritems():
            sql_df = pd.melt(df.reset_index(level=0), id_vars='DataDate', var_name='SecuCode',
                             value_name='Value').dropna()
            sql_df['FactorName'] = item
            sql_df['StockPool'] = self.base_data.stock_pool

            # 储存到数据库中
            sql_df.to_sql('BarraBaseFactorExpo', self.rm_engine.engine, if_exists='append', index=False)
            # 打印储存成功的提示
            print('FactorExpo: {0} has been successfully saved into sql database!\n'.format(item))

    # 储存因子收益数据
    def save_factor_return_to_sql(self):
        self.base_factor_return.index.rename('DataDate', inplace=True)
        sql_df = pd.melt(self.base_factor_return.reset_index(level=0), id_vars='DataDate',
                         var_name='FactorName', value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseFactorReturn', self.rm_engine.engine, if_exists='append', index=False)
        # 打印储存成功的提示
        print('FactorReturn has been successfully saved into sql database!\n')

    # 储存specific return数据
    def save_specific_return_to_sql(self):
        self.specific_return.index.rename('DataDate', inplace=True)
        sql_df = pd.melt(self.specific_return.reset_index(level=0), id_vars='DataDate',
                         var_name='SecuCode', value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseSpecificReturn', self.rm_engine.engine, if_exists='append', index=False)
        # 打印储存成功的提示
        print('SpecificReturn has been successfully saved into sql database!\n')

    # save_data函数将所有需要储存的数据储存到数据库中去
    def save_data(self):
        self.save_factor_to_sql()
        self.save_factor_expo_to_sql()
        self.save_factor_return_to_sql()
        self.save_specific_return_to_sql()

    # 从数据库中删除数据的函数, 用于更新的时候, 需要更新的数据已经在数据库中, 则要把这些数据删除掉
    # 同database_prod中的删除函数几乎一模一样, 除了需要加上代表投资域的stock pool
    def delete_from_sql_table(self, stock_pool, *, table_name='BarraBaseFactor', factor_name=None,
                              data_date=None):
        sql_delete = "delete from " + table_name + " where StockPool = '" + str(stock_pool) + "' "
        # 将list的参数改写为sql中的形式
        if factor_name is not None:
            sql_factor_name = "FactorName in ("
            for name in factor_name:
                sql_factor_name += "'" + str(name) + "', "
            # 去掉最后一个空格和逗号
            sql_factor_name = sql_factor_name[:-2]
            sql_factor_name += ") "
        if data_date is not None:
            sql_data_date = "DataDate in ("
            for date in data_date:
                sql_data_date += "'" + str(date) + "', "
            sql_data_date = sql_data_date[:-2]
            sql_data_date += ") "

        if factor_name is not None and data_date is None:
            sql_delete = sql_delete + "and " + sql_factor_name
        elif factor_name is None and data_date is not None:
            sql_delete = sql_delete + "and " + sql_data_date
        elif factor_name is not None and data_date is not None:
            sql_delete = sql_delete + "and " + sql_factor_name + 'and ' + sql_data_date

        self.rm_engine.engine.execute(sql_delete)

    # 构建barra base的所有风格因子和行业因子
    # 注意这里的构建函数需要关系新建和删除表的问题
    def construct_factor_base(self, *, if_save=False):
        # 如果更新数据, 则不需要删除旧表新建新表
        if not self.is_update:
            self.create_base_factor_table()
            self.create_base_factor_expo_table()
            self.create_base_factor_return_table()
            self.create_base_specific_return_table()
        self.construct_reading_file_appendix()
        # 与研究版本不同, 即使是更新数据, 也是需要读取数据的
        self.read_original_data()
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
        self.get_bp()
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

        # 如果是更新数据, 则不在这里储存, 而是在更新数据的地方储存
        if not self.is_update and if_save:
            self.save_data()

    # 更新数据的函数, 这个函数是生产类区别于研究类的另一个关键点
    def update_factor_base_data(self, *, start_date=None):
        self.is_update = True
        # 永远不允许从本地读取数据
        self.try_to_read = False

        # 取数据库中最后一天的日期, 以BarraBaseFactor数据库为准
        last_day = self.rm_engine.get_original_data("select max(DataDate) from BarraBaseFactor where "
            "StockPool = '" + self.base_data.stock_pool + "' ").iloc[0, 0]
        # 如果有传入的指定更新数据的开始时间, 则选取last_day和指定开始时间更早的那天
        if isinstance(start_date, pd.Timestamp):
            update_start_date = min(start_date, last_day)
        else:
            update_start_date = last_day
        # 打印更新起始日作为参考
        print('BarraBase base data update starts from {0}!\n'.format(update_start_date))

        # 当衔接新旧数据的时候, 需要丢弃的数据是从开始更新的那一天, 一直到last_day
        sql_dropped_time = "select distinct DataDate from BarraBaseFactor where DataDate >= '" + \
                           str(update_start_date) + "' and DataDate <= '" + str(last_day) + "' and " \
                           "StockPool = '" + self.base_data.stock_pool + "' order by DataDate"
        dropped_time = self.rm_engine.get_original_data(sql_dropped_time).iloc[:, 0].tolist()

        # 由于取的原始数据要从更新开始的第一天的前525个交易日取, 因此, 需要得到self.start_date,
        # 即取原始数据的开始日期, 由于barra因子定义中最早使用的是525个交易日前的数据,
        # 另外, 由于计算因子收益时需要用到前一天的因子暴露, 因此这里还需要计算更新时间前一个交易日的因子暴露
        # 因此, 取数据的时候实际上会取小于更新开始日期的525个交易日, 而不是小于等于
        sql_start_date = "select top 525 DataDate from (select distinct DataDate from RawData where " \
                         "DataDate < '" + str(update_start_date) + "' ) temp order by DataDate desc "
        self.start_date = self.rm_engine.get_original_data(sql_start_date).iloc[-1, 0]

        # 更新数据
        self.construct_factor_base()
        # 计算因子收益
        self.get_base_factor_return()

        # 首先删除这些数据, 根据dropped time把老数据删除
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseFactor',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseFactorExpo',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseFactorReturn',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseSpecificReturn',
                                   data_date=dropped_time)

        # 插入数据库的新数据的时间从update start time开始一直到有数据的最后一个
        self.base_data.factor = self.base_data.factor.ix[:, update_start_date:, :]
        self.base_data.factor_expo = self.base_data.factor_expo.ix[:, update_start_date:, :]
        self.base_factor_return = self.base_factor_return.ix[update_start_date:, :]
        self.specific_return = self.specific_return.ix[update_start_date:, :]
        # 储存数据
        self.save_data()

        self.is_update = False


    ###########################################################################################################
    # 下面是风险预测的部分
    ###########################################################################################################

    # 建立储存风险预测中的因子协方差数据的数据库
    def create_base_cov_mat_table(self):
        sql = "drop table BarraBaseCovMat; " \
              "create table BarraBaseCovMat (" \
              "DataDate smalldatetime, " \
              "FactorName1 varchar(255), " \
              "FactorName2 varchar(255), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立储存风险预测中的个股特定风险数据的数据库
    def create_base_spec_var_table(self):
        sql = "drop table BarraBaseSpecVar; " \
              "create table BarraBaseSpecVar (" \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立储存每日的当个因子方差的预测的数据库, 在做vra的时候需要用到
    def create_base_daily_factor_var_table(self):
        sql = "drop table BarraBaseDailyFactorVar; " \
              "create table BarraBaseDailyFactorVar (" \
              "DataDate smalldatetime, " \
              "FactorName varchar(255), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 建立储存每日的每支股票特定风险预测, 在做vra的时候需要用到
    def create_base_daily_spec_var_table(self):
        sql = "drop table BarraBaseDailySpecVar; " \
              "create table BarraBaseDailySpecVar (" \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "StockPool varchar(10) )"
        self.rm_engine.engine.execute(sql)

    # 储存因子协方差数据
    def save_cov_mat_to_sql(self):
        # 首先要将cov mat从panel转为dataframe
        cov_mat_df = self.eigen_adjusted_cov_mat.transpose(1, 0, 2).to_frame().reset_index(level=[0, 1])
        # 要重新命名, 更新时和全部计算时的列名不同, 因此, 都要考虑到
        cov_mat_df.rename(columns={'major': 'DataDate', 'minor': 'FactorName1',
                                   'FactorName':'FactorName1'}, inplace=True)
        sql_df = pd.melt(cov_mat_df, id_vars=['DataDate', 'FactorName1'], var_name='FactorName2',
                         value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseCovMat', self.rm_engine.engine, if_exists='append', index=False)
        print('Factor Covariance Matrix has been successfully saved into sql database!\n')

    # 储存股票特定风险数据
    def save_spec_var_to_sql(self):
        self.spec_var.index.rename('DataDate', inplace=True)
        sql_df = pd.melt(self.spec_var.reset_index(level=0), id_vars='DataDate',
                         var_name='SecuCode', value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseSpecVar', self.rm_engine.engine, if_exists='append', index=False)
        print('Specific Variance has been successfully saved into sql database!\n')

    # 储存每日的因子方差
    def save_daily_factor_var_to_sql(self):
        self.daily_var_forecast.index.rename('DataDate', inplace=True)
        sql_df = pd.melt(self.daily_var_forecast.reset_index(level=0), id_vars='DataDate',
                         var_name='FactorName', value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseDailyFactorVar', self.rm_engine.engine, if_exists='append', index=False)
        print('Daily Factor Var has been successfully saved into sql database!\n')

    # 储存每日的每支股票特定风险
    def save_daily_spec_var_to_sql(self):
        self.initial_daily_spec_vol.index.rename('DataDate', inplace=True)
        # 先把数据改为var, 而不是vol
        daily_specific_var = self.initial_daily_spec_vol.pow(2)
        sql_df = pd.melt(daily_specific_var.reset_index(level=0), id_vars='DataDate', var_name='SecuCode',
                         value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseDailySpecVar', self.rm_engine.engine, if_exists='append', index=False)
        print('Daily Spec Var has been successfully saved into sql database!\n')

    # 更新风险预测数据的函数, 这个函数在取数据方面比较复杂, 因为首先是计算更新时间段的初始预测,
    # 然后计算vra的时候, 又需要用到初始预测的时间序列, 因此需要再次取数据,
    # 注意, 由于更新数据的时候会使用base factor return等数据, 因此读取这些数据的时候会把更新base数据的数据覆盖掉
    # 因此, 可以建两个不同的实例, 一个更新base data, 一个更新风险预测, 或是风险预测一定要等到base data更新完毕后再使用
    def update_risk_forecast(self, *, start_date=None, freq='a', covmat_sample_size=504, var_half_life=84,
            corr_half_life=504, var_nw_lag=5, corr_nw_lag=2, vra_sample_size=252, vra_half_life=42,
            eigen_adj_sims=1000, scaling_factor=1.4, specvol_sample_size=360, specvol_half_life=84,
            specvol_nw_lag=5, shrinkage_parameter=0.1):
        # 将freq转到forecast_step
        freq_map =  {'a':252, 'm': 21, 'w': 5}
        forecast_steps = freq_map[freq]

        # 首先仍然是找开始更新的时间, 逻辑和更新base data的时候一样, 以base cov mat数据库为标准
        last_day = self.rm_engine.get_original_data("select max(DataDate) from BarraBaseCovMat where "
            "StockPool = '" + self.base_data.stock_pool + "' ").iloc[0, 0]
        # 如果有传入的指定更新数据的开始时间, 则选取last_day和指定开始时间更早的那天
        if isinstance(start_date, pd.Timestamp):
            update_start_date = min(start_date, last_day)
        else:
            update_start_date = last_day
        # 打印更新起始日作为参考
        print('BarraBase risk forecast update starts from {0}!\n'.format(update_start_date))

        # 需要删除的数据的时间段, 也是基于同样的逻辑
        sql_dropped_time = "select distinct DataDate from BarraBaseCovMat where DataDate >= '" + \
                           str(update_start_date) + "' and DataDate <= '" + str(last_day) + "' and " \
                           "StockPool = '" + self.base_data.stock_pool + "' order by DataDate"
        dropped_time = self.rm_engine.get_original_data(sql_dropped_time).iloc[:, 0].tolist()

        # 删除需要删除的数据
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseCovMat',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseSpecVar',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseDailyFactorVar',
                                   data_date=dropped_time)
        self.delete_from_sql_table(self.base_data.stock_pool, table_name='BarraBaseDailySpecVar',
                                   data_date=dropped_time)

        # 不同的数据需要定义不同的取数据开始时间, 这是根据sample size决定的, 而vra的算法又决定了,
        # 这里不能算出来后一起存, 而要一边算, 一边存.

        # 最开始是读取一些共同要用到的数据, 包括: if tradable中的标记数据, 投资域的权重数据,
        # 流通市值数据, 以及用来算str spec vol的因子暴露数据, 这些数据只需要取更新时间段
        # 注意更新时间段是从update start date到self.end_data
        name_RawData = ('FreeMarketValue', 'is_enlisted', 'is_delisted', 'is_suspended')
        if self.base_data.stock_pool != 'all':
            name_RawData += ('Weight_' + self.base_data.stock_pool, )
        sql_RawData = "select * from RawData where DataDate >= '" + str(update_start_date) + \
                      "' and DataDate <= '" + str(self.end_date) + "' and DataName in " + \
                      str(name_RawData)
        RawData = self.rm_engine.get_original_data(sql_RawData)
        RawData = RawData.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel().transpose(2, 1, 0)
        self.base_data.stock_price = RawData.ix[['FreeMarketValue']]
        self.base_data.if_tradable = RawData.ix[['is_enlisted', 'is_delisted', 'is_suspended']].fillna(0)
        if self.base_data.stock_pool != 'all':
            self.base_data.benchmark_price = RawData.ix[['Weight_' + self.base_data.stock_pool]].fillna(0.0)
        # 构造因子暴露数据
        sql_expo = "select * from BarraBaseFactorExpo where DataDate >= '" + str(update_start_date) + \
                   "' and DataDate <= '" + str(self.end_date) + "' and StockPool = '" + \
                   str(self.base_data.stock_pool) + "' "
        expo = self.rm_engine.get_original_data(sql_expo)
        expo = expo.pivot_table(index=['DataDate', 'FactorName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel().transpose(2, 1, 0)
        # 将顺序换成风格因子在前, 行业因子在后, country factor在最后的形式, 才能在之后做回归
        # 这里将风格因子的顺序换成了熟悉的顺序, 实际上只要行业因子在风格因子后面, country因子最后, 结果就是一样的
        self.base_data.factor_expo = expo[['lncap', 'beta', 'momentum', 'rv', 'nls', 'bp', 'liquidity', 'ey',
            'growth', 'leverage']]
        sql_industry = "select * from IndustryMark where DataDate >= '" + str(update_start_date) + "' and " \
                       "DataDate <= '" + str(self.end_date) + "' "
        industry = self.rm_engine.get_original_data(sql_industry)
        industry = industry.pivot_table(index='DataDate', columns='SecuCode', values='Value', aggfunc='first')
        self.industry = industry
        self.get_industry_factor()
        self.add_country_factor()
        # 生成投资域相关数据
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        # 过滤不可投资的数据
        self.base_data.discard_uninv_data()

        # 然后先计算specific risk, 第一步是算初始的估计值
        # 需要的数据是specvol_sample_size长度的specific return的数据
        sql_spec_start_date = "select top " + str(specvol_sample_size) + " DataDate from ( " \
                              "select distinct DataDate from BarraBaseSpecificReturn where DataDate <= '" + \
                              str(update_start_date) + "' and StockPool = '" + str(self.base_data.stock_pool) + \
                              "' ) temp order by DataDate desc "
        spec_start_date = self.rm_engine.get_original_data(sql_spec_start_date).iloc[-1, 0]
        sql_spec_ret = "select * from BarraBaseSpecificReturn where DataDate >= '" + str(spec_start_date) + \
                       "' and DataDate <= '" + str(self.end_date) + "' and StockPool = '" + \
                       str(self.base_data.stock_pool) + "' "
        spec_ret = self.rm_engine.get_original_data(sql_spec_ret)
        self.specific_return = spec_ret.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                                    aggfunc='first')
        # 进行specific risk的初始估计
        self.get_initial_spec_vol_parallel(sample_size=specvol_sample_size,
            spec_var_half_life=specvol_half_life, nw_lag=specvol_nw_lag, forecast_steps=forecast_steps)
        # 然后需要将估计出来的daily spec var数据存入数据库, 为了保险, 只取开始更新的日期,
        # 虽然本来也应该是这个时间段, 但是害怕中间有什么修改, 所以再索引一次保险一些.
        self.initial_daily_spec_vol = self.initial_daily_spec_vol.ix[update_start_date:]
        self.save_daily_spec_var_to_sql()

        # 第二步是做vra, vra需要的数据长度是vra sample size, 数据为估计出的daily spec var, 以及specific return
        # 注意, 由于估计出的daily spec var是要用shift一天的, 因此这个数据要多取一天, 然后在vra的函数中会进行reindex
        sql_spec_vra_start_date = "select top " + str(vra_sample_size) + " DataDate from ( " \
                                  "select distinct DataDate from BarraBaseSpecificReturn where DataDate < '" + \
                                  str(update_start_date) + "' and StockPool = '" + str(self.base_data.stock_pool) + \
                                  "' ) temp order by DataDate desc "
        # 取specific return数据, 注意会覆盖之前的specific return
        spec_vra_ret_date = self.rm_engine.get_original_data(sql_spec_vra_start_date).iloc[-2, 0]
        sql_vra_spec_ret = "select * from BarraBaseSpecificReturn where DataDate >= '" + str(spec_vra_ret_date) + \
                           "' and DataDate <= '" + str(self.end_date) + "' and StockPool = '" + \
                           str(self.base_data.stock_pool) + "' "
        vra_spec_ret = self.rm_engine.get_original_data(sql_vra_spec_ret)
        self.specific_return = vra_spec_ret.pivot_table(index='DataDate', columns='SecuCode',
                                                        values='Value', aggfunc='first')
        # 取daily spec vol, 注意除了要覆盖之外, 还需要将数据库里的var开根号算出vol
        spec_vra_daily_var_date = self.rm_engine.get_original_data(sql_spec_vra_start_date).iloc[-1, 0]
        sql_vra_daily_spec_var = "select * from BarraBaseDailySpecVar where DataDate >= '" + \
                                 str(spec_vra_daily_var_date) + "' and DataDate <= '" + \
                                 str(self.end_date) + "' and StockPool = '" + \
                                 str(self.base_data.stock_pool) + "' "
        vra_daily_spec_var = self.rm_engine.get_original_data(sql_vra_daily_spec_var)
        self.initial_daily_spec_vol = vra_daily_spec_var.pivot_table(index='DataDate',
            columns='SecuCode', values='Value', aggfunc='first').pow(0.5)
        # 市值数据也要改变, 之前用到的市值数据是更新期的, 现在要改为和specific return一样的时间段
        sql_spec_vra_mv = "select * from RawData where DataDate >= '" + str(spec_vra_ret_date) + \
                          "' and DataDate <= '" + str(self.end_date) + "' and DataName = 'FreeMarketValue' "
        spec_vra_mv = self.rm_engine.get_original_data(sql_spec_vra_mv)
        spec_vra_mv = spec_vra_mv.pivot_table(index='DataDate', columns='SecuCode', values='Value', aggfunc='first')
        # 取出来的市值数据由于数据都是nan的原因(譬如最近一天更新于收盘前),
        # 有可能时间段和specific return不一致, 再次重索引
        self.base_data.stock_price = pd.Panel({'FreeMarketValue': spec_vra_mv}).reindex(
                                              major_axis=self.specific_return.index)
        # 进行vra的计算,
        self.get_vra_spec_vol(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 第三步进行bayesian shrinkage
        # 这一步需要的数据只有市值数据, 且只需要更新期的, 而这个时间段被包含在了vra中取的市值当中,
        # 因此不再需要重新取市值数据了, 直接进行计算即可
        self.get_bayesian_shrinkage_spec_vol(shrinkage_parameter=shrinkage_parameter)
        # 将vol改为var
        self.spec_var = self.bs_spec_vol.pow(2)

        # 之后是计算协方差矩阵
        # 第一步是计算初始的协方差矩阵的估计和日度的因子方差的估计
        # 需要的数据是covmat_sample_size长度的base factor return数据
        sql_covmat_start_date = "select top " + str(covmat_sample_size) + " DataDate from ( " \
                                "select distinct DataDate from BarraBaseFactorReturn where DataDate <= '" + \
                                str(update_start_date) + "' and StockPool = '" + str(self.base_data.stock_pool) + \
                                "' ) temp order by DataDate desc "
        covmat_start_date = self.rm_engine.get_original_data(sql_covmat_start_date).iloc[-1, 0]
        sql_fac_ret = "select * from BarraBaseFactorReturn where DataDate >= '" + str(covmat_start_date) + \
                      "' and DataDate <= '" + str(self.end_date) + "' and StockPool = '" + \
                      str(self.base_data.stock_pool) + "' "
        fac_ret = self.rm_engine.get_original_data(sql_fac_ret)
        self.base_factor_return = fac_ret.pivot_table(index='DataDate', columns='FactorName',
                                                      values='Value', aggfunc='first')
        # 进行初始估计值的计算
        self.get_initial_cov_mat_parallel(sample_size=covmat_sample_size, var_half_life=var_half_life,
                                          corr_half_life=corr_half_life, var_nw_lag=var_nw_lag,
                                          corr_nw_lag=corr_nw_lag, forecast_steps=forecast_steps)
        # 同理储存daily factor var
        self.daily_var_forecast = self.daily_var_forecast.ix[update_start_date:]
        self.save_daily_factor_var_to_sql()

        # 第二步是做vra, 和spec vol中的vra一样, 使用daily factor var, 以及factor return.
        # 同样的, daily factor var要多取一天的
        sql_cov_vra_start_date = "select top " + str(vra_sample_size) + " DataDate from ( " \
                                 "select distinct DataDate from BarraBaseFactorReturn where DataDate < '" + \
                                 str(update_start_date) + "' and StockPool = '" + str(self.base_data.stock_pool) + \
                                 "' ) temp order by DataDate desc "
        # 取factor return数据, 注意会覆盖之前的factor return
        cov_vra_ret_date = self.rm_engine.get_original_data(sql_cov_vra_start_date).iloc[-2, 0]
        sql_vra_fac_ret = "select * from BarraBaseFactorReturn where DataDate >= '" + str(cov_vra_ret_date) + \
                           "' and DataDate <= '" + str(self.end_date) + "' and StockPool = '" + \
                           str(self.base_data.stock_pool) + "' "
        vra_fac_ret = self.rm_engine.get_original_data(sql_vra_fac_ret)
        self.base_factor_return = vra_fac_ret.pivot_table(index='DataDate', columns='FactorName',
                                                          values='Value', aggfunc='first')
        # 取daily factor var, 也要覆盖, 但是由于cov vra函数使用的就是var, 因此不需要开根号
        cov_vra_daily_var_date = self.rm_engine.get_original_data(sql_cov_vra_start_date).iloc[-1, 0]
        sql_vra_daily_fac_var = "select * from BarraBaseDailyFactorVar where DataDate >= '" + \
                                 str(cov_vra_daily_var_date) + "' and DataDate <= '" + \
                                 str(self.end_date) + "' and StockPool = '" + \
                                 str(self.base_data.stock_pool) + "' "
        vra_daily_fac_var = self.rm_engine.get_original_data(sql_vra_daily_fac_var)
        self.daily_var_forecast = vra_daily_fac_var.pivot_table(index='DataDate', columns='FactorName',
                                                                values='Value', aggfunc='first')
        # 进行vra的计算
        self.get_vra_cov_mat(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 第三步进行eigen factor调整, 这一步不需要其他任何新数据, 直接进行计算
        self.get_eigen_adjusted_cov_mat_parallel(n_of_sims=eigen_adj_sims, scaling_factor=scaling_factor,
                                                 simed_sample_size=covmat_sample_size)

        # 更新数据完毕, 进行数据储存
        self.eigen_adjusted_cov_mat = self.eigen_adjusted_cov_mat.ix[update_start_date:]
        self.spec_var = self.spec_var.ix[update_start_date:]
        self.save_cov_mat_to_sql()
        self.save_spec_var_to_sql()


if __name__ == '__main__':
    pool = ['all', 'hs300', 'zz500']
    # factor = ['lncap', 'beta', 'momentum', 'rv', 'nls', 'bp', 'liquidity', 'ey', 'growth', 'leverage']

    for p in pool:
        # bbp = barra_base_prod(stock_pool=p)
        # factor_name = [i + '_' + p for i in factor]
        # bbp.base_data.factor = data.read_data(factor_name, item_name=factor)
        # bbp.base_data.factor_expo = pd.read_hdf('RiskModelData/bb_factor_expo_'+p, '123')
        # bbp.base_factor_return = data.read_data(['bb_factor_return_'+p]).iloc[0]
        # bbp.specific_return = data.read_data(['bb_specific_return_'+p]).iloc[0]
        # bbp.get_factor_group_count()

        # bbp.base_data.factor.major_axis.rename('DataDate', inplace=True)
        # bbp.base_data.factor_expo.major_axis.rename('DataDate', inplace=True)
        # bbp.base_factor_return.index.rename('DataDate', inplace=True)
        # bbp.specific_return.index.rename('DataDate', inplace=True)

        # bbp.save_data()
        # bbp.save_factor_expo_to_sql()


        # bbp = barra_base_prod(stock_pool=p)
        # bbp.eigen_adjusted_cov_mat = pd.read_hdf('RiskModelData/bb_riskmodel_covmat_'+p, '123')
        # bbp.spec_var = pd.read_hdf('RiskModelData/bb_riskmodel_specvar_'+p, '123')
        # bbp.initial_daily_spec_vol = pd.read_hdf('RiskModelData/bb_riskmodel_dailyspecvar_'+p, '123').pow(0.5)
        # bbp.daily_var_forecast = pd.read_hdf('RiskModelData/bb_riskmodel_dailyfacvar_'+p, '123')
        #
        # bbp.spec_var.index.rename('DataDate', inplace=True)
        # bbp.initial_daily_spec_vol.index.rename('DataDate', inplace=True)
        # bbp.daily_var_forecast.index.rename('DataDate', inplace=True)
        #
        # bbp.save_daily_factor_var_to_sql()
        # bbp.save_daily_spec_var_to_sql()
        # bbp.save_cov_mat_to_sql()
        # bbp.save_spec_var_to_sql()

        # bbp = barra_base_prod(stock_pool=p)
        # bbp.update_factor_base_data(start_date=pd.Timestamp('2017-12-13'))

        bbp = barra_base_prod(stock_pool=p)
        bbp.update_risk_forecast(start_date=pd.Timestamp('2017-12-13'))

        print(p+' has been completed!\n')
        pass











































