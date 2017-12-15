import numpy as np
import pandas as pd
from datetime import datetime
import copy

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
        barra_base.__init__(stock_pool=stock_pool)
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
        self.rm_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
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
            data_needed = data_needed + ('Weight_'+self.base_data.stock_pool, )
        sql_data_needed = "select * from RawData where DataDate >= '" + str(self.start_date) + "' and " \
                          "DataDate <= '" + str(self.end_date) + "' and DataName in " + str(data_needed)
        original_RawData = self.rm_engine.get_original_data(sql_data_needed)
        original_RawData = original_RawData.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel.transpose(2, 1, 0)
        # 将数据分配到base data的各个部分中去
        self.base_data.stock_price = original_RawData.ix[['FreeMarketValue', 'MarketValue', 'ClosePrice_adj',
                                                          'Volume', 'FreeShares']]
        self.base_data.raw_data = original_RawData.ix[['PB', 'NetIncome_fy1', 'NetIncome_fy2', 'CashEarnings_ttm',
                                                       'PE_ttm', 'NetIncome_ttm', 'NetIncome_ttm_growth_8q',
                                                       'Revenue_ttm_growth_8q', 'TotalAssets', 'TotalLiability']]
        self.base_data.if_tradable = original_RawData.ix[['is_enlisted', 'is_delisted', 'is_suspended']]
        if self.base_data.stock_pool != 'all':
            self.base_data.benchmark_price = original_RawData.ix[['Weight_'+self.base_data.stock_pool]]

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
        sql_df = pd.melt(self.base_factor_return.reset_index(level=0), id_vars='DataDate',
                         var_name='FactorName', value_name='Value').dropna()
        sql_df['StockPool'] = self.base_data.stock_pool
        # 储存到数据库中
        sql_df.to_sql('BarraBaseFactorReturn', self.rm_engine.engine, if_exists='append', index=False)
        # 打印储存成功的提示
        print('FactorReturn has been successfully saved into sql database!\n')

    # 储存specific return数据
    def save_specific_return_to_sql(self):
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

    # 构建barra base的所有风格因子和行业因子
    # 注意这里的构建函数需要关系新建和删除表的问题
    def construct_factor_base(self, *, if_save=False):
        # 如果更新数据, 则不需要删除旧表新建新表
        if not self.is_update:
            self.create_base_factor_table()
            self.create_base_factor_expo_table()
            self.create_base_factor_return_table()
            self.create_base_specific_return_table()
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
        # 当衔接新旧数据的时候, 需要丢弃的数据是从开始更新的那一天, 一直到last_day
        sql_dropped_time = "select distinct DataDate from BarraBaseFactor where DataDate >= '" + \
                           str(update_start_date) + "' and DataDate <= '" + str(last_day) + "' and " \
                           "StockPool = '" + self.base_data.stock_pool + "' order by DataDate"
        dropped_time = self.rm_engine.get_original_data(sql_dropped_time).squeeze().tolist()

        # 由于取的原始数据要从更新开始的第一天的前525个交易日取, 因此, 需要得到self.start_date,
        # 即取原始数据的开始日期, 由于barra因子定义中最早使用的是525个交易日前的数据,
        # 因此这个日期是update_start_date的前525个交易日
        sql_start_date = "select top 525 DataDate from (select distinct DataDate from RawData where " \
                         "DataDate <= '" + str(update_start_date) + "' ) temp order by DataDate desc "
        self.start_date = self.rm_engine.get_original_data(sql_start_date).iloc[-1, 0]

        # 更新数据
        self.construct_factor_base()









































