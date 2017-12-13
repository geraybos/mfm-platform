import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from db_engine import db_engine
from database import database

# 生产系统中维护数据库的类
# 主要区别是, 不是将数据存在本地, 而是存在数据库里

class database_prod(database):
    """ Database class in production system.

    foo
    """
    def __init__(self, *, start_date=None, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d')),
                 market="83"):
        database.__init__(self, start_date=start_date, end_date=end_date, market=market)
        self.initialize_rm()

    # 初始化链接到RiskModel数据库的引擎
    def initialize_rm(self):
        self.rm_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='RiskModel', add_info='')

    # 新建raw data表的函数, 第一次运行时需要建表. 更新数据时不需要新建表, 重新取数据时需要重新建表
    def create_rawdata_table(self):
        sql = "drop table RawData; " \
              "create table RawData ( " \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value decimal(27, 10), " \
              "DataName varchar(255) )"
        self.rm_engine.engine.execute(sql)

    # 新建行业标记表, 由于行业数据是string, 和其他数据类型不一样, 因此需要单独建一个表来储存
    def create_industry_mark_table(self):
        sql = "drop table IndustryMark; " \
              "create table IndustryMark (" \
              "DataDate smalldatetime, " \
              "SecuCode varchar(10), " \
              "Value varchar(100) )"
        self.rm_engine.engine.execute(sql)

    # 新建常量数据的表, 常量数据没有股票标记, 因此只有一个时间序列, 现在用到的常量数据只有无风险利率
    def create_const_data_table(self):
        sql = "drop table ConstData; " \
              "create table ConstData (" \
              "DataDate smalldatetime, " \
              "RiskFreeRate decimal(27, 10) )"
        self.rm_engine.engine.execute(sql)

    # 将dataframe储存到raw data表中的函数
    def save_to_sql(self, item, df):
        # 将标准数据格式转化为sql数据格式
        sql_df = pd.melt(df.reset_index(level=0), id_vars='trading_days', var_name='SecuCode').dropna()
        # 列名命名为数据库中的名字
        sql_df = sql_df.rename(columns={'trading_days': 'DataDate', 'SecuCode': 'SecuCode', 'value': 'Value'})
        sql_df['DataName'] = item

        # 储存到数据库中
        sql_df.to_sql('RawData', self.rm_engine.engine, if_exists='append', index=False)
        # 打印储存成功的提示
        print('Data: {0} has been successfully saved into sql database!\n'.format(item))

    # 将行业数据储存到industry mark表中去
    def save_industry_mark_to_sql(self):
        sql_df = pd.melt(self.data.raw_data['Industry'].reset_index(level=0), id_vars='trading_days',
                         var_name='SecuCode').dropna()
        sql_df = sql_df.rename(columns={'trading_days': 'DataDate', 'SecuCode': 'SecuCode', 'value': 'Value'})
        sql_df.to_sql('IndustryMark', self.rm_engine.engine, if_exists='append', index=False)
        print('Industry mark has been successfully saved into sql database!\n')

    # 将常量数据储存到const data表中去
    # 暂时不会用到这个函数, 因为暂时不会对常量做任何的更改
    def save_const_data_to_sql(self):
        if self.data.const_data.dropna().empty:
            print('const data is empty, thus it will not be written into sql database!\n')
        else:
            sql_df = self.data.const_data.reset_index(level=0).dropna().\
                rename(columns={'trading_days': 'DataDate'})
            sql_df.to_sql('ConstData', self.rm_engine.engine, if_exists='append', index=False)
            print('const data has been successfully saved into sql databse!\n')


    # save data函数是将数据全部储存到数据库中去, 而不是存在本地
    def save_data(self):
        for item, df in self.data.stock_price.iteritems():
            self.save_to_sql(item, df)
        for item, df in self.data.raw_data.iteritems():
            # 如果是行业标记数据, 需要单独操作, 因为需要把它存到另外一张表中
            if item == 'Industry':
                self.save_industry_mark_to_sql()
            else:
                self.save_to_sql(item, df)
        for item, df in self.data.benchmark_price.iteritems():
            self.save_to_sql(item, df)
        for item, df in self.data.if_tradable.iteritems():
            self.save_to_sql(item, df)
        # 常量数据也需要单独储存
        self.save_const_data_to_sql()

    # 取数据的主函数, 这里的主函数, 除了要关心数据储存的问题, 还需要关心新建表和删除表的问题
    def get_data_from_db(self, *, update_time=pd.Timestamp('1900-01-01')):
        # 如果不是在更新, 即重新取所有数据, 则需要把3张相关的表全都删除重新建立一份
        if not self.is_update:
            self.create_rawdata_table()
            self.create_industry_mark_table()
            self.create_const_data_table()
        # 使用database类中的函数来取数据, 注意, 这里的get_data_from_db中的save_data函数
        # 还是会调用db_prod类中的save_data函数, 即, 会把数据存在数据库中, 而不是存在本地
        database.get_data_from_db(self, update_time=update_time)

    # 删除数据库中数据的函数, 用于更新的时候, 需要更新的数据已经在数据库中, 则要把这些数据删除掉
    def delete_from_sql_table(self, *, table_name='RawData', data_name=None, data_date=None):
        sql_delete = 'delete from ' + table_name + ' '
        # 将list的参数改写为sql中的形式
        if data_name is not None:
            sql_data_name = "DataName in ("
            for name in data_name:
                sql_data_name += "'" + str(name) + "', "
            # 去掉最后一个空格和逗号
            sql_data_name = sql_data_name[:-2]
            sql_data_name += ") "
        if data_date is not None:
            sql_data_date = "DataDate in ("
            for date in data_date:
                sql_data_date += "'" + str(date) + "', "
            sql_data_date = sql_data_date[:-2]
            sql_data_date += ") "

        if data_name is not None and data_date is None:
            sql_delete = sql_delete + "where " + sql_data_name
        elif data_name is None and data_date is not None:
            sql_delete = sql_delete + "where " + sql_data_date
        elif data_name is not None and data_date is not None:
            sql_delete = sql_delete + "where " + sql_data_name + 'and ' + sql_data_date

        self.rm_engine.engine.execute(sql_delete)

    # 更新数据的函数
    # 生产上的更新数据与本地更新数据有很大不同, 是从数据库里取数据, 然后将新数据写到数据库里
    def update_data_from_db(self, *, start_date=None, end_date=None):
        # 首先更新标记
        self.is_update = True
        # 取数据库中最新一天的日期, 以RawData数据库为准
        last_day = self.rm_engine.get_original_data("select max(DataDate) from RawData").iloc[0, 0]
        # 如果有传入的指定更新数据的开始时间, 则选取last_day和指定开始时间更早的那天
        if isinstance(start_date, pd.Timestamp):
            self.start_date = min(start_date, last_day)
        else:
            self.start_date = last_day
        # 当衔接新旧数据的时候, 需要丢弃的数据是从开始更新的那一天, 一直到last_day
        sql_dropped_time = "select distinct DataDate from RawData where DataDate >= '" + \
            str(self.start_date) + "' and DataDate <= '" + str(last_day) + "' order by DataDate"
        dropped_time = self.rm_engine.get_original_data(sql_dropped_time).squeeze().tolist()
        # 可以设置更新数据的更新截止日，默认为更新到当天
        if isinstance(end_date, pd.Timestamp):
            self.end_date = end_date
            assert self.end_date >= self.start_date, 'Please make sure that end date is not earlier than ' \
                                                     'start date!\n'
        # 删除需要删除的数据
        self.delete_from_sql_table(table_name='RawData', data_date=dropped_time)
        self.delete_from_sql_table(table_name='IndustryMark', data_date=dropped_time)
        self.delete_from_sql_table(table_name='ConstData', data_date=dropped_time)

        # 更新数据, 注意这里更新数据时不会删除原来的表并新建表, 而且暂时不会储存数据
        self.get_data_from_db(update_time=self.start_date)

        # 需要取旧数据, 这里和研究中不一样, 不能把所有旧数据都读进来, 只能读一部分旧数据,
        # 因此旧数据读取的时间选为更新开始时间的前一天, 即只用一天的旧数据, 这样还是能够完成新旧数据的衔接
        # 但是副作用是, 对于更新期内的新上市股票, 其此前的数据会在数据库里没有条目, 因此之后用到这些数据的时候, 需要注意
        # 尤其是if tradable中的标记数据, 这些数据对于没有上市的股票, 本来也会有数据(会是0), 但是现在没有了,
        # 因此在用到这些数据的时候要注意到nan的调整. 另外指数权重数据也是, 本来没上市的股票也会是0,
        # 但现在也会没有, 取出来之后就是nan, 因此同样要注意nan
        sql_old_data_time = "select max(DataDate) from RawData where DataDate < '" + str(self.start_date) + "' "
        old_data_time = self.rm_engine.get_original_data(sql_old_data_time).iloc[0, 0]

        # 读取老数据
        sql_old_data_RawData = "select * from RawData where DataDate = '" + str(old_data_time) + "' "
        sql_old_data_IndustryMark = "select * from IndustryMark where DataDate = '" + str(old_data_time) + "' "
        sql_old_data_ConstData = "select * from ConstData where DataDate = '" + str(old_data_time) + "' "
        old_RawData = self.rm_engine.get_original_data(sql_old_data_RawData)
        old_IndustryMark = self.rm_engine.get_original_data(sql_old_data_IndustryMark)
        old_ConstData = self.rm_engine.get_original_data(sql_old_data_ConstData)
        old_RawData = old_RawData.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode', values='Value',
            aggfunc='first').to_panel().transpose(2, 1, 0)
        old_IndustryMark = old_IndustryMark.pivot_table(index=['DataDate'], columns='SecuCode', values='Value',
            aggfunc='first')
        old_IndustryMark = pd.Panel({'Industry': old_IndustryMark})
        old_ConstData = old_ConstData.set_index('DataDate')
        old_data = pd.concat([old_RawData, old_IndustryMark], axis=0)
        # 将self.data中的几个部分也都拼成一个大的panel, 这样比把读进来的老数据的大panel拆分开要方便
        new_data = pd.concat([self.data.stock_price, self.data.raw_data, self.data.benchmark_price,
                              self.data.if_tradable], axis=0)
        # 其实不需要reindex, 因为concat的时候实际上会自动reindex
        whole_data = pd.concat([old_data, new_data], axis=1)
        whole_const_data = pd.concat([old_ConstData, self.data.const_data], axis=0)

        # 对需要的数据进行fillna，以及fillna后的重新计算
        # 主要是那些用到first_date参数的数据，以及涉及这些数据的衍生数据
        # 这里的填充就直接在whole_data里填充就行
        whole_data['TotalAssets'] = whole_data['TotalAssets'].fillna(method='ffill')
        whole_data['TotalLiability'] = whole_data['TotalLiability'].fillna(method='ffill')
        whole_data['TotalEquity'] = whole_data['TotalEquity'].fillna(method='ffill')
        whole_data['PB'] = whole_data['MarketValue']/whole_data['TotalEquity']
        # 现在上市退市标记和指数权重数据只进行向前填充, 不再进行将所有nan填成0的步骤
        whole_data['is_enlisted'] = whole_data['is_enlisted'].fillna(method='ffill')
        whole_data['is_delisted'] = whole_data['is_delisted'].fillna(method='ffill')
        for item, df in whole_data.iteritems():
            if item.startswith('Weight_'):
                whole_data[item] = df.fillna(method='ffill')
        # 复权因子在更新的时候, 需要在衔接了停牌标记后, 在此进行复权因子(以及之后的后复权价格)的计算
        # 同直接取所有的复权因子数据时一样, 首先将停牌期间的复权因子设置为nan,
        # 然后使用停牌前最后一天的复权因子向前填充, 使得停牌期间的复权因子变化反映在复牌后第一天,
        # 不需要再将数据填成1, 在算adj price的时候进行fill value就可以了
        # is_suspened是nan的股票, 都是那些根本没上市或者退市的股票, 因此如何改都没有影响
        whole_data['AdjustFactor'] = whole_data['AdjustFactor'].where(np.logical_not(
            whole_data['is_suspended']), np.nan).fillna(method='ffill')
        # 因为衔接并向前填充了复权因子, 因此要重新计算后复权价格, 否则之前的后复权价格将是nan
        ochl = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'vwap']
        for data_name in ochl:
            whole_data[data_name + '_adj'] = whole_data[data_name].mul(whole_data['AdjustFactor'].fillna(1))

        # 将数据储存到数据库去, 这里因为都在一个大数据whole data中, 因此无法使用save data函数
        for item, df in whole_data.iteritems():
            if item == 'Industry':
                sql_df = pd.melt(df.reset_index(level=0), id_vars='trading_days', var_name='SecuCode').dropna()
                sql_df = sql_df.rename(columns={'trading_days': 'DataDate', 'SecuCode': 'SecuCode', 'value': 'Value'})
                sql_df.to_sql('IndustryMark', self.rm_engine.engine, if_exists='append', index=False)
                print('Industry mark has been successfully saved into sql database!\n')
            else:
                self.save_to_sql(item, df)
        if whole_const_data.dropna().empty:
            print('const data is empty, thus it will not be written into sql database!\n')
        else:
            sql_df = whole_const_data.reset_index(level=0).dropna().\
                rename(columns={'trading_days': 'DataDate'})
            sql_df.to_sql('ConstData', self.rm_engine.engine, if_exists='append', index=False)
            print('const data has been successfully saved into sql databse!\n')




if __name__ == '__main__':
    import time
    start_time = time.time()
    dbp = database_prod(start_date=pd.Timestamp('2007-01-01'))
    dbp.get_data_from_db()
    # dbp.update_data_from_db(start_date=pd.Timestamp('2017-12-01'))
    # dbp.create_rawdata_table()
    # dbp.get_trading_days()
    # dbp.get_labels()
    # dbp.get_asset_liability_equity()
    # df = pd.DataFrame(index=dbp.trading_days)
    # dbp.save_to_sql('test', df)
    print('Time: {0}\n'.format(time.time() - start_time))