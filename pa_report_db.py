import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from db_engine import db_engine
from database import database

# 对生成归因报告的数据进行维护的类

class pa_report_db(database):
    """ This is the class of handling pa report data, including fetch and update data from database.

    foo
    """
    def __init__(self, *, start_date=None, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d')),
                 market="83"):
        database.__init__(self, start_date=start_date, end_date=end_date, market=market)
        # trading数据库的引擎
        self.trading_engine = None

    # 初始化trading数据库
    def initialize_trading(self):
        self.trading_engine = db_engine(server_type='mssql', driver='pymssql', username='trading.user',
            password='vzxc2w563#', server_ip='192.168.66.12', port='1433', db_name='Trading', add_info='')

    # 取trading库中的sample表, 即策略的目标持仓
    def get_tar_holding_vol(self):
        sql_query = "select * from sample where Dt>='" + str(self.trading_days.iloc[0]) + "' and Dt<='" + \
            str(self.trading_days.iloc[-1]) + "' and Strg in ('300alpha套保', '300alpha投机', '500alpha套保', " \
            "'500alpha投机', '50alpha套保', '50alpha投机')"
        tar_holding_vol_ori = self.trading_engine.get_original_data(sql_query)
        tar_holding_vol_ori['Dt'] = pd.to_datetime(tar_holding_vol_ori['Dt'])
        tar_holding_vol_table = tar_holding_vol_ori.pivot_table(index=['Dt', 'Strg'], columns='SecuCode',
                                                            values='PosNum')
        tar_holding_vol = tar_holding_vol_table.to_panel().transpose(2, 1, 0).fillna(0)
        # 将持仓时间重索引为交易日序列, 持仓每日都有, 不需要进行填充
        # 持仓的股票重索引为A股股票序列, 即暂时排除了港股和期货持仓
        self.tar_holding_vol = tar_holding_vol.reindex(major_axis=self.trading_days, minor_axis=
            self.data.stock_price.minor_axis).fillna(0)

        pass

    def get_data_from_db(self, *, update_time=pd.Timestamp('1900-01-01')):
        database.get_data_from_db(self, update_time=update_time)
        self.initialize_trading()
        # 取sample表中的持仓数据
        self.get_tar_holding_vol()
        # 更新数据的情况下不能储存数据
        if not self.is_update:
            print('get target holding from trading.sample has been completed...\n')
            self.tar_holding_vol.to_hdf('tar_holding_vol', '123')

    def update_data_from_db(self, *, end_date=None):
        database.update_data_from_db(self, end_date=end_date)
        # 为了保持一致性, 这里再次把更新标记设为True
        self.is_update = True
        # 读取旧的目标持仓数据
        old_tar_holding_vol = pd.read_hdf('tar_holding_vol', '123')

        self.initialize_trading()
        # 计算更新时间段的目标持仓数据
        self.get_tar_holding_vol()
        # 将旧数据中的股票数据重索引成新数据中的股票数据
        old_tar_holding_vol = old_tar_holding_vol.reindex(minor_axis=self.tar_holding_vol.minor_axis, fill_value=0)
        # 将新旧数据衔接
        new_tar_holding_vol = pd.concat([old_tar_holding_vol.drop(self.start_date, axis=1).sort_index(),
                                         self.tar_holding_vol.sort_index()], axis=1)
        # 储存新数据
        new_tar_holding_vol.to_hdf('tar_holding_vol', '123')
        # 重置标记
        self.is_update = False

    def update_holdng_tar_data(self, *, end_date=None):
        self.is_update = True
        # 读取旧的目标持仓数据
        old_tar_holding_vol = pd.read_hdf('tar_holding_vol', '123')

        # 寻找最后一天
        last_day = old_tar_holding_vol.major_axis[-1]
        self.start_date = last_day
        if isinstance(end_date, pd.Timestamp):
            self.end_date = end_date

        # 计算更新时间段的目标持仓数据
        self.get_tar_holding_vol()
        # 将旧数据中的股票数据重索引成新数据中的股票数据
        old_tar_holding_vol = old_tar_holding_vol.reindex(minor_axis=self.tar_holding_vol.minor_axis, fill_value=0)
        # 将新旧数据衔接
        new_tar_holding_vol = pd.concat([old_tar_holding_vol.drop(self.start_date, axis=1).sort_index(),
                                         self.tar_holding_vol.sort_index()], axis=1)
        # 储存新数据
        new_tar_holding_vol.to_hdf('tar_holding_vol', '123')
        # 重置标记
        self.is_update = False


if __name__ == '__main__':
    pa_db = pa_report_db(start_date=pd.Timestamp('2007-01-04'), end_date=pd.Timestamp('2017-11-14'))
    pa_db.initialize_jydb()
    pa_db.initialize_sq()
    pa_db.initialize_gg()
    pa_db.initialize_trading()
    pa_db.get_trading_days()
    pa_db.get_labels()
    pa_db.get_tar_holding_vol()
    pass