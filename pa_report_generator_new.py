import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages

from data import data
from strategy_data import strategy_data
from barra_base import barra_base
from pa_report_db import pa_report_db
from db_engine import db_engine

# 生成归因报告的类, 新版本, 为从数据库读取数据, 而不在本地读写数据

class pa_report_generator_new:
    """ The class for generating performance attribution report.

    foo
    """
    def __init__(self):
        # 归因生成器里的归因类, 现在还没有去对他进行初始化
        self.pa = None
        self.pa_db = pa_report_db()

    # 更新策略持仓量数据
    def update_tar_holding_data(self, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d'))):
        self.pa_db.update_holdng_tar_data(end_date=end_date)
        print('target holding data has been updated!\n')

    # 生成策略投资域和benchmark
    def get_strategy_config(self):
        self.strg_pools = {'300alpha套保':'hs300', '300alpha投机':'all', '500alpha套保':'zz500',
            '500alpha投机':'all', '50alpha套保':'sz50', '50alpha投机':'all'}

        self.strg_benchmark = {'300alpha套保':'hs300', '300alpha投机':'hs300', '500alpha套保':'zz500',
            '500alpha投机':'zz500', '50alpha套保':'sz50', '50alpha投机':'sz50'}

    # 从数据库构建factor expo和factor return的函数
    def get_expo_return_data(self, stock_pool):
        base = barra_base(stock_pool=stock_pool)


    # 准备base
    def prepare_base(self):
        pass


if __name__ == '__main__':
    pa_report_new = pa_report_generator_new()






































