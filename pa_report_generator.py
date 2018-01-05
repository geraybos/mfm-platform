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

from data import data
from strategy_data import strategy_data
from position import position
from barra_base import barra_base
from performance_attribution import performance_attribution
from pa_report_db import pa_report_db

# 用来生成归因报告的类

class pa_report_generator:
    """ The class for generating performance attribution report.

    foo
    """
    def __init__(self):
        # 归因生成器里的归因类, 现在还没有去对他进行初始化
        self.pa = None
        self.db = pa_report_db()

    # 维护更新数据的函数
    def update_base_data(self, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d'))):
        # 首先更新数据库数据, 注意这里不会更新持仓数据
        super(pa_report_db, self.db).update_data_from_db(end_date=end_date)
        # 更新base中的数据
        for i in ['all', 'hs300', 'zz500', 'sz50']:
            base = barra_base(stock_pool=i)
            base.update_factor_base_data()

        print('base data has been updated!\n')

    # 生成策略持仓数据
    def update_tar_holding_data(self, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d'))):
        self.db.update_holdng_tar_data(end_date=end_date)
        print('target holding data has been updated!\n')

    # 生成策略持仓数据, 用ClosePrice得到持仓占比
    def get_tar_holding_position(self):
        self.tar_holding_vol = pd.read_hdf('tar_holding_vol', '123')
        cp = data.read_data(['ClosePrice'], shift=True).iloc[0]
        holding_value = self.tar_holding_vol.mul(cp, axis=0)
        # 归一化
        self.tar_position = self.tar_holding_vol * np.nan
        for strg, holding in holding_value.iteritems():
            self.tar_position[strg] = holding.div(holding.sum(1), axis=0)
        self.tar_position = self.tar_position.fillna(0.0)
        pass

    # 生成策略投资域和benchmark
    def get_strategy_config(self):
        self.strg_pools = {'300alpha套保':'hs300', '300alpha投机':'all', '500alpha套保':'zz500',
            '500alpha投机':'all', '50alpha套保':'sz50', '50alpha投机':'all'}

        self.strg_benchmark = {'300alpha套保':'hs300', '300alpha投机':'hs300', '500alpha套保':'zz500',
            '500alpha投机':'zz500', '50alpha套保':'sz50', '50alpha投机':'sz50'}

    # 准备base
    def prepare_base(self):
        base_all = barra_base(stock_pool='all')
        base_all.base_data.factor_expo = pd.read_hdf('bb_factor_expo_all', '123')
        base_all.base_factor_return = data.read_data(['bb_factor_return_all']).iloc[0]
        base_all.base_data.generate_if_tradable()
        base_all.base_data.handle_stock_pool()

        base_sz50 = barra_base(stock_pool='sz50')
        base_sz50.base_data.factor_expo = pd.read_hdf('bb_factor_expo_sz50', '123')
        base_sz50.base_factor_return = data.read_data(['bb_factor_return_sz50']).iloc[0]
        base_sz50.base_data.generate_if_tradable()
        base_sz50.base_data.handle_stock_pool()

        base_hs300 = barra_base(stock_pool='hs300')
        base_hs300.base_data.factor_expo = pd.read_hdf('bb_factor_expo_hs300', '123')
        base_hs300.base_factor_return = data.read_data(['bb_factor_return_hs300']).iloc[0]
        base_hs300.base_data.generate_if_tradable()
        base_hs300.base_data.handle_stock_pool()

        base_zz500 = barra_base(stock_pool='zz500')
        base_zz500.base_data.factor_expo = pd.read_hdf('bb_factor_expo_zz500', '123')
        base_zz500.base_factor_return = data.read_data(['bb_factor_return_zz500']).iloc[0]
        base_zz500.base_data.generate_if_tradable()
        base_zz500.base_data.handle_stock_pool()

        self.bases = {'all':base_all, 'sz50':base_sz50, 'hs300':base_hs300, 'zz500':base_zz500}

    # 准备benchmark
    def prepare_benchmark(self):
        self.benchmarks = data.read_data(['Weight_sz50', 'Weight_hs300', 'Weight_zz500'])
        # 归一化
        for i, df in self.benchmarks.iteritems():
            self.benchmarks[i] = df.div(df.sum(1), axis=0)
        self.benchmarks = self.benchmarks.fillna(0.0)

    # 计算组合暴露
    def get_strg_tar_expo(self):
        strategy_expo = {}
        for strg, pos in self.tar_position.iteritems():
            curr_pool = self.strg_pools[strg]
            curr_bench_weight= self.benchmarks['Weight_'+self.strg_benchmark[strg]]
            curr_base = self.bases[curr_pool]

            # 计算相对benchmark的超额持仓
            active_position = pos - curr_bench_weight

            # 注意, 由于每天的目标持仓是每天早上跑出来的, 因此此时并没有当天的数据, 数据应当使用昨天的
            # 也就是说, 其实这个东西昨天收盘的时候我们就可以做出来, 但是是今天早上才做,
            # 因此, factor expo和if tradable的数据都需要lag一天, 使用昨天的数据
            lagged_expo = curr_base.base_data.factor_expo.shift(1).reindex(major_axis=pos.index)
            lagged_tradable = curr_base.base_data.if_tradable.shift(1). \
                reindex(major_axis=pos.index).fillna(0).astype(np.bool)

            # 计算组合暴露
            strategy_expo[strg] = strategy_data.get_port_expo(active_position, lagged_expo, lagged_tradable)

        # 将结果转成一个panel
        self.strategy_expo = pd.Panel(strategy_expo)
        self.strategy_expo.to_hdf('strategy_expo', '123')

    # 对暴露进行画图
    def plot_strg_tar_expo(self, date=None, *, folder_name=None):
        if not hasattr(self, 'strategy_expo'):
            self.strategy_expo = pd.read_hdf('strategy_expo', '123')

        # 取要画图的暴露数据
        if isinstance(date, pd.Timestamp):
            plot_data = self.strategy_expo.ix[:, date, :]
        else:
            date = self.strategy_expo.major_axis[-1]
            plot_data = self.strategy_expo.ix[:, date, :]

        # 文件名
        if folder_name is None:
            folder_name = date.strftime('%Y%m%d') + '_expo'
        if not os.path.exists(str(os.path.abspath('.')) + '/' + folder_name):
            os.makedirs(str(os.path.abspath('.')) + '/' + folder_name)
        pdfs = PdfPages(str(os.path.abspath('.')) + '/' + folder_name + '/expo_pdf.pdf')

        # 循环画图
        # make the sort of factors is from top(lncap)  to bottom(leverage)
        factor_order = np.arange(9, -1, -1)
        for strg, curr_expo in plot_data.iteritems():
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            plt.barh(np.arange(10), curr_expo.iloc[factor_order], tick_label=curr_expo.index[factor_order])
            ax.set_xlabel('Factors')
            ax.set_ylabel('Factor Exposures')
            appendix = ''
            if strg[-2:] == '套保':
                strg_appendix = 'hedge'
            elif strg[-2:] == '投机':
                strg_appendix = 'speculate'
            ax.set_title('Factor Exposures of Strategy: ' + strg[:-2] + ' ' + strg_appendix)
            plt.xticks(rotation=30)
            plt.grid()
            ax.legend(loc='best')
            plt.savefig(folder_name + '/' + strg +'.png', dpi=1200)
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')
        pdfs.close()


    def generate_strg_expo_report(self, *, date=None, folder_name=None):
        self.get_tar_holding_position()
        self.get_strategy_config()
        self.prepare_base()
        self.prepare_benchmark()
        self.get_strg_tar_expo()
        self.plot_strg_tar_expo(date=date, folder_name=folder_name)


if __name__ == '__main__':
    import time
    start = time.time()
    pa_generator = pa_report_generator()
    # print("time: {0} seconds\n".format(time.time() - start))
    # pa_generator.update_base_data()
    # print("time: {0} seconds\n".format(time.time() - start))
    pa_generator.update_tar_holding_data()
    print("time: {0} seconds\n".format(time.time() - start))
    pa_generator.generate_strg_expo_report()
    print("time: {0} seconds\n".format(time.time() - start))

    pass















































































