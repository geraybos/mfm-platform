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
import pathos.multiprocessing as mp

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from single_factor_strategy import single_factor_strategy
from backtest import backtest
from dynamic_backtest import dynamic_backtest
from barra_base import barra_base
from optimizer import optimizer
from optimizer_utility import optimizer_utility
from factor_base import factor_base
from multi_factor_strategy import multi_factor_strategy

# 测试优化组合的类
# 根据输入的alpha, 解优化组合, 进行回测, 归因, 然后画图储存.
# 用此来测试风险模型的表现

class opt_portfolio_test(multi_factor_strategy):
    """ Class for optimized portfolio testing. Mainly for risk model testing.

    foo
    """
    def __init__(self):
        multi_factor_strategy.__init__(self)

    # 解优化组合的函数, 由于此策略主要用于测试风险模型的表现, 即测试不同的优化组合的回测表现
    # 这些优化组合是给定要测试的风险模型, 然后变化alpha得来的, 因此, 这里不对换手率等做什么限制
    # 所以每一期的优化没有什么联系, 因此可以用并行来解优化组合
    def construct_optimized_portfolio(self, *, indus_neutral=False):
        global holding_days_g, alpha_g, factor_expo_g, factor_cov_g, spec_var_g, benchmark_g

        holding_days_g = self.holding_days
        alpha_g = self.factor_return
        factor_expo_g = self.strategy_data.factor_expo
        factor_cov_g = self.factor_cov
        spec_var_g = self.spec_var
        benchmark_g = self.strategy_data.benchmark_price.ix['Weight_'+self.strategy_data.stock_pool]

        if indus_neutral:
            global indus_cons

            indus_cons = pd.DataFrame(factor_expo_g.items[10:38], columns=['factor'])
            indus_cons['if_eq'] = True
            indus_cons['if_lower_bound'] = True
            indus_cons['limit'] = 0

        # 定义解单次优化组合的函数
        def one_time_opt_func(cursor):
            curr_time = holding_days_g.iloc[cursor]
            curr_factor_ret = alpha_g.ix[curr_time, :].dropna()
            # 如果换仓当天所有股票都没有因子值, 则返回0序列, 即持有现金
            # 加入这个机制主要是为了防止交易日没有因子值报错的问题
            if curr_factor_ret.isnull().all():
                return pd.Series(0.0, index=alpha_g.columns)
            curr_factor_expo = factor_expo_g.ix[:, curr_time, curr_factor_ret.index].T
            if curr_factor_expo.isnull().all().all():
                return pd.Series(0.0, index=alpha_g.columns)
            curr_factor_cov = factor_cov_g.ix[curr_time]
            curr_spec_var = spec_var_g.ix[curr_time, curr_factor_ret.index]
            curr_bench_weight = benchmark_g.ix[curr_time, curr_factor_ret.index]
            opt = optimizer_utility()

            # 如果当期某个行业所有的股票都是0暴露, 则在行业限制中去除这个行业
            empty_indus = curr_factor_expo[10:38].sum(1)==0
            curr_indus_cons = indus_cons[np.logical_not(empty_indus.values)]
            # curr_indus_cons = None

            # 不添加任何限制的解最大化IR的组合
            optimized_weight = opt.solve_optimization(curr_bench_weight, curr_factor_expo,
                curr_factor_ret, curr_factor_cov, specific_var=curr_spec_var, factor_expo_cons=curr_indus_cons,
                enable_full_inv_cons=False)

            return optimized_weight.reindex(alpha_g.columns)

        ncpus = 20
        p = mp.ProcessPool(ncpus=ncpus)
        p.close()
        p.restart()
        data_size = np.arange(self.holding_days.shape[0])
        chunksize = int(len(data_size) / ncpus)
        results = p.map(one_time_opt_func, data_size, chunksize=chunksize)
        tar_holding = pd.DataFrame({i: v for i, v in zip(self.holding_days.index, results)}).T
        p.close()
        p.join()

        self.position.holding_matrix = tar_holding.fillna(0.0)

    # 解最优化持仓, 进行回测, 然后归因的函数
    def do_opt_portfolio_test(self, *, start_date=None, end_date=None, loc=-1, foldername='',
                              indus_neutral=False):
        # 生成调仓日
        self.generate_holding_days(loc=loc, start_date=start_date, end_date=end_date)

        # 初始化持仓矩阵
        self.initialize_position(self.strategy_data.stock_price.ix[0, self.holding_days, :])

        # 过滤不可交易数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()
        self.factor_return = self.factor_return.where(self.strategy_data.if_tradable['if_inv'], np.nan)

        # 解优化组合
        self.construct_optimized_portfolio(indus_neutral=indus_neutral)

        # 如果没有路径名, 则自己创建一个
        if not os.path.exists(str(os.path.abspath('.')) + '/' + foldername + self.strategy_data.stock_pool +
                                      '/'):
            os.makedirs(str(str(os.path.abspath('.')) + '/' + foldername + self.strategy_data.stock_pool +
                            '/'))
        # 建立画pdf的对象
        self.pdfs = PdfPages(str(os.path.abspath('.')) + '/' + foldername + self.strategy_data.stock_pool +
                             '/allfigs.pdf')

        # 进行回测
        self.bkt_obj = backtest(self.position, bkt_start=start_date, bkt_end=end_date,
                                bkt_benchmark_data='ClosePrice_adj_'+self.strategy_data.stock_pool)
        self.bkt_obj.execute_backtest()
        self.bkt_obj.get_performance(foldername=foldername + self.strategy_data.stock_pool, pdfs=self.pdfs)

        # 做真实情况下的超额归因
        pa_benchmark_weight = data.read_data(['Weight_' + self.strategy_data.stock_pool],
                                             ['Weight_' + self.strategy_data.stock_pool])
        pa_benchmark_weight = pa_benchmark_weight['Weight_' + self.strategy_data.stock_pool]

        self.bkt_obj.get_performance_attribution(benchmark_weight=pa_benchmark_weight, show_warning=False,
            pdfs=self.pdfs, is_real_world=True, foldername=foldername + self.strategy_data.stock_pool,
            real_world_type=2, enable_read_base_expo=True, enable_read_pa_return=True,
            base_stock_pool=self.strategy_data.stock_pool)

        self.pdfs.close()


if __name__ == '__main__':
    rv = pd.read_hdf('stock_alpha_hs300_split', '123')
    for iname, idf in rv.iteritems():

        opt_test = opt_portfolio_test()
        opt_test.strategy_data.stock_pool = 'hs300'

        # 读取数据, 注意数据需要shift一个交易日
        opt_test.factor_cov = pd.read_hdf('bb_riskmodel_covmat_hs300', '123')
        opt_test.strategy_data.factor_expo = pd.read_hdf('bb_factor_expo_hs300', '123')
        opt_test.spec_var = pd.read_hdf('bb_riskmodel_specvar_hs300', '123')

        # opt_test.factor_cov = pd.read_hdf('barra_riskmodel_covmat_all_facret', '123')
        # opt_test.strategy_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
        # opt_test.strategy_data.factor_expo = opt_test.strategy_data.factor_expo[['CNE5S_SIZE', 'CNE5S_BETA', 'CNE5S_MOMENTUM',
        #                                                      'CNE5S_RESVOL', 'CNE5S_SIZENL', 'CNE5S_BTOP',
        #                                                      'CNE5S_LIQUIDTY', 'CNE5S_EARNYILD', 'CNE5S_GROWTH',
        #                                                      'CNE5S_LEVERAGE', 'CNE5S_AERODEF', 'CNE5S_AIRLINE',
        #                                                      'CNE5S_AUTO', 'CNE5S_BANKS', 'CNE5S_BEV',
        #                                                      'CNE5S_BLDPROD', 'CNE5S_CHEM', 'CNE5S_CNSTENG',
        #                                                      'CNE5S_COMSERV', 'CNE5S_CONMAT', 'CNE5S_CONSSERV',
        #                                                      'CNE5S_DVFININS', 'CNE5S_ELECEQP', 'CNE5S_ENERGY',
        #                                                      'CNE5S_FOODPROD', 'CNE5S_HDWRSEMI', 'CNE5S_HEALTH',
        #                                                      'CNE5S_HOUSEDUR', 'CNE5S_INDCONG', 'CNE5S_LEISLUX',
        #                                                      'CNE5S_MACH', 'CNE5S_MARINE', 'CNE5S_MATERIAL',
        #                                                      'CNE5S_MEDIA', 'CNE5S_MTLMIN', 'CNE5S_PERSPRD',
        #                                                      'CNE5S_RDRLTRAN', 'CNE5S_REALEST', 'CNE5S_RETAIL',
        #                                                      'CNE5S_SOFTWARE', 'CNE5S_TRDDIST', 'CNE5S_UTILITIE',
        #                                                      'CNE5S_COUNTRY']]
        # opt_test.strategy_data.factor_expo['CNE5S_COUNTRY'] = opt_test.strategy_data.factor_expo['CNE5S_COUNTRY'].fillna(1)
        # opt_test.strategy_data.factor_expo.ix['CNE5S_AERODEF':'CNE5S_UTILITIE'].fillna(0, inplace=True)
        # opt_test.spec_var = pd.read_hdf('barra_riskmodel_specvar_all_facret', '123')

        # opt_test.factor_cov = pd.read_hdf('barra_fore_cov_mat', '123')
        # opt_test.strategy_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
        # opt_test.spec_var = pd.read_hdf('barra_fore_spec_var_new', '123')

        opt_test.factor_cov = opt_test.factor_cov.shift(1, axis=0).reindex(items=opt_test.factor_cov.items)
        opt_test.strategy_data.factor_expo = opt_test.strategy_data.factor_expo.shift(1).reindex(major_axis=
                                        opt_test.strategy_data.factor_expo.major_axis)
        opt_test.spec_var = opt_test.spec_var.shift(1)

        # opt_test.factor_return = pd.read_hdf('stock_alpha_zelong_hs300', '123')
        opt_test.factor_return = idf
        opt_test.factor_return = opt_test.factor_return.shift(1)
        opt_test.strategy_data.benchmark_price = data.read_data(['Weight_hs300'], shift=True)

        folder_name = 'opt_test/' + iname + '_'
        opt_test.do_opt_portfolio_test(start_date=pd.Timestamp('2011-05-04'), end_date=pd.Timestamp('2017-03-09'),
            loc=-1, foldername=folder_name, indus_neutral=True)



















































































