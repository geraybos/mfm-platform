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
        self.factor_cov = None,
        self.cov_risk_aversion = None
        self.spec_risk_aversion = None

    # 解优化组合的函数, 由于此策略主要用于测试风险模型的表现, 即测试不同的优化组合的回测表现
    # 这些优化组合是给定要测试的风险模型, 然后变化alpha得来的, 因此, 这里不对换手率等做什么限制
    # 所以每一期的优化没有什么联系, 因此可以用并行来解优化组合
    def construct_optimized_portfolio(self, *, indus_neutral=False):
        global holding_days_g, alpha_g, factor_expo_g, factor_cov_g, spec_var_g, benchmark_g, \
               cov_risk_aversion_g, spec_risk_aversion_g

        holding_days_g = self.holding_days
        alpha_g = self.factor_return
        factor_expo_g = self.strategy_data.factor_expo
        factor_cov_g = self.factor_cov
        spec_var_g = self.spec_var
        benchmark_g = self.strategy_data.benchmark_price.iloc[0]

        cov_risk_aversion_g = self.cov_risk_aversion
        spec_risk_aversion_g = self.spec_risk_aversion

        if indus_neutral:
            global indus_cons

            indus_name = factor_expo_g.items[10:38]
            indus_cons = pd.DataFrame(indus_name.values, columns=['factor'])
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
            opt.set_risk_aversion(cov_risk_aversion=cov_risk_aversion_g,
                                  spec_risk_aversion=spec_risk_aversion_g)

            # 如果当期某个行业所有的股票都是0暴露, 则在行业限制中去除这个行业
            if indus_neutral:
                empty_indus = curr_factor_expo[10:38].fillna(method='bfill').sum(1)==0
                curr_indus_cons = indus_cons[np.logical_not(empty_indus.values)]
                enable_full_inv_cons = False
            else:
                curr_indus_cons = None
                enable_full_inv_cons = True
            # curr_indus_cons = None

            # 不添加任何限制的解最大化IR的组合
            opt.solve_optimization(curr_bench_weight, curr_factor_expo, curr_factor_cov,
                residual_return=curr_factor_ret, specific_var=curr_spec_var,
                factor_expo_cons=curr_indus_cons, enable_full_inv_cons=enable_full_inv_cons,
                asset_cap=None)

            return (opt.optimized_weight.reindex(alpha_g.columns), opt.forecasted_vol)

        ncpus = 20
        p = mp.ProcessPool(ncpus=ncpus)
        p.close()
        p.restart()
        data_size = np.arange(self.holding_days.shape[0])
        chunksize = int(len(data_size) / ncpus)
        results = p.map(one_time_opt_func, data_size, chunksize=chunksize)
        tar_holding = pd.DataFrame({i: v[0] for i, v in zip(self.holding_days.index, results)}).T
        forecasted_vol = pd.Series({i: v[1] for i, v in zip(self.holding_days.index, results)})
        p.close()
        p.join()

        self.position.holding_matrix = tar_holding.fillna(0.0)
        self.forecasted_vol = forecasted_vol

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

        # # 投资域为全市场时的tricky解法, 只保留有alpha, 且在benchmark中的股票
        # self.factor_return = self.factor_return.mask(np.logical_and(self.strategy_data.benchmark_price.iloc[0]>0,
        #         self.factor_return.isnull()), 0)

        # 解优化组合
        self.construct_optimized_portfolio(indus_neutral=indus_neutral)
        # 将优化组合中小于万分之一的持仓扔掉
        self.position.holding_matrix = self.position.holding_matrix.where(
            self.position.holding_matrix >= 1e-4, 0)

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

        # 根据策略的超额净值, 可以计算策略的实现的超额收益波动率, 对比实现波动率和预测波动率, 可以看出风险预测能力
        # 由于是周5换仓, 因此实现的区间为周5到下周4, 标签选择周5, 因此选择的resample区间为周五开始, 周五结束
        # 闭区间选择为left, 即这周五, 标签也选择为left, 即这周五, 这样就能和预测风险的标签相吻合
        active_nav_ret_std = self.bkt_obj.bkt_performance.active_nav.pct_change(). \
            resample('w-fri', closed='left', label='left').std().mul(np.sqrt(252))
        # 统计量为实现波动率除以预测波动率
        self.risk_pred_ratio = active_nav_ret_std / self.forecasted_vol
        # 实现波动率储存起来以供参考
        self.realized_vol = active_nav_ret_std
        # 打印统计量的均值和方差
        str_risk = 'Risk forecast accuracy ratio, mean: {0}, std: {1}\n'. \
            format(self.risk_pred_ratio.mean(), self.risk_pred_ratio.std())
        print(str_risk)
        # 根据优化组合算每期组合的预期alpha
        self.forecasted_alpha = self.position.holding_matrix.mul(self.factor_return.reindex(index=
            self.position.holding_matrix.index)).sum(1)
        # 打印组合的预期alpha均值和方差
        str_alpha = 'Alpha forecast, mean: {0}, std: {1}\n'. \
            format(self.forecasted_alpha.mean(), self.forecasted_alpha.std())
        print(str_alpha)

        # 将风险预测统计量, 预期alpha, 以及风险厌恶系数等量写入txt文件
        target_str = str_risk + str_alpha
        target_str += 'Cov Mat risk aversion: {0}, Spec Var risk aversion: {1}\n'. \
            format(self.cov_risk_aversion, self.spec_risk_aversion)
        with open(str(os.path.abspath('.'))+'/'+foldername+self.strategy_data.stock_pool+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)

        # 做真实情况下的超额归因
        pa_benchmark_weight = data.read_data('Weight_' + self.strategy_data.stock_pool).fillna(0.0)
        # pa_benchmark_weight = data.read_data(['Weight_hs300'],
        #                                      ['Weight_hs300'])
        # pa_benchmark_weight = pa_benchmark_weight['Weight_hs300']

        self.bkt_obj.get_performance_attribution(benchmark_weight=pa_benchmark_weight, show_warning=False,
            pdfs=self.pdfs, is_real_world=True, foldername=foldername + self.strategy_data.stock_pool,
            real_world_type=2, enable_read_pa_return=True, base_stock_pool=self.strategy_data.stock_pool)

        self.pdfs.close()


if __name__ == '__main__':

    for spec_ra in np.arange(1, 10.5, 0.5):
        opt_test = opt_portfolio_test()
        opt_test.strategy_data.stock_pool = 'hs300'

        # 读取数据, 注意数据需要shift一个交易日
        risk_model_version = 'all'
        opt_test.factor_cov = data.read_data('bb_riskmodel_covmat_' + risk_model_version)
        opt_test.strategy_data.factor_expo = data.read_data('bb_factor_expo_' + risk_model_version)
        opt_test.spec_var = data.read_data('bb_riskmodel_specvar_' + risk_model_version)

        # opt_test.factor_cov = pd.read_hdf('barra_fore_cov_mat', '123') * 12
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
        # opt_test.factor_cov = opt_test.factor_cov.reindex(major_axis=opt_test.strategy_data.factor_expo.items,
        #                                                   minor_axis=opt_test.strategy_data.factor_expo.items)
        # opt_test.spec_var = pd.read_hdf('barra_fore_spec_var_new', '123') * 12


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


        opt_test.factor_cov = opt_test.factor_cov.shift(1, axis=0).reindex(items=opt_test.factor_cov.items)
        opt_test.strategy_data.factor_expo = opt_test.strategy_data.factor_expo.shift(1).reindex(major_axis=
                                        opt_test.strategy_data.factor_expo.major_axis)
        opt_test.spec_var = opt_test.spec_var.shift(1)

        opt_test.factor_return = data.read_data('runner_value_63')
        # opt_test.factor_return = opt_test.factor_return.div(20)
        opt_test.factor_return = opt_test.factor_return.div(opt_test.factor_return.std(1), axis=0)
        opt_test.factor_return = opt_test.factor_return.shift(1)
        opt_test.strategy_data.benchmark_price = data.read_data(['Weight_hs300'],
                                                                shift=True).fillna(0.0)

        # 设置风险厌恶系数
        opt_test.cov_risk_aversion = 0.75
        opt_test.spec_risk_aversion = spec_ra

        folder_name = 'tar_holding_bkt/spec_ra_tuning/spec_ra_' + str(spec_ra) + '_'
        opt_test.do_opt_portfolio_test(start_date=pd.Timestamp('2016-01-04'), end_date=pd.Timestamp('2018-01-16'),
            loc=-1, foldername=folder_name, indus_neutral=True)



















































































