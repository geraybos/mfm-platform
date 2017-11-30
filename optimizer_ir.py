import numpy as np
import pandas as pd
import matplotlib
import os
from scipy import optimize
import functools

from strategy_data import strategy_data
from position import position
from strategy import strategy
from optimizer import optimizer


# 优化器类
class optimizer_ir(optimizer):
    def __init__(self):
        optimizer.__init__(self)

    # 最大化IR的目标函数
    def objective(self, w, add_params):
        # 为了能传参数, 其余的参数按照字典的形式传入
        bench_weight = add_params['bench_weight']
        factor_expo = add_params['factor_expo']
        factor_ret = add_params['factor_ret']
        factor_cov = add_params['factor_cov']
        # 如果不传入手续费函数, 则只考虑扣费前收益
        if 'trans_cost_func' in add_params:
            trans_cost_func = add_params['trans_cost_func']
        else:
            trans_cost_func = lambda x: 0
        # 另外可以加入预测的每支股票的specific risk或者residual return, 如果没有, 则默认为0
        # 这时只考虑common factor对股票的风险收益的影响
        if 'specific_var' in add_params:
            specific_var = add_params['specific_var']
        else:
            specific_var = np.zeros(w.shape)
        if 'residual_return' in add_params:
            residual_return = add_params['residual_return']
        else:
            residual_return = np.zeros(w.shape)

        # 计算组合相对基准的超额持仓
        active_w = w - bench_weight
        # # 计算active return的部分
        # # 首先是common factor的收益
        # active_ret = factor_ret.dot(factor_expo.dot(active_w))
        # # 如果有预测的residual return, 还需要加上这个部分
        # active_ret += residual_return.dot(active_w)

        # 为了验证barra, 先改成直接输入股票的return
        active_ret = factor_ret.dot(active_w)

        # active sigma的部分
        # 首先是common factor部分的风险
        active_var = factor_expo.dot(active_w).T.dot(factor_cov).dot(factor_expo.dot(active_w))
        # 如果有预测的specific risk, 需要加上这个部分
        active_var += active_w.dot(np.multiply(specific_var, active_w))

        # 计算IR
        # 默认的手续费函数是0, 因此如果有手续费函数, 还需要将收益减去手续费函数这一部分
        ir = (active_ret - trans_cost_func(w)) / np.sqrt(active_var)

        # 因为目标函数的形式是最小化, 因此要取负号
        return - ir


if __name__ == '__main__':
    from barra_base import barra_base
    from data import data

    opt = optimizer_ir()

    # universe = pd.read_csv('universe.csv')
    # AssetData = pd.read_csv('CNE5S_100_Asset_Data.20170309')
    # AssetExpo = pd.read_csv('CNE5S_100_Asset_Exposure.20170309')
    # Covariance = pd.read_csv('CNE5S_100_Covariance.20170309')
    #
    # # 取股票的收益
    # stock_return = universe.pivot_table(index='Barrid', values='Signal').div(100)
    # # stock_return = stock_return[0:50]
    # # 取残余收益
    # spec_risk = AssetData.pivot_table(index='!Barrid', values='SpecRisk%').reindex(stock_return.index)
    # spec_var = (spec_risk/100)**2
    # # 取因子暴露
    # factor_expo = AssetExpo.pivot_table(index='!Barrid', columns='Factor', values='Exposure').\
    #     reindex(stock_return.index).fillna(0.0).T
    # # 取因子协方差矩阵
    # factor_cov1 = Covariance.pivot_table(index='!Factor1', columns='Factor2', values='VarCovar')
    # factor_cov2 = Covariance.pivot_table(index='Factor2', columns='!Factor1', values='VarCovar')
    # factor_cov = factor_cov1.where(factor_cov1.notnull(), factor_cov2).div(10000)

    bb = barra_base()
    # bb.base_data.stock_pool = 'hs300'
    # bb.construct_factor_base()
    # bb.base_data.factor_expo.to_hdf('bb_factorexpo_hs300', '123')

    factor_cov = pd.read_hdf('bb_factor_eigencovmat_hs300_sf3', '123')
    bb.base_data.factor_expo = pd.read_hdf('bb_factorexpo_hs300', '123')
    spec_vol = pd.read_hdf('bb_factor_vraspecvol_hs300', '123')
    spec_var = spec_vol ** 2
    stock_return = pd.read_hdf('stock_alpha_hs300', '123')

    bench = data.read_data(['Weight_hs300'])
    bench = bench['Weight_hs300']

    # factor_cov = pd.read_hdf('barra_fore_cov_mat', '123')
    # bb.base_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
    # spec_var = pd.read_hdf('barra_fore_spec_var_new', '123') * 0
    # stock_return = pd.read_hdf('barra_real_spec_ret_new', '123')

    spec_var = spec_var.where(bench>0, np.nan)
    stock_return = stock_return.where(bench>0, np.nan)
    bb.base_data.factor_expo = bb.base_data.factor_expo.apply(lambda x: x.where(bench>0, np.nan), axis=(1,2))

    curr_time = '20110509'
    stock_return = stock_return.ix[curr_time, :].dropna()
    # stock_return = stock_return.iloc[0:300]
    factor_expo = bb.base_data.factor_expo.ix[:, curr_time, stock_return.index].T
    factor_cov = factor_cov.ix[curr_time]
    spec_var = spec_var.ix[curr_time, stock_return.index]

    # benchmark先假设没有
    # bench_weight = pd.Series(np.zeros(stock_return.shape), index=stock_return.index)
    bench_weight = bench.ix[curr_time, stock_return.index]

    # 测试transaction cost与turnover constraints时用到的上一期的持仓, 设置为等权
    # old_w = pd.Series(1/stock_return.shape[0], index=stock_return.index)
    old_w = pd.Series(0.0, index=stock_return.index)

    optimized_weight = opt.solve_optimization(bench_weight, factor_expo, stock_return, factor_cov,
                                                 specific_var=spec_var, enable_turnover_cons=False,
                                                 enable_trans_cost=False, long_only=True, old_w=old_w,
                                                 turnover_cap=0.3, enable_factor_expo_cons=False,
                                                 asset_cap=None)
    pass























































































































































