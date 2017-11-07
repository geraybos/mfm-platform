import numpy as np
import pandas as pd
import matplotlib
import os
from scipy import optimize
import functools

from strategy_data import strategy_data
from position import position
from strategy import strategy

# 优化器类
class optimizer(object):
    def __init__(self):
        pass

    # 最大化IR的目标函数
    def ir_objective(self, w, add_params):
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

    # 最大化IR的优化函数
    def ir_optimizer(self, obj_func_params, *, asset_bounds=(None, None,), eq_cons_funcs=None,
                     ineq_cons_funcs=None, n_split=None):
        # 去因子暴露矩阵来做参照物, 以得到资产数和因子数
        n_factors = obj_func_params['factor_expo'].shape[0]
        n_varaibles = obj_func_params['factor_expo'].shape[1]
        # 初始的w点为0
        # w = np.zeros(n_assets)
        if isinstance(n_split, int):
            n_assets = int(n_split / 2)
            w = np.concatenate((np.full(n_assets, 1/n_assets), np.zeros(2*n_assets)), axis=0)
        else:
            n_assets = n_varaibles
            w = np.full(n_assets, 1/n_assets)
            # w = np.random.uniform(0, 1, n_assets)
            # w = w/np.sum(w)

        # 设置限制条件
        # 股票的边界限制条件, 如股票不能做空时, 必须大于0, 或者设置每支股票权重不能超过多少的限制
        bounds = [asset_bounds for i in range(n_assets)]
        # 注意, 当有split variable时, 需要将split variable重新设置
        if n_assets != n_varaibles:
            for i in range(2*n_assets):
                bounds.append((0, None, ))

        # 等式的限制条件
        cons = []
        if bool(eq_cons_funcs):
            for key, value in eq_cons_funcs.items():
                cons.append({'type': 'eq',
                            'fun': value})
        # 不等式的限制条件
        if bool(ineq_cons_funcs):
            for key, value in ineq_cons_funcs.items():
                cons.append({'type': 'ineq',
                             'fun': value})

        import time
        start_time = time.time()
        opt_results = optimize.minimize(self.ir_objective, w, args=(obj_func_params, ), bounds=bounds,
                                        constraints=cons, method='SLSQP',
                                        options={'disp':True, 'maxiter':1000})
        print("time: {0} seconds\n".format(time.time() - start_time))
        return opt_results

    # 计算手续费的函数
    @staticmethod
    def trans_cost(w, *, old_w, buy_cost=1.5/1000, sell_cost=1.5/1000):
        # 买入的手续费
        trans_cost_buy = np.sum(np.maximum(w - old_w, 0)) * buy_cost
        trans_cost_sell = np.sum(np.maximum(old_w - w, 0)) * sell_cost
        total_trans_cost = trans_cost_buy + trans_cost_sell
        return total_trans_cost

    # # 添加换手率限制条件的函数,
    # @staticmethod
    # def turnover_cons(w, *, old_w, turnover_cap=1.0):
    #     # turnover = np.sum(np.abs(w - old_w))/2
    #     # turnover = np.sum(np.where(w>old_w, w-old_w, 0))
    #     turnover = np.sum((w-old_w)**2)
    #     # 根据设置的限制条件, 减去设置的上限
    #     turnover_cons = turnover - turnover_cap
    #     # 因为scipy.optimize.minimize函数的条件的形式是大于0, 因此要取负号
    #     # 暂时不支持现金, 等添加了现金功能之后再来修改
    #     return - turnover_cons

    # 添加换手率限制条件的一系列函数
    # 首先是split plus + split minus <= 换手率的限制条件
    @staticmethod
    def turnover_cons_ineq(w, *, n_assets, turnover_cap=1.0):
        turnover_cons = np.sum(w[n_assets:]) / 2 - turnover_cap
        # 因为scipy.optimize.minimize函数的条件的形式是大于0, 因此要取负号
        return - turnover_cons

    # 然后是w(i) = SplitPlus(i) - SplitMinus(i)的等式条件
    # 注意这个等式条件返回的是一个为n_asset的向量, 代表有个n_asset个限制条件
    @staticmethod
    def turnover_cons_eq(w, *, old_w, n_assets):
        equality = w[0:n_assets] - old_w[0:n_assets] - w[n_assets:(2*n_assets)] + w[(2*n_assets):]
        return equality

    # 添加因子暴露值的限制条件
    @staticmethod
    def factor_expo_cons(w, *, factor, lower_bound=True, limit=0):
        # 根据选择的因子暴露和限制, 减去限制
        factor_expo = factor.dot(w) - limit
        # 因为函数的条件形式是大于0, 因此如果是下限, 则不管, 上限则取负号
        # 如果是希望设置等式条件, 则取不取负号都是一样的
        if not lower_bound:
            factor_expo = -factor_expo
        return factor_expo

    # 设置持仓之和的条件函数, 在不支持现金资产的时候, 总是限制持仓之和为1
    @staticmethod
    def full_inv_cons(w, *, n_assets, cash_ratio=0):
        total_weight = np.sum(w[:n_assets])
        # 减去设置的限制
        full_inv_cons = total_weight - (1 - cash_ratio)
        return full_inv_cons

    # 建立最大化IR问题的函数, 这个函数可以作为和外界的接口来使用
    # 注意, 这里输入的变量是pd.DataFrame或者pd.Series, 但是要将其做成np.ndarray的类型
    def solve_ir_optimization(self, bench_weight, factor_expo, factor_ret, factor_cov, *,
                              specific_var=None, residual_return=None, old_w=None, enable_trans_cost=True,
                              buy_cost=1.5/1000, sell_cost=1.5/1000, enable_turnover_cons=True,
                              turnover_cap=1.0, enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                              asset_cap=None, enable_factor_expo_cons=False):

        if enable_turnover_cons:
            optimized_weight = self.solve_ir_opt_with_turnover_cons(bench_weight, factor_expo, factor_ret, factor_cov,
                specific_var=specific_var, residual_return=residual_return, old_w=old_w,
                enable_trans_cost=enable_trans_cost, buy_cost=buy_cost, sell_cost=sell_cost,
                turnover_cap=turnover_cap, enable_full_inv_cons=enable_full_inv_cons,
                cash_ratio=cash_ratio, long_only=long_only, asset_cap=asset_cap,
                enable_factor_expo_cons=enable_factor_expo_cons)
        else:
            optimized_weight = self.solve_ordinary_ir_opt(bench_weight, factor_expo, factor_ret, factor_cov,
                specific_var=specific_var, residual_return=residual_return, old_w=old_w,
                enable_trans_cost=enable_trans_cost, buy_cost=buy_cost, sell_cost=sell_cost,
                turnover_cap=turnover_cap, enable_full_inv_cons=enable_full_inv_cons,
                cash_ratio=cash_ratio, long_only=long_only, asset_cap=asset_cap,
                enable_factor_expo_cons=enable_factor_expo_cons)

        return optimized_weight

    # 在没有换手率限制条件下的问题, 按照常规方法来解
    def solve_ordinary_ir_opt(self, bench_weight, factor_expo, factor_ret, factor_cov, *,
                              specific_var=None, residual_return=None, old_w=None, enable_trans_cost=True,
                              buy_cost=1.5/1000, sell_cost=1.5/1000, turnover_cap=1.0,
                              enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                              asset_cap=None, enable_factor_expo_cons=False):
        # 注意要把dataframe的变量做成ndarray的格式, 主要是不能有nan
        # 因此规则是, 只要股票的某一因子暴露是nan, 则填为0
        # 因此, 如果这里有投资域的问题, 则需要注意, 要在传入此函数之前, 就将投资域外的股票全部去掉,
        # 否则, 一旦填成0, 投资域的限制就不复存在(因为投资域是按照nan来限制的)
        # 传入目标函数作为参数的字典
        obj_func_params = {}
        # 第一步, 设置传入目标函数作为参数的字典
        obj_func_params['factor_expo'] = factor_expo.fillna(0.0).values
        obj_func_params['bench_weight'] = bench_weight.fillna(0.0).values
        obj_func_params['factor_ret'] = factor_ret.fillna(0.0).values
        obj_func_params['factor_cov'] = factor_cov.fillna(0.0).values
        if isinstance(specific_var, pd.Series):
            obj_func_params['specific_var'] = specific_var.fillna(0.0).values
        if isinstance(residual_return, pd.Series):
            obj_func_params['residual_return'] = residual_return.fillna(0.0).values

        n_factors = factor_expo.shape[0]
        n_assets = factor_expo.shape[1]

        # 如果没有输入之前的持仓, 则默认是0, 即第一次买入
        if isinstance(old_w, pd.Series):
            old_w = old_w.values
        else:
            old_w = np.zeros(n_assets)

        # 开始设置手续费函数
        if enable_trans_cost:
            obj_func_params['trans_cost_func'] = functools.partial(optimizer.trans_cost, old_w = old_w,
                                                                  buy_cost=buy_cost, sell_cost=sell_cost)

        # 第二步, 设置限制条件
        # 不等式的限制条件
        ineq_cons_funcs = {}
        # # 首先是换手率的限制条件
        # if enable_turnover_cons:
        #     ineq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons, old_w=old_w,
        #                                                        turnover_cap=turnover_cap)

        # 因子暴露的不等式限制条件
        # 先实验一下限制市值的暴露
        if enable_factor_expo_cons:
            ineq_cons_funcs['size_cons'] = functools.partial(optimizer.factor_expo_cons, factor=
                                                             factor_expo.ix['CNE5S_SIZE', :],
                                                             lower_bound=False, limit=0)

        # 等式的限制条件
        eq_cons_funcs = {}
        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons, n_assets=n_assets,
                                                               cash_ratio=cash_ratio)

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap, )
        else:
            asset_bounds = (None, asset_cap, )

        # 条件设置完成, 开始调用对应的IR优化器来求解
        opt_results = self.ir_optimizer(obj_func_params, asset_bounds=asset_bounds,
                                        eq_cons_funcs=eq_cons_funcs, ineq_cons_funcs=ineq_cons_funcs)

        # 取优化的权重结果, 将其设置为series的格式
        optimized_weight = pd.Series(opt_results.x, index=factor_expo.columns)

        return optimized_weight

    # 在有换手率限制条件下, 解优化问题, 此时涉及到要将换手率的绝对值限制条件进行改写,
    # 因此要用一个新的函数来建立优化问题
    def solve_ir_opt_with_turnover_cons(self, bench_weight, factor_expo, factor_ret, factor_cov, *,
                              specific_var=None, residual_return=None, old_w=None, enable_trans_cost=True,
                              buy_cost=1.5/1000, sell_cost=1.5/1000, turnover_cap=1.0,
                              enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                              asset_cap=None, enable_factor_expo_cons=False):
        n_factors = factor_expo.shape[0]
        n_assets = factor_expo.shape[1]
        # split variable的数量, 为资产数量的2倍
        n_split = n_assets * 2

        # 注意要把dataframe的变量做成ndarray的格式, 主要是不能有nan
        # 因此规则是, 只要股票的某一因子暴露是nan, 则填为0
        # 因此, 如果这里有投资域的问题, 则需要注意, 要在传入此函数之前, 就将投资域外的股票全部去掉,
        # 否则, 一旦填成0, 投资域的限制就不复存在(因为投资域是按照nan来限制的)
        # 传入目标函数作为参数的字典
        obj_func_params = {}
        # 第一步, 设置传入目标函数作为参数的字典
        # 注意, 有split varaible的存在, 且把这些variable的暴露都设置为0, 让它们不会影响目标函数
        obj_func_params['factor_expo'] = np.concatenate((factor_expo.fillna(0.0).values,
                                                       np.zeros((n_factors, n_split))), axis=1)
        obj_func_params['bench_weight'] = np.concatenate((bench_weight.fillna(0.0).values,
                                                          np.zeros(n_split)), axis=0)
        # 这里暂时需要加上split variable, 因为现在给的是股票收益, 而不是因子收益
        obj_func_params['factor_ret'] = np.concatenate((factor_ret.fillna(0.0).values,
                                                        np.zeros(n_split)), axis=0)
        obj_func_params['factor_cov'] = factor_cov.fillna(0.0).values
        if isinstance(specific_var, pd.Series):
            obj_func_params['specific_var'] = np.concatenate((specific_var.fillna(0.0).values,
                                                              np.zeros(n_split)), axis=0)
        if isinstance(residual_return, pd.Series):
            obj_func_params['residual_return'] = np.concatenate((residual_return.fillna(0.0).values,
                                                                 np.zeros(n_split)), axis=0)

        # 如果没有输入之前的持仓, 则默认是0, 即第一次买入, 同样注意split variable的问题
        if isinstance(old_w, pd.Series):
            old_w = np.concatenate((old_w.values, np.zeros(n_split)), axis=0)
        else:
            old_w = np.zeros(n_assets + n_split)

        # 开始设置手续费函数
        if enable_trans_cost:
            obj_func_params['trans_cost_func'] = functools.partial(optimizer.trans_cost, old_w = old_w,
                                                                  buy_cost=buy_cost, sell_cost=sell_cost)

        # 第二步, 设置限制条件
        # 不等式的限制条件
        ineq_cons_funcs = {}
        # 等式的限制条件
        eq_cons_funcs = {}
        # 首先是换手率的限制条件, 注意换手率的限制条件包括不等式条件, 也包括不等式的条件
        ineq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons_ineq, n_assets=n_assets,
                                                             turnover_cap=turnover_cap)
        eq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons_eq, old_w=old_w,
                                                           n_assets=n_assets)
        # 还有一个关于split varaible的边界条件, 将在优化器函数中设置

        # # 因子暴露的不等式限制条件
        # # 先实验一下限制市值的暴露
        # if enable_factor_expo_cons:
        #     ineq_cons_funcs['size_cons'] = functools.partial(optimizer.factor_expo_cons, factor=
        #       factor_expo.ix['CNE5S_SIZE', :], lower_bound=False, limit=0)


        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons, n_assets=n_assets,
                                                               cash_ratio=cash_ratio)

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap,)
        else:
            asset_bounds = (None, asset_cap,)

        # 条件设置完成, 开始调用对应的IR优化器来求解
        opt_results = self.ir_optimizer(obj_func_params, asset_bounds=asset_bounds,
                                        eq_cons_funcs=eq_cons_funcs, ineq_cons_funcs=ineq_cons_funcs,
                                        n_split=n_split)

        # 取优化的权重结果, 将其设置为series的格式
        optimized_weight = pd.Series(opt_results.x[0:n_assets], index=factor_expo.columns)

        return optimized_weight



if __name__ == '__main__':
    from barra_base import barra_base
    from data import data

    opt = optimizer()

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

    # factor_cov = pd.read_hdf('bb_factor_eigencovmat_hs300_sf3', '123')
    # bb.base_data.factor_expo = pd.read_hdf('bb_factorexpo_hs300', '123')
    # spec_vol = pd.read_hdf('bb_factor_vraspecvol_hs300', '123')
    # spec_var = spec_vol ** 2
    # stock_return = pd.read_hdf('stock_alpha_hs300', '123')

    bench = data.read_data(['Weight_hs300'])
    bench = bench['Weight_hs300']

    factor_cov = pd.read_hdf('barra_fore_cov_mat', '123')
    bb.base_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
    spec_var = pd.read_hdf('barra_fore_spec_var_new', '123') * 0
    stock_return = pd.read_hdf('barra_real_spec_ret_new', '123')

    spec_var = spec_var.where(bench>0, np.nan)
    stock_return = stock_return.where(bench>0, np.nan)
    bb.base_data.factor_expo = bb.base_data.factor_expo.apply(lambda x: x.where(bench>0, np.nan), axis=(1,2))

    curr_time = '20170224'
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

    optimized_weight = opt.solve_ir_optimization(bench_weight, factor_expo, stock_return, factor_cov,
                                                 specific_var=spec_var, enable_turnover_cons=False,
                                                 enable_trans_cost=False, long_only=True, old_w=old_w,
                                                 turnover_cap=0.3, enable_factor_expo_cons=False,
                                                 asset_cap=None)
    pass























































































































































