import numpy as np
import pandas as pd
import matplotlib
import os
from scipy import optimize
import functools

# 优化器类
class optimizer(object):
    def __init__(self):
        # 储存优化组合权重的series
        self.optimized_weight = pd.Series()
        # 储存优化器返回的sicpy OptimizeResult对象
        self.opt_result = None
        # 最优化组合的预测风险
        self.forecasted_vol = None

    # 目标函数
    def objective(self, w, add_params):
        pass

    # 优化函数, 作为从外界处理信息的solve_optimization函数与objective函数的媒介
    def optimizer(self, obj_func_params, *, asset_bounds=(None, None,), eq_cons_funcs=None,
                  ineq_cons_funcs=None, n_split=None):
        # 去因子暴露矩阵来做参照物, 以得到资产数和因子数
        n_factors = obj_func_params['factor_expo'].shape[0]
        n_varaibles = obj_func_params['factor_expo'].shape[1]
        # 初始的w点为benchmark weight
        if isinstance(n_split, int):
            n_assets = int(n_split / 2)
            w = np.concatenate((obj_func_params['bench_weight'], np.zeros(2*n_assets)), axis=0)
        else:
            n_assets = n_varaibles
            # w = np.full(n_assets, 1/n_assets)
            w = obj_func_params['bench_weight']
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

        # import time
        # start_time = time.time()
        opt_result = optimize.minimize(self.objective, w, args=(obj_func_params, ), bounds=bounds,
                                        constraints=cons, method='SLSQP',
                                        options={'disp':False, 'maxiter':10000})
        self.opt_result = opt_result
        # print("time: {0} seconds\n".format(time.time() - start_time))



    # 建立优化问题的函数, 这个函数可以作为和外界的接口来使用
    # 注意, 这里输入的变量是pd.DataFrame或者pd.Series, 但是要将其做成np.ndarray的类型
    def solve_optimization(self, bench_weight, factor_expo, factor_cov, *, factor_ret=None,
                           residual_return=None, specific_var=None, old_w=None, enable_trans_cost=False,
                           buy_cost=1.5/1000, sell_cost=1.5/1000, enable_turnover_cons=False,
                           turnover_cap=1.0, enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                           asset_cap=None, factor_expo_cons=None):
        # 因子收益或股票的alpha收益, 必须要有一个, 否则报错
        assert np.logical_or(factor_ret is not None, residual_return is not None), 'Factor return and ' \
            'residual return of stocks cannot both be None, please specify at least one of them!\n'

        if enable_turnover_cons:
            self.solve_opt_with_turnover_cons(bench_weight, factor_expo, factor_cov,
                factor_ret=factor_ret, residual_return=residual_return, specific_var=specific_var,
                old_w=old_w, enable_trans_cost=enable_trans_cost, buy_cost=buy_cost,
                sell_cost=sell_cost, turnover_cap=turnover_cap, enable_full_inv_cons=enable_full_inv_cons,
                cash_ratio=cash_ratio, long_only=long_only, asset_cap=asset_cap,
                factor_expo_cons=factor_expo_cons)
        else:
            self.solve_ordinary_opt(bench_weight, factor_expo, factor_cov,
                factor_ret=factor_ret, residual_return=residual_return, specific_var=specific_var,
                old_w=old_w, enable_trans_cost=enable_trans_cost, buy_cost=buy_cost,
                sell_cost=sell_cost, turnover_cap=turnover_cap, enable_full_inv_cons=enable_full_inv_cons,
                cash_ratio=cash_ratio, long_only=long_only, asset_cap=asset_cap,
                factor_expo_cons=factor_expo_cons)

        # 计算最优化结果的预测波动
        self.get_forecasted_vol(bench_weight, factor_expo, factor_cov, specific_var=specific_var)

    # 在没有换手率限制条件下的问题, 按照常规方法来解
    def solve_ordinary_opt(self, bench_weight, factor_expo, factor_cov, *, factor_ret=None,
                           residual_return=None, specific_var=None, old_w=None, enable_trans_cost=False,
                           buy_cost=1.5/1000, sell_cost=1.5/1000, turnover_cap=1.0,
                           enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                           asset_cap=None, factor_expo_cons=None):
        # 注意要把dataframe的变量做成ndarray的格式, 主要是不能有nan
        # 因此规则是, 只要股票的某一因子暴露是nan, 则填为0
        # 因此, 如果这里有投资域的问题, 则需要注意, 要在传入此函数之前, 就将投资域外的股票全部去掉,
        # 否则, 一旦填成0, 投资域的限制就不复存在(因为投资域是按照nan来限制的)
        # 传入目标函数作为参数的字典
        obj_func_params = {}
        # 第一步, 设置传入目标函数作为参数的字典
        obj_func_params['factor_expo'] = factor_expo.fillna(0.0).values
        obj_func_params['bench_weight'] = bench_weight.fillna(0.0).values
        obj_func_params['factor_cov'] = factor_cov.fillna(0.0).values
        if isinstance(factor_ret, pd.Series):
            obj_func_params['factor_ret'] = factor_ret.fillna(0.0).values
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

        # 等式的限制条件
        eq_cons_funcs = {}
        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons, n_assets=n_assets,
                                                               cash_ratio=cash_ratio)

        # 因子暴露的限制条件, 包括等式和不等式, 需要用户自己定义
        if isinstance(factor_expo_cons, pd.DataFrame):
            # 根据用户设定的因子暴露限制条件进行设置.
            for i, curr_con in factor_expo_cons.iterrows():
                curr_limit = curr_con['limit']
                # 先判断是等式条件还是不等式条件
                if not curr_con['if_eq']:
                    # 如果是不等式条件, 则判断是upper的条件还是lower的条件, 即设置的是上限还是下限
                    if_lower_bound = curr_con['if_lower_bound']
                    cons_name = curr_con['factor'] + '_ineq_' + ('lower' if if_lower_bound else 'upper') + '_cons'
                    ineq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons, factor=
                        factor_expo.ix[curr_con['factor'], :].fillna(0.0), lower_bound=if_lower_bound,
                        limit=curr_limit, bench_weight=bench_weight.fillna(0.0))
                else:
                    # 等式条件就不需要判断上下限的问题
                    cons_name = curr_con['factor'] + '_eq_cons'
                    eq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons, factor=
                        factor_expo.ix[curr_con['factor'], :].fillna(0.0), limit=curr_limit,
                        bench_weight=bench_weight.fillna(0.0))

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap, )
        else:
            asset_bounds = (None, asset_cap, )

        # 条件设置完成, 开始调用对应的IR优化器来求解
        self.optimizer(obj_func_params, asset_bounds=asset_bounds,
                       eq_cons_funcs=eq_cons_funcs, ineq_cons_funcs=ineq_cons_funcs)

        # 取优化的权重结果, 将其设置为series的格式
        self.optimized_weight = pd.Series(self.opt_result.x, index=factor_expo.columns)

    # 在有换手率限制条件下, 解优化问题, 此时涉及到要将换手率的绝对值限制条件进行改写,
    # 因此要用一个新的函数来建立优化问题
    def solve_opt_with_turnover_cons(self, bench_weight, factor_expo, factor_cov, *, factor_ret=None,
                                     specific_var=None, residual_return=None, old_w=None,
                                     enable_trans_cost=False, buy_cost=1.5/1000, sell_cost=1.5/1000,
                                     turnover_cap=1.0, enable_full_inv_cons=True, cash_ratio=0,
                                     long_only=True, asset_cap=None, factor_expo_cons=None):
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
        obj_func_params['factor_cov'] = factor_cov.fillna(0.0).values
        if isinstance(factor_ret, pd.Series):
            # 因子收益不需要加split variable
            obj_func_params['factor_ret'] = factor_ret.fillna(0.0).values
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

        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons, n_assets=n_assets,
                                                               cash_ratio=cash_ratio)

        # 因子暴露的限制条件, 包括等式和不等式, 需要用户自己定义
        if isinstance(factor_expo_cons, pd.DataFrame):
            # 根据用户设定的因子暴露限制条件进行设置.
            for i, curr_con in factor_expo_cons.iterrows():
                curr_limit = curr_con['limit']
                # 先判断是等式条件还是不等式条件
                if not curr_con['if_eq']:
                    # 如果是不等式条件, 则判断是upper的条件还是lower的条件, 即设置的是上限还是下限
                    if_lower_bound = curr_con['if_lower_bound']
                    cons_name = curr_con['factor'] + '_ineq_' + ('lower' if if_lower_bound else 'upper') + '_cons'
                    ineq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons, factor=
                        factor_expo.ix[curr_con['factor'], :].fillna(0.0), lower_bound=if_lower_bound,
                        limit=curr_limit, bench_weight=bench_weight.fillna(0.0))
                else:
                    # 等式条件就不需要判断上下限的问题
                    cons_name = curr_con['factor'] + '_eq_cons'
                    eq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons, factor=
                        factor_expo.ix[curr_con['factor'], :].fillna(0.0), limit=curr_limit,
                        bench_weight=bench_weight.fillna(0.0))

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap,)
        else:
            asset_bounds = (None, asset_cap,)

        # 条件设置完成, 开始调用对应的IR优化器来求解
        self.optimizer(obj_func_params, asset_bounds=asset_bounds, eq_cons_funcs=eq_cons_funcs,
                       ineq_cons_funcs=ineq_cons_funcs, n_split=n_split)

        # 取优化的权重结果, 将其设置为series的格式
        self.optimized_weight = pd.Series(self.opt_result.x[0:n_assets], index=factor_expo.columns)

    # 根据最优化持仓结果计算出的预期波动
    def get_forecasted_vol(self, bench_weight, factor_expo, factor_cov, *, specific_var=None):
        var_common_factor = factor_expo.fillna(0).dot(self.optimized_weight.sub(bench_weight).fillna(0)).\
            T.dot(factor_cov.fillna(0)).dot(factor_expo.fillna(0).dot(
            self.optimized_weight.sub(bench_weight).fillna(0)))
        if specific_var is None:
            specific_var = pd.Series(0.0, self.optimized_weight.index)
        var_specific = self.optimized_weight.sub(bench_weight).fillna(0).dot(
            specific_var.mul(self.optimized_weight.sub(bench_weight).fillna(0)).fillna(0))

        self.forecasted_vol = np.sqrt(var_common_factor + var_specific)

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
    def factor_expo_cons(w, *, factor, lower_bound=True, limit=0, bench_weight=None):
        # 如果有bench_weight, 则是对active部分的因子暴露做限制
        if isinstance(bench_weight, pd.Series):
            w = w - bench_weight
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

















































































































































