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
        if 'specific_vol' in add_params:
            specific_vol = add_params['specific_vol']
        else:
            specific_vol = np.zeros(w.shape)
        if 'residual_return' in add_params:
            residual_return = add_params['residual_return']
        else:
            residual_return = np.zeros(w.shape)

        # 计算组合相对基准的超额持仓
        active_w = w - bench_weight
        # 计算active return的部分
        # 首先是common factor的收益
        active_ret = factor_ret.dot(factor_expo.dot(active_w))
        # 如果有预测的residual return, 还需要加上这个部分
        active_ret += residual_return.dot(active_w)

        # active sigma的部分
        # 首先是common factor部分的风险
        active_var = factor_expo.dot(active_w).T.dot(factor_cov).dot(factor_expo.dot(active_w))
        # 如果有预测的specific risk, 需要加上这个部分
        active_var += active_w.T.dot(specific_vol.dot(active_w))

        # 计算IR
        # 默认的手续费函数是0, 因此如果有手续费函数, 还需要将收益减去手续费函数这一部分
        ir = (active_ret - trans_cost_func(w)) / np.sqrt(active_var)

        return ir

    # 最大化IR的优化函数
    def ir_optimizer(self, obj_func_params, *, asset_bounds=(None, None,), eq_cons_funcs=None,
                     ineq_cons_funcs=None):
        # 去因子暴露矩阵来做参照物, 以得到资产数和因子数
        n_factors = obj_func_params['factor_expo'].shape[0]
        n_assets = obj_func_params['factor_expo'].shape[1]
        # 初始的w点为0
        w = np.zeros(len(n_assets))

        # 设置限制条件
        # 股票的边界限制条件, 如股票不能做空时, 必须大于0, 或者设置每支股票权重不能超过多少的限制
        bounds = [asset_bounds for i in range(len(w))]

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

        opt_results = optimize.minimize(self.ir_objective, w, args=(obj_func_params, ), bounds=bounds,
                                        constraints=cons)

        return opt_results

    # 计算手续费的函数
    @staticmethod
    def trans_cost(w, *, old_w, buy_cost=1.5/1000, sell_cost=1.5/1000):
        # 买入的手续费
        trans_cost_buy = np.sum(np.maximum(w - old_w, 0)) * buy_cost
        trans_cost_sell = np.sum(np.maximum(old_w - w, 0)) * sell_cost
        total_trans_cost = trans_cost_buy + trans_cost_sell
        return total_trans_cost

    # 添加换手率限制条件的函数,
    @staticmethod
    def turnover_cons(w, *, old_w, turnover_cap=1):
        turnover = np.abs(w - old_w)/2
        # 根据设置的限制条件, 减去设置的上限
        turnover_cons = turnover - turnover_cap
        # 因为scipy.optimize.minimize函数的条件的形式是大于0, 因此要取负号
        # 暂时不支持现金, 等添加了现金功能之后再来修改
        return - turnover_cons

    # 设置持仓之和的条件函数, 在不支持现金资产的时候, 总是限制持仓之和为1
    @staticmethod
    def full_inv_cons(w, *, cash_ratio=0):
        total_weight = np.sum(w)
        # 减去设置的限制
        full_inv_cons = total_weight - (1 - cash_ratio)
        return full_inv_cons

    # 建立最大化IR问题的函数, 这个函数可以作为和外界的接口来使用
    # 注意, 这里输入的变量是pd.DataFrame或者pd.Series, 但是要将其做成np.ndarray的类型
    def solve_ir_optimization(self, bench_weight, factor_expo, factor_ret, factor_cov, *,
                              specific_vol=None, residual_return=None, old_w=None, enable_trans_cost=True,
                              buy_cost=1.5/1000, sell_cost=1.5/1000, enable_turnover_cons=True,
                              turnover_cap=1, enable_full_inv_cons=True, cash_ratio=0, long_only=True,
                              asset_cap=None):

        # 注意要把dataframe的变量做成ndarray的格式, 主要是不能有nan
        # 因此规则是, 只要股票的某一因子暴露是nan, 则填为0
        # 因此, 如果这里有投资域的问题, 则需要注意, 要在传入此函数之前, 就将投资域外的股票全部去掉,
        # 否则, 一旦填成0, 投资域的限制就不复存在(因为投资域是按照nan来限制的)
        # 传入目标函数作为参数的字典
        obj_func_params = {}
        # 第一步, 设置传入目标函数作为参数的字典
        obj_func_params['factor_expo'] = factor_expo.fillna(0.0).values
        obj_func_params['bench_weight'] = bench_weight.fillna(0.0).values
        obj_func_params['factor_ret'] = factor_ret.values
        obj_func_params['factor_cov'] = factor_cov.values
        if bool(specific_vol):
            obj_func_params['specific_vol'] = specific_vol.fillna(0.0).values
        if bool(residual_return):
            obj_func_params['residual_return'] = residual_return.fillna(0.0).values

        n_factors = factor_expo.shape[0]
        n_assets = factor_expo.shape[1]

        # 如果没有输入之前的持仓, 则默认是0, 即第一次买入
        if not old_w:
            old_w = np.zeros(len(n_assets))

        # 开始设置手续费函数
        if enable_trans_cost:
            obj_func_params['trans_cost_func'] = functools.partial(optimizer.trans_cost, old_w = old_w,
                                                                  buy_cost=buy_cost, sell_cost=sell_cost)

        # 第二步, 设置限制条件
        # 不等式的限制条件
        ineq_cons_funcs = {}
        # 首先是换手率的限制条件
        if enable_turnover_cons:
            ineq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons, old_w=old_w,
                                                               turnover_cap=turnover_cap)

        # 等式的限制条件
        eq_cons_funcs = {}
        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons,
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




























































































































































