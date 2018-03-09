import numpy as np
import pandas as pd
import matplotlib
import os
from scipy import optimize
import functools
from cvxopt import solvers, matrix

# 优化器类


class optimizer(object):
    def __init__(self):
        # 储存优化组合权重的series
        self.optimized_weight = pd.Series()
        # 储存优化器输出的原始结果对象, 不同的优化器包输出的结果对象不一样
        self.original_result = None
        # 储存标准的优化器结果, 为一个字典
        self.opt_result = {}
        # 最优化组合的预测风险
        self.forecasted_vol = None
        # 使用的优化包, 默认为cvxopt
        self.opt_package = 'cvxopt.solvers.qp'

    # 设置所使用的优化器
    def set_opt_package(self, opt_package):
        if opt_package in ['scipy', 'fmin', 'scipy.optimize.minimize', 'slsqp', 'sqp']:
            self.opt_package = 'scipy.optimize.minimize'
        elif opt_package in ['cvxopt', 'qp', 'cvxopt.solvers.qp']:
            self.opt_package = 'cvxopt.solvers.qp'
        else:
            raise ValueError('Please specify a valid optimization package, '
                             'currently supported: scipy and cvxopt.\n')

    # 目标函数
    def objective(self, w, add_params):
        pass

    # 优化函数, 作为从外界处理信息的solve_optimization函数与objective函数的媒介
    # optimizer fmin是作为scipy优化包的媒介
    def optimizer_fmin(self, obj_func_params, *, asset_bounds=(None, None,), eq_cons_funcs=None,
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

        self.original_result = optimize.minimize(self.objective, w, args=(obj_func_params, ),
            bounds=bounds, constraints=cons, method='SLSQP', options={'disp':False, 'maxiter':10000})

    # 作为媒介的函数, 在这里配置优化器需要的条件
    # optimizer qp是作为cvxopt优化包的媒介
    def optimizer_qp(self, opt_params, *, eq_cons=None, ineq_cons=None):
        # 提取因子暴露
        factor_expo = opt_params['factor_expo']
        # 资产个数
        n_assets = factor_expo.shape[1]
        # 首先配置收益部分
        # 因子收益
        if 'factor_ret' in opt_params:
            factor_ret_part = np.dot(opt_params['factor_ret'], factor_expo)
        else:
            factor_ret_part = np.zeros(n_assets)
        # 个股收益
        if 'residual_return' in opt_params:
            residual_ret_part = opt_params['residual_return']
        else:
            residual_ret_part = np.zeros(n_assets)
        return_part = factor_ret_part + residual_ret_part

        # 配置风险部分
        # 首先计算协方差矩阵部分
        cov_mat_part = np.dot(np.dot(factor_expo.T, opt_params['factor_cov']), factor_expo)
        # 如果有风险厌恶系数, 要对协方差矩阵进行风险厌恶系数的调整
        if hasattr(self, 'cov_risk_aversion'):
            cov_mat_part = cov_mat_part * self.cov_risk_aversion
        # 计算残余风险部分
        if 'specific_var' in opt_params:
            spec_risk_part = np.diag(opt_params['specific_var'])
        else:
            spec_risk_part = np.zeros((n_assets, n_assets))
        # 如果有风险厌恶系数, 要对协方差矩阵进行风险厌恶系数的调整
        if hasattr(self, 'spec_risk_aversion'):
            spec_risk_part = spec_risk_part * self.spec_risk_aversion
        risk_part = cov_mat_part + spec_risk_part

        # 首先是P, 注意, 由于qp中会对P乘以0.5, 因此这里先乘以2, 以保证数量级不改变
        P = matrix(risk_part * 2)
        # 对于q, 由于目标函数是最小化, 因此要在收益部分前加上负号
        q = matrix(return_part * -1)

        # 第二步, 对不等式限制条件进行配置
        ineq_constraints = pd.concat([cons[0] for cons in ineq_cons.values()], axis=0)
        ineq_targets = pd.concat([cons[1] for cons in ineq_cons.values()], axis=0)

        # 不等式限制条件分别为G和h
        G = matrix(ineq_constraints.as_matrix())
        h = matrix(ineq_targets)

        # 第三步, 对等式限制条件进行配置
        eq_constraints = pd.concat([cons[0] for cons in eq_cons.values()], axis=0)
        # eq_targets = pd.concat([cons[1] for cons in eq_cons.values()], axis=0)
        eq_targets = [cons[1] for cons in eq_cons.values()]

        # 等式限制条件分别为A和b
        A = matrix(eq_constraints.as_matrix())
        b = matrix(eq_targets)

        # 设置优化器的最大迭代次数
        solvers.options['maxiters'] = 10000
        # 隐藏优化器输出
        solvers.options['show_progress'] = False
        # 解优化, 这里的结果先存在orginal result里, 因为opt result需要统一成一个标准形式
        self.original_result = solvers.qp(P=P, q=q, A=A, b=b, G=G, h=h)
        pass


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

        # 根据不同的情况, 选择不同的优化器进行使用
        if self.opt_package == 'cvxopt.solvers.qp':
            # 如果有qp无法解决的限制条件, 则报错提示用户
            if enable_trans_cost or enable_turnover_cons:
                raise ValueError('Quadratic programming cannot handle some of constraints you specified.\n')

            self.solve_optimization_qp(bench_weight, factor_expo, factor_cov,
                factor_ret=factor_ret, residual_return=residual_return, specific_var=specific_var,
                enable_full_inv_cons=enable_full_inv_cons, long_only=long_only,
                asset_cap=asset_cap, factor_expo_cons=factor_expo_cons)

        elif self.opt_package == 'scipy.optimize.minimize':

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
        else:
            raise ValueError('Please specify a valid optimization package, '
                             'currently supported: scipy and cvxopt.\n')

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
            obj_func_params['trans_cost_func'] = functools.partial(optimizer.trans_cost,
                old_w = old_w, buy_cost=buy_cost, sell_cost=sell_cost)

        # 第二步, 设置限制条件
        # 不等式的限制条件
        ineq_cons_funcs = {}

        # 等式的限制条件
        eq_cons_funcs = {}
        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons,
                n_assets=n_assets, cash_ratio=cash_ratio)

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
                    ineq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons,
                        factor=factor_expo.ix[curr_con['factor'], :].fillna(0.0),
                        lower_bound=if_lower_bound, limit=curr_limit,
                        bench_weight=bench_weight.fillna(0.0))
                else:
                    # 等式条件就不需要判断上下限的问题
                    cons_name = curr_con['factor'] + '_eq_cons'
                    eq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons,
                        factor=factor_expo.ix[curr_con['factor'], :].fillna(0.0),
                        limit=curr_limit, bench_weight=bench_weight.fillna(0.0))

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap, )
        else:
            asset_bounds = (None, asset_cap, )

        # 条件设置完成, 开始调用对应的IR优化器来求解
        self.optimizer_fmin(obj_func_params, asset_bounds=asset_bounds,
            eq_cons_funcs=eq_cons_funcs, ineq_cons_funcs=ineq_cons_funcs)

        # 取优化的权重结果, 将其设置为series的格式
        self.optimized_weight = pd.Series(self.original_result.x, index=factor_expo.columns)

        # 将优化器输出的原始的结果, 转换为标准输出结果
        self.opt_result['success'] = self.original_result.success
        self.opt_result['status'] = self.original_result.status
        self.opt_result['message'] = self.original_result.message
        self.opt_result['fun'] = self.original_result.fun

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
            obj_func_params['trans_cost_func'] = functools.partial(optimizer.trans_cost,
                old_w = old_w, buy_cost=buy_cost, sell_cost=sell_cost)

        # 第二步, 设置限制条件
        # 不等式的限制条件
        ineq_cons_funcs = {}
        # 等式的限制条件
        eq_cons_funcs = {}
        # 首先是换手率的限制条件, 注意换手率的限制条件包括不等式条件, 也包括不等式的条件
        ineq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons_ineq,
            n_assets=n_assets, turnover_cap=turnover_cap)
        eq_cons_funcs['turnover_cons'] = functools.partial(optimizer.turnover_cons_eq,
            old_w=old_w, n_assets=n_assets)
        # 还有一个关于split varaible的边界条件, 将在优化器函数中设置

        # 持仓之和的限制条件
        if enable_full_inv_cons:
            eq_cons_funcs['full_inv_cons'] = functools.partial(optimizer.full_inv_cons,
                n_assets=n_assets, cash_ratio=cash_ratio)

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
                    ineq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons,
                        factor=factor_expo.ix[curr_con['factor'], :].fillna(0.0),
                        lower_bound=if_lower_bound, limit=curr_limit,
                        bench_weight=bench_weight.fillna(0.0))
                else:
                    # 等式条件就不需要判断上下限的问题
                    cons_name = curr_con['factor'] + '_eq_cons'
                    eq_cons_funcs[cons_name] = functools.partial(optimizer.factor_expo_cons,
                        factor=factor_expo.ix[curr_con['factor'], :].fillna(0.0),
                        limit=curr_limit, bench_weight=bench_weight.fillna(0.0))

        # 第三步, 设置变量的边界条件
        # 注意, 这里对变量的边界条件是对所有变量都是一样的,
        # 如果需要设置对某几个变量的条件, 则需要在上面的限制条件中单独设置
        if long_only:
            asset_bounds = (0, asset_cap,)
        else:
            asset_bounds = (None, asset_cap,)

        # 条件设置完成, 开始调用对应的IR优化器来求解
        self.optimizer_fmin(obj_func_params, asset_bounds=asset_bounds, eq_cons_funcs=eq_cons_funcs,
            ineq_cons_funcs=ineq_cons_funcs, n_split=n_split)

        # 取优化的权重结果, 将其设置为series的格式
        self.optimized_weight = pd.Series(self.opt_result.x[0:n_assets], index=factor_expo.columns)

        # 将优化器输出的原始的结果, 转换为标准输出结果
        self.opt_result['success'] = self.original_result.success
        self.opt_result['status'] = self.original_result.status
        self.opt_result['message'] = self.original_result.message
        self.opt_result['fun'] = self.original_result.fun

    # 利用标准二次规划解优化的函数, 二次规划函数只能求解线性限制条件下的解.
    # 注意: 传给二次规划函数的求解变量x在这里是超额持仓
    def solve_optimization_qp(self, bench_weight, factor_expo, factor_cov, *, factor_ret=None,
            residual_return=None, specific_var=None, enable_full_inv_cons=True, long_only=True,
            asset_cap=None, factor_expo_cons=None):
        # 因子收益或股票的alpha收益, 必须要有一个, 否则报错
        assert np.logical_or(factor_ret is not None, residual_return is not None), 'Factor return and ' \
            'residual return of stocks cannot both be None, please specify at least one of them!\n'

        # 首先将要用到的数据存入一个字典
        opt_params = {}
        opt_params['factor_expo'] = factor_expo.fillna(0.0).values
        opt_params['bench_weight'] = bench_weight.fillna(0.0).values
        opt_params['factor_cov'] = factor_cov.fillna(0.0).values
        if isinstance(factor_ret, pd.Series):
            opt_params['factor_ret'] = factor_ret.fillna(0.0).values
        if isinstance(specific_var, pd.Series):
            opt_params['specific_var'] = specific_var.fillna(0.0).values
        if isinstance(residual_return, pd.Series):
            opt_params['residual_return'] = residual_return.fillna(0.0).values

        n_factors = factor_expo.shape[0]
        n_assets = factor_expo.shape[1]

        # 第二步, 设置限制条件
        # 添加限制条件, 注意, 为了和qp函数一样, 这里的每一个限制条件都是一个中的元素都是一个tuple,
        # 分别表示左边的constraint和右边的target
        # 不等式的限制条件
        ineq_cons = {}
        # 等式的限制条件
        eq_cons = {}

        # 持仓之和的限制条件, 由于传入的求解变量是超额持仓, 因此, 全投资的条件,
        # 实际上是超额持仓之和等于 1 - 基准权重之和. 因此, 可以保证最终的组合持仓之和一定等于1.
        # 注意区分full inv和country factor中性. 由于基准有停牌股的原因, 基准权重之和可能不为1, 这时
        # full inv并不能做到country factor中性, 即full inv此时会有市场暴露(country factor暴露)
        if enable_full_inv_cons:
            eq_cons['full_inv_cons'] = (pd.Series(1.0, index=factor_expo.columns).to_frame().T,
                                        1.0 - bench_weight.fillna(0.0).sum())

        # 设置因子暴露的限制条件
        if isinstance(factor_expo_cons, pd.DataFrame):
            # 根据用户设定的因子暴露限制条件进行设置.
            for i, curr_con in factor_expo_cons.iterrows():
                curr_limit = curr_con['limit']
                # 各个股票在当前因子上的暴露
                curr_factor_expo = factor_expo.ix[curr_con['factor'], :].fillna(0.0)
                # 然后判断是等式条件还是不等式条件
                if not curr_con['if_eq']:
                    # 如果是不等式条件, 则判断是upper条件还是lower条件, 即要设置的是上限还是下限
                    # 由于qp的标准形式是小于等于, 因此如果是设置下限, 则要取负号
                    if curr_con['if_lower_bound']:
                        curr_factor_expo = - curr_factor_expo
                        curr_limit = - curr_limit
                    # 将限制和目标储存在不等式限制条件中
                    cons_name = curr_con['factor'] + '_ineq_' + \
                        ('lower' if curr_con['if_lower_bound'] else 'upper') + '_cons'
                    ineq_cons[cons_name] = (curr_factor_expo.to_frame().T, curr_limit)
                else:
                    # 如果是等式条件, 则直接储存, 不需要考虑上下限的问题
                    cons_name = curr_con['factor'] + '_eq_cons'
                    eq_cons[cons_name] = (curr_factor_expo.to_frame().T, curr_limit)

        # 设置单支股票的持仓限制条件
        if long_only:
            # 由于qp的标准形式是小于等于, 因此需要加上负号
            long_only_cons = pd.DataFrame(-1.0 * np.eye(n_assets), columns=factor_expo.columns)
            # 由于传入的求解变量是超额持仓, 因此, 持仓大于0, 即超额持仓大于负基准权重,
            # 而由于qp的标准形式是小于等于, 加上负号后, 等式右边的target直接变成了正的基准权重
            long_only_targets = bench_weight.fillna(0.0)
            ineq_cons['long_only'] = (long_only_cons, long_only_targets)
        if asset_cap is not None:
            # 设置单支股票持仓上限
            asset_cap_cons = pd.DataFrame(np.eye(n_assets), columns=factor_expo.columns)
            # 由于传入的求解变量是超额持仓, 因此, 持仓小于asset cap, 即超额持仓小于asset cap减基准权重
            asset_cap_targets = asset_cap - bench_weight.fillna(0.0)
            ineq_cons['asset_cap'] = (asset_cap_cons, asset_cap_targets)

        # 设置完毕, 利用优化器求解
        self.optimizer_qp(opt_params, eq_cons=eq_cons, ineq_cons=ineq_cons)

        results_np = np.array(self.original_result['x']).squeeze()
        # 注意, 由于解出的x是超额持仓, 因此需要加上基准权重, 才等于最终的组合持仓
        self.optimized_weight = pd.Series(results_np, index=factor_expo.columns) + \
                                bench_weight.fillna(0.0)

        # 将优化器输出的原始的结果, 转换为标准输出结果
        if self.original_result['status'] == 'optimal':
            self.opt_result['success'] = 1
            self.opt_result['status'] = 0
            self.opt_result['message'] = 'Optimal solution found.'
        else:
            self.opt_result['success'] = 0
            self.opt_result['status'] = 1
            self.opt_result['message'] = 'Unknown error caused optimization failed.\n'
        self.opt_result['fun'] = self.original_result['primal objective']
        pass

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
    # 注意区分full inv和country factor中性. 由于基准有停牌股的原因, 基准权重之和可能不为1, 这时,
    # 不加入现金资产的full inv并不能做到country factor中性, 即full inv此时会有市场暴露(country factor暴露)
    @staticmethod
    def full_inv_cons(w, *, n_assets, cash_ratio=0):
        total_weight = np.sum(w[:n_assets])
        # 减去设置的限制
        full_inv_cons = total_weight - (1 - cash_ratio)
        return full_inv_cons

































