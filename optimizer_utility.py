import numpy as np
import pandas as pd
import matplotlib
import os
from scipy import optimize
import functools

from optimizer import optimizer


# 优化器类
class optimizer_utility(optimizer):
    def __init__(self):
        optimizer.__init__(self)
        # 初始化两个风险部分的风险厌恶系数
        # 将初始值设置为0.75, 由于从打分到预期收益要乘以IC, 假设IC为0.01, 则需要将常说的风险厌恶系数乘以100
        self.cov_risk_aversion = 0.75 * 100
        self.spec_risk_aversion = 0.75 * 100

    # 设置风险厌恶系数
    def set_risk_aversion(self, *, cov_risk_aversion=None, spec_risk_aversion=None):
        if cov_risk_aversion is not None:
            self.cov_risk_aversion = cov_risk_aversion * 100
        if spec_risk_aversion is not None:
            self.spec_risk_aversion = spec_risk_aversion * 100

    # 最大化效用函数的目标函数
    def objective(self, w, add_params):
        # 为了能传参数, 其余的参数按照字典的形式传入
        bench_weight = add_params['bench_weight']
        factor_expo = add_params['factor_expo']
        factor_cov = add_params['factor_cov']
        # 如果不传入手续费函数, 则只考虑扣费前收益
        if 'trans_cost_func' in add_params:
            trans_cost_func = add_params['trans_cost_func']
        else:
            trans_cost_func = lambda x: 0
        # 可以加入预测的每支股票的specific risk或者residual return, 如果没有, 则默认为0
        # 这时只考虑common factor对股票的风险收益的影响
        # 也可以加入预测的每个因子的因子收益, 因子收益和每支股票的residual return至少要有一个
        if 'factor_ret' in add_params:
            factor_ret = add_params['factor_ret']
        else:
            factor_ret = np.zeros(factor_cov.shape[0])
        if 'residual_return' in add_params:
            residual_return = add_params['residual_return']
        else:
            residual_return = np.zeros(w.shape)
        if 'specific_var' in add_params:
            specific_var = add_params['specific_var']
        else:
            specific_var = np.zeros(w.shape)

        # 计算组合相对基准的超额持仓
        active_w = w - bench_weight
        # 计算active return的部分
        # 首先是common factor的收益
        active_ret = factor_ret.dot(factor_expo.dot(active_w))
        # 如果有预测的residual return, 还需要加上这个部分
        active_ret += residual_return.dot(active_w)

        # active sigma的部分
        # 首先是common factor部分的风险
        # 注意需要乘以common factor部分的风险厌恶系数
        active_var_penalty = factor_expo.dot(active_w).T.dot(factor_cov).dot(factor_expo.dot(active_w)) \
                             * self.cov_risk_aversion
        # 如果有预测的specific risk, 需要加上这个部分
        # 同样注意乘以specific var部分的风险厌恶系数
        active_var_penalty += active_w.dot(np.multiply(specific_var, active_w)) * self.spec_risk_aversion

        # 计算效用函数
        # 默认的手续费函数是0, 因此如果有手续费函数, 还需要将收益减去手续费函数这一部分
        # 这里手续费函数直接从收益上减去, 没有乘以惩罚系数, 因为手续费已经表示成了收益的形式, 和收益具有一样的数量级
        utility = active_ret - trans_cost_func(w) - active_var_penalty

        # 因为目标函数的形式是最小化, 因此要取负号
        return - utility
