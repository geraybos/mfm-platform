import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from single_factor_strategy import single_factor_strategy
from backtest import backtest
from dynamic_backtest import dynamic_backtest
from barra_base import barra_base
from optimizer import optimizer

# 多因子策略的类

class multi_factor_strategy(single_factor_strategy):
    """ Base class for multifactor strategy

    foo
    """
    def __init__(self):
        single_factor_strategy.__init__(self)

    # 构造最大化IR的组合
    def construct_ir_optimized_portfolio(self):
        # 首先设置optimizer
        self.optimizer = optimizer()

        # 第一次调仓的优化必须单独解, 因为我们的动态回测顺序是先回测这一天, 然后设置下一天的目标持仓
        # 如果不单独解第一天, 遇到第一天就是换仓日的时候, 这次换仓将不会执行, 因为会先回测这一天,
        # 这时还没有解出来第一天的目标持仓的话, 换仓就被错过了, 因为第二天就是新的一天了, 不一定会是换仓日
        first_holding_day = self.holding_days[self.dy_bkt.bkt_start:self.dy_bkt.bkt_end].iloc[0]
        # 先把数据取出来
        if self.strategy_data.stock_pool == 'all':
            bench_name = 'Weight_zz500'
        else:
            bench_name = 'Weight_' + self.strategy_data.stock_pool
        curr_bench_weight = self.strategy_data.benchmark_price.ix[bench_name, first_holding_day, :]
        curr_factor_expo = self.strategy_data.factor_expo.ix[:, first_holding_day, :].T
        factor_ret = self.factor_return.ix[first_holding_day, :]
        factor_cov = self.factor_cov.ix[first_holding_day, :, :]
        # 因为是第一次调仓, 因此初始持仓是0, 而且几乎百分之百的换手率, 因此此处不应当添加换手率的限制

        optimized_weight = self.optimizer.solve_ir_optimization(curr_bench_weight, curr_factor_expo,
            factor_ret, factor_cov, enable_turnover_cons=False)
        self.dy_bkt.tar_pct_position.holding_matrix.ix[first_holding_day, :] = optimized_weight


        # 开始循环解优化, 注意, 循环迭代的是动态回测类里的execute_backtest这个函数
        # 由于该函数不返回任何值, 因此这里的n是None
        for cursor, n in enumerate(self.dy_bkt.execute_backtest()):
            # 根据cursor取当前的时间, 注意, 因为动态回测是先回测一次, 然后再返回到这里来
            # 因此, 当前时间点是cursor+1, 即cursor已经回测过了
            if cursor < self.dy_bkt.tar_pct_position.holding_matrix.shape[0]:
                curr_time = self.dy_bkt.tar_pct_position.holding_matrix.index[cursor+1]
            else:
                # 如果已经循环测完了最后一次, 则直接结束这次循环, 也即结束整个循环
                continue

            # 如果不是调仓日, 则不进行操作, 直接进行下一次的迭代
            if curr_time not in self.holding_days:
                continue

            # 在是调仓日的情况下
            else:
                # 先把数据取出来
                curr_bench_weight = self.strategy_data.benchmark_price.ix[bench_name, curr_time, :]
                curr_factor_expo = self.strategy_data.factor_expo.ix[:, curr_time, :].T
                factor_ret = self.factor_return.ix[curr_time, :]
                factor_cov = self.factor_cov.ix[curr_time, :, :]
                # 最新的持仓
                latest_holding = self.dy_bkt.real_pct_position.holding_matrix.ix[cursor, :]

                # 已经不可能是第一个调仓日, 因此按照正常的逻辑解优化即可
                optimized_weight = self.optimizer.solve_ir_optimization(curr_bench_weight,
                    curr_factor_expo, factor_ret, factor_cov, old_w=latest_holding,
                    enable_turnover_cons=True)
                self.dy_bkt.tar_pct_position.holding_matrix.ix[curr_time, :] = optimized_weight

        pass


if __name__ == '__main__':
    bb = barra_base()
    bb.construct_barra_base()
    mf = multi_factor_strategy()
    mf.strategy_data.factor_expo = bb.bb_data.factor_expo.iloc[0:10]
    mf.strategy_data.benchmark_price = data.read_data(['Weight_zz500'], shift=True)
    factor_return = data.read_data(['bb_factor_return_all'], shift=True)
    mf.factor_return = factor_return.ix['bb_factor_return_all', :, 0:10]
    mf.factor_cov = mf.factor_return.expanding().corr()
    mf.generate_holding_days()
    mf.initialize_dynamic_backtest(bkt_start=pd.Timestamp('2010-04-02'), bkt_end=pd.Timestamp('2017-06-20'))
    mf.construct_ir_optimized_portfolio()

























































































































































