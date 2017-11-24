import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from backtest_data import backtest_data
from position import position
from performance import performance
from performance_attribution import performance_attribution
from backtest import backtest

# 动态回测类
# 即, 执行回测的函数execute_backtest的循环并不是直接循环, 而是每次循环后,
# 等待对应的策略类根据已经计算出的回测结果, 生成新的目标持仓, 然后根据新的目标再进行下一期的回测
# 因此, 初始化回测类的持仓矩阵, 实际可能只有第一期是有用的
# 所有需要用到此前结果的动态策略, 都需要用到动态回测类,
# 但是, 在策略构建时, 回测就已经完成了, 并不需要再在策略构建完成后, 再单独进行回测.

class dynamic_backtest(backtest):
    """ The class for dynamic backtest

    foo
    """

    # 动态回测的核心, 执行策略的函数, 并不是直接循环的, 而是一个generator, 每次循环后, 会等待外部调用它的函数
    # 待外部调用它的函数根据之前回测出的结果, 进行各种修改后, 传入新的交易指令,
    # 此时此执行策略的函数才进行下一次循环
    # 注意这里的函数不是普通函数, 而是一个generator, 因此, 对应的调用他的策略类里的函数需要对它进行循环调用
    def execute_backtest(self):
        """ Execute dynamic backtest. Note that this function is actually a generator.

        foo
        """
        cursor = -1
        # 开始执行循环，对tar_pct_position.holding_matrix进行循环
        for curr_time, curr_tar_pct_holding in self.tar_pct_position.holding_matrix.iterrows():

            cursor += 1

            # 如果为回测第一天
            if cursor == 0:
                self.deal_with_first_day(curr_time, curr_tar_pct_holding)

            # 非回测第一天
            # 如果为非调仓日
            elif curr_time not in self.bkt_position.holding_matrix.index:
                # 移动持仓和现金
                self.real_vol_position.holding_matrix.ix[cursor, :] = \
                    self.real_vol_position.holding_matrix.ix[cursor - 1, :] * 1
                # 注意这里应当还有昨天的现金无风险收益, 暂时还未添加
                self.real_vol_position.cash.iloc[cursor] = \
                    self.real_vol_position.cash.iloc[cursor - 1] * 1

                # 处理当日退市的股票
                self.deal_with_held_delisted(curr_time, cursor)

            # 如果为调仓日
            else:
                # 首先，将上一期的持仓移动到这一期，同时移动现金
                self.real_vol_position.holding_matrix.ix[cursor, :] = \
                    self.real_vol_position.holding_matrix.ix[cursor - 1, :] * 1
                # 注意这里应当还有昨天的现金无风险收益, 暂时还未添加
                self.real_vol_position.cash.iloc[cursor] = \
                    self.real_vol_position.cash.iloc[cursor - 1] * 1

                # 首先必须有对当天退市股票的处理
                self.deal_with_held_delisted(curr_time, cursor)

                # 检查当前持仓的股票是否有已经停牌的, 输出提示
                self.check_if_holding_tradable(curr_time)
                # 检查目标买入股票是否有不可交易的, 输出提示
                self.check_if_tar_holding_tradable(curr_tar_pct_holding, curr_time)

                # 计算预计持仓量矩阵，以确定当期的交易计划
                proj_vol_holding = self.get_proj_vol_holding(curr_tar_pct_holding, cursor)

                # 根据预计持仓矩阵，进行实际交易
                self.execute_real_trading(curr_time, cursor, proj_vol_holding)

            # 对回测结果进行中间总结, 对应的外部的策略类可能会需要这些数据来做出动态策略
            self.dynamic_finalize_backtest(curr_time, cursor)

            # 什么也不返回, 但是使得函数成为一个generator, 必须加上这一句
            yield None


    # 计算实际持仓的比例, 即real_pct_position
    # 在动态回测中, 每一期都需要计算一次实际的持仓比例, 账户的价值序列, 以及各种信息
    # 因为这些信息都有可能会被外部的动态策略用到, 因此需要全部计算出来
    def dynamic_finalize_backtest(self, curr_time, cursor):
        # 计算当期的实际持仓
        curr_real_pct_matrix = self.real_vol_position.holding_matrix.ix[curr_time, :]. \
            mul(self.bkt_data.stock_price.ix['ClosePrice_adj', curr_time, :]).fillna(0.0)
        if (curr_real_pct_matrix != 0.0).any():
            curr_real_pct_matrix = curr_real_pct_matrix.div(curr_real_pct_matrix.sum())
        self.real_pct_position.holding_matrix.ix[curr_time, :] = curr_real_pct_matrix

        # 计算账面的价值
        self.account_value.ix[curr_time] = (self.real_vol_position.holding_matrix.ix[curr_time, :] *
            100 * self.bkt_data.stock_price.ix['ClosePrice_adj', curr_time, :]).sum() + \
            self.real_vol_position.cash.ix[curr_time]
        # 现金的实际持有比例, 为实际现金数量除以账面价值
        self.real_pct_position.cash.ix[curr_time] = self.real_vol_position.cash.ix[curr_time] / \
            self.account_value.ix[curr_time]

        # 如果是第一期, 需要将账面价值序列和基准序列进行拼接, 及加上初始资金的第一项, 时间为回测开始前1秒
        # 这一点和一般的backtest的操作是一样的, 只不过改到了只有底一期的时候才加上去
        # 注意, 因为账户价值序列和基准序列的索引均为直接使用时间(除了下面要提到的, 计算cost ratio的时候),
        # 因此加上一期后并不会影响索引, 但是注意一旦要用cursor索引, 就需要留意!
        if cursor == 0:
            # 我们的账面价值序列，如果第一天就调仓（默认就是这种情况），最开始会不是初始资金，因此在第一行加入初始资金行
            # 初始资金这一行的时间设定为回测开始时间的前一秒
            base_time = self.bkt_start - pd.tseries.offsets.Second(1)
            base_value = pd.Series(self.initial_money, index=[base_time])
            # 拼接在一起
            self.account_value = pd.concat([base_value, self.account_value])
            # 拼接benchmark价值序列，本来第一项应当是回测开始那天的指数开盘价, 但是由于全收益指数没有开盘价,
            # 因此只能用第一天的收盘价替代, 即第一天基准指数的收益率一定是0
            benchmark_base_value = pd.Series(self.benchmark_value.iloc[0], index=[base_time])
            self.benchmark_value = pd.concat([benchmark_base_value, self.benchmark_value])

        # 计算每天的持股数
        self.info_series.ix[curr_time, 'holding_num'] = (self.real_vol_position.holding_matrix.
            ix[curr_time, :] != 0).sum()
        # 计算手续费所占总价值序列的比例, 注意是占上个交易日的账户价值的比例,
        # 注意, 因为分母是用上一期的账户价值的比例, 而账户价值序列刚好多了一期
        # 因此, 只需要用cursor索引就好, 不需要用cursor-1
        self.info_series.ix[curr_time, 'cost_ratio'] = (self.info_series.ix[curr_time, 'cost_value'] / \
                                                        self.account_value.ix[cursor])
        # 计算真实的股票仓位的持仓比例和目标持仓比例的差别
        self.info_series.ix[curr_time, 'holding_diff'] = self.real_pct_position.holding_matrix. \
            ix[curr_time, :].sub(self.tar_pct_position.holding_matrix.ix[curr_time, :]).abs().sum()
        # 计算真实的现金仓位的比例和目标的现金仓位比例的差别,
        # 注意, 现金的比例差别应该和股票的比例差别一致
        self.info_series.ix[curr_time, 'cash_diff'] = self.real_pct_position.cash.iloc[cursor] - \
            self.tar_pct_position.cash.iloc[cursor-1]



































































































































































