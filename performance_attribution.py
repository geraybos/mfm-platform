#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:50:11 2017

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

from data import data
from strategy_data import strategy_data
from position import position
from barra_base import barra_base
from factor_base import factor_base

# 业绩归因类，对策略中的股票收益率（注意：并非策略收益率）进行归因

class performance_attribution(object):
    """This is the class for performance attribution analysis.

    foo
    """

    def __init__(self, input_position, portfolio_returns, *, benchmark_weight=None,
                 intra_holding_deviation=None, trans_cost=None, show_warning=True):
        self.pa_position = position(input_position.holding_matrix)
        # 如果传入基准持仓数据，则归因超额收益
        if isinstance(benchmark_weight, pd.DataFrame):
            # 一些情况下benchmark的权重和不为1（一般为差一点），为了防止偏差，这里重新归一化
            # 同时将时间索引控制在回测期间内
            new_benchmark_weight = benchmark_weight.reindex(self.pa_position.holding_matrix.index).\
                apply(lambda x:x if (x==0).all() else x.div(x.sum()), axis=1)
            self.pa_position.holding_matrix = input_position.holding_matrix.sub(new_benchmark_weight, fill_value=0)
            # 提示用户, 归因变成了对超额部分的归因
            print('Note that with benchmark_weight being passed, the performance attribution will be based on the '
                  'active part of the portfolio against the benchmark. Please make sure that the portfolio returns '
                  'you passed to the pa is the corresponding active return! \n')
        else:
            self.pa_position.holding_matrix = input_position.holding_matrix

        # 检测是否要对现金部分进行归因，设置一个检测现金部分的阈值
        self.pa_position.cash = input_position.cash
        if (self.pa_position.cash >= 1e-6).any():
            self.do_cash_pa = True
        else:
            self.do_cash_pa = False

        # 传入的组合收益, 由于传入的组合收益是对数收益，而归因基于简单收益
        # 因此，要将对数收益转换成简单收益
        # 归因使用简单收益的原因是, 归因涉及同一时间上不同资产的组合加总，简单收益更为准确
        self.port_returns = portfolio_returns
        self.port_returns = np.exp(self.port_returns).sub(1.0)

        # 有可能被传入的调仓期间的偏离, 即由于调仓期内多头与空头涨幅不一致带来的组合价值的偏离,
        # 这会导致country factor的暴露不总是0, 只有在使用真实世界的超额归因的时候, 才会用到这个数据
        self.intra_holding_deviation = intra_holding_deviation
        # 如果传入了这个参数, 但确做的不是超额归因, 要报错
        if isinstance(intra_holding_deviation, pd.Series) and benchmark_weight is None:
            print('Warning: intra-holding deviation be passed but no benchmark weight. The pa system '
                  'has automatically ignored the intra-holding deviation, since it is implied by'
                  'no benchmark weight being passed that the pa will not be based on active part! \n')
            self.intra_holding_deviation = None
        # 有可能传入的手续费带来的负收益序列, 在真实归因时用到
        # 如果传入了手续费序列, 但是其都是0, 则自动不画手续费曲线
        self.trans_cost = trans_cost
        if isinstance(self.trans_cost, pd.Series) and (self.trans_cost == 0).all():
            print('Warning: transaction cost series has been passed but is an all-zeros series. The pa'
                  'system has automatically ignored this series, since it is implied that there is no '
                  'transaction cost! \n')
            self.trans_cost = None

        self.pa_returns = pd.DataFrame()
        self.port_expo = pd.DataFrame()
        self.port_pa_returns = pd.DataFrame()
        self.style_factor_returns = pd.Series()
        self.industry_factor_returns = pd.Series()
        self.country_factor_returns = pd.Series()
        self.residual_returns = pd.Series()
        # 业绩归因为基于barra因子的业绩归因
        self.base = barra_base()

        self.discarded_stocks_num = pd.DataFrame()
        self.discarded_stocks_wgt = pd.DataFrame()
        self.show_warning = show_warning

    # 建立barra因子库，有些时候可以直接用在其他地方（如策略中）已计算出的barra因子库，就可以不必计算了
    def construct_base(self, *, outside_base=None, base_stock_pool='all'):
        if isinstance(outside_base, factor_base):
            self.base = outside_base
            print('PA system has successfully got base from outside! Please note that the base from '
                  'outside should at least contain basic data.\n')
            # 外部的base，如果没有factor expo则需要读取或再次计算, 股票池则用外部base的股票池
            if self.base.base_data.factor_expo.empty:
                    self.base.construct_factor_base()
        else:
            self.base.base_data.stock_pool = base_stock_pool
            self.base.construct_factor_base()

    # 进行业绩归因
    # 用discard_factor可以定制用来归因的因子，将不需要的因子的名字或序号以list写入即可
    # 注意，只能用来删除风格因子，不能用来删除行业因子或country factor
    def get_pa_return(self, *, discard_factor=(), enable_read_pa_return=False):
        # 如果有储存的因子收益, 且没有被丢弃的因子, 则读取储存在本地的因子
        if os.path.isfile('bb_factor_return_'+self.base.base_data.stock_pool+'.csv') and \
                        len(discard_factor) == 0 and enable_read_pa_return:
            self.pa_returns = data.read_data('bb_factor_return_'+self.base.base_data.stock_pool)
            print('PA system has successfully read base return from local files! \n')
        else:
            # 将被删除的风格因子的暴露全部设置为0
            self.base.base_data.factor_expo.ix[list(discard_factor), :, :] = 0
            # 再次将不能交易的值设置为nan
            self.base.base_data.discard_uninv_data()
            # 建立储存因子收益的dataframe
            self.pa_returns = pd.DataFrame(0, index=self.base.base_data.factor_expo.major_axis,
                                           columns=self.base.base_data.factor_expo.items)
            # 计算barra base因子的因子收益
            self.base.get_base_factor_return()
            # barra base因子的因子收益即是归因的因子收益
            self.pa_returns = self.base.base_factor_return

            # 将回归得到的因子收益储存在本地, 每次更新了新的数据都要重新回归后储存一次
            self.pa_returns.to_csv('bb_factor_return_'+self.base.base_data.stock_pool+'.csv',
                                   index_label='datetime', na_rep='NaN', encoding='GB18030')

        # 将pa_returns的时间轴改为业绩归因的时间轴（而不是base的时间轴）
        self.pa_returns = self.pa_returns.reindex(self.pa_position.holding_matrix.index)

    # 将收益归因的结果进行整理
    def analyze_pa_return_outcome(self):
        # 首先将传入的要归因的持仓矩阵的代码重索引为base factor的股票代码
        # 注意这里之后需要加一个像回测里那样的检查持仓矩阵里的股票代码是否都在base factor的股票代码中
        # 因为如果不这样可能会遗失掉某些股票
        self.pa_position.holding_matrix = self.pa_position.holding_matrix.reindex(
            columns=self.base.base_data.factor_expo.minor_axis, fill_value=0.0)

        # 首先根据持仓比例计算组合在各个因子上的暴露
        # 计算组合暴露需要用专门定制的函数来进行修正计算, 不能简单的用持仓矩阵乘以因子暴露
        self.port_expo = strategy_data.get_port_expo(self.pa_position.holding_matrix,
            self.base.base_data.factor_expo, self.base.base_data.if_tradable, show_warning=self.show_warning)

        # 根据因子收益和因子暴露计算组合在因子上的收益，注意因子暴露用的是组合上一期的因子暴露
        # 为了使得每个因子第一天的收益都从0开始, 因此将第一天的nan填成0
        self.port_pa_returns = self.pa_returns.mul(self.port_expo.shift(1)).fillna(0.0)

        # 将组合因子收益和因子暴露数据重索引为pa position的时间（即持仓区间），原时间为barra base的区间
        self.port_expo = self.port_expo.reindex(self.pa_position.holding_matrix.index)
        self.port_pa_returns = self.port_pa_returns.reindex(self.pa_position.holding_matrix.index)

        # 注意, 如果进行的是真实世界的超额收益归因, 如果传入了偏离度序列, 则要将country factor因子的暴露重新
        # 调整为偏离度序列, 然后重新计算收益, 注意, 这里的调整是原始的暴露加上偏离度序列,
        # 因为原始的暴露可能因为调仓日和起始回测日的原因, 导致最开始的几天是-1.
        # 同样要记得把第一天的nan填成0
        if isinstance(self.intra_holding_deviation, pd.Series):
            self.port_expo['country_factor'] += self.intra_holding_deviation
            self.port_pa_returns['country_factor'] = self.pa_returns['country_factor'].\
                mul(self.port_expo['country_factor'].shift(1)).fillna(0)

        if self.do_cash_pa:
            # 首先要将在base中的无风险收益的时间索引重索引为归因时间段
            self.risk_free_rate_simple = self.base.base_data.const_data['RiskFreeRate_simple'].\
                reindex(index=self.pa_position.holding_matrix.index)
            # 现在来根据现金资产的存在对收益归因进行调整，首先，组合的暴露全部被现金比例等比例缩小
            self.port_expo = self.port_expo.mul(1 - self.pa_position.cash, axis=0)
            # 组合在因子上的收益，也要进行一样的调整，因为组合的暴露被等比例缩小了
            self.port_pa_returns = self.port_pa_returns.mul(1 - self.pa_position.cash, axis=0)
            # 现金部分的收益，应该等于上一期的现金因子的暴露，乘以无风险利率
            # 为了保证第一天的收益都到residual上去，第一天的现金收益也要fillna成0
            self.cash_returns = self.pa_position.cash.shift(1).mul(self.risk_free_rate_simple).fillna(0.0)

        # 计算各类因子的总收益情况
        # 注意, 由于计算组合收益的时候, 组合暴露要用上一期的暴露, 因此第一期统一没有因子收益
        # 这一部分收益会被归到residual return中去, 从而提升residual return
        # 而fillna是为了确保这部分收益会到residual中去, 否则residual会变成nan, 从而丢失这部分收益
        # 风格因子收益
        self.style_factor_returns = self.port_pa_returns.ix[:, 0:self.base.n_style].sum(1)
        # 行业因子收益
        self.industry_factor_returns = self.port_pa_returns.ix[:,
                                       self.base.n_style:(self.base.n_style+self.base.n_indus)].sum(1)
        # 国家因子收益
        self.country_factor_returns = self.port_pa_returns.ix[:,
                                      (self.base.n_style+self.base.n_indus)].fillna(0.0)

        # 残余收益，即alpha收益，为组合收益减去之前那些因子的收益
        # 注意下面会提到，缺失数据会使得残余收益变大
        self.residual_returns = self.port_returns - (self.style_factor_returns+self.industry_factor_returns+
                                                     self.country_factor_returns)
        # 如果有手续费序列, 则残余收益应当是减去手续费的(手续费为负, 因此残余收益会变大)
        if isinstance(self.trans_cost, pd.Series):
            self.residual_returns -= self.trans_cost
        pass
        # 如果要做现金部分的归因，则还需要减去现金
        if self.do_cash_pa:
            self.residual_returns -= self.cash_returns

        # 与风险不同，归因最后画出的图是对累计收益曲线进行分解，因此还需要计算各个部分的累计曲线
        # 首先计算组合的累计净值，之所以要先算净值，是因为净值是作为之后算各个部分累计收益时的基数
        port_nav = self.port_returns.add(1).cumprod()
        # 算各个部分的累计收益贡献，这一期的简单收益（或净值增长）乘以上一期的组合净值基数
        # 就可以得到这一期该部分取得的收益，注意，fillna是为了让第一期的各个部分的收益都跑到residual那里去
        self.port_pa_returns_cum = self.port_pa_returns.mul(port_nav.shift(1), axis=0).fillna(0).cumsum()
        self.style_factor_returns_cum = self.port_pa_returns_cum.ix[:, 0:self.base.n_style].sum(1)
        self.industry_factor_returns_cum = self.port_pa_returns_cum.ix[:,
                                           self.base.n_style:(self.base.n_style+self.base.n_indus)].sum(1)
        self.country_factor_returns_cum = self.port_pa_returns_cum.ix[:, (self.base.n_style+self.base.n_indus)]
        # 残余收益的部分，第一期要填成自身第一期的收益
        self.residual_returns_cum = self.residual_returns.mul(port_nav.shift(1))
        self.residual_returns_cum.iloc[0] = self.residual_returns.iloc[0]
        self.residual_returns_cum = self.residual_returns_cum.cumsum()
        # 手续费方面，由于手续费的cost_ratio是占上一个交易日的账户价值的比例（见backtest）
        # 因此计算方法也是一样的，只不过port_nav在shift过后，将第一期的nan填成1，以表示最一开始的资产净值是1
        if isinstance(self.trans_cost, pd.Series):
            self.trans_cost_cum = self.trans_cost.mul(port_nav.shift(1).fillna(1)).cumsum()
        # 对于现金来讲, 由于现金在t日的收益, 实际上是昨日持有现金比例的隔夜收益,
        # 因此第一天事实上考虑的是刚刚拿到资产进场的那天, 所以不存在昨日的隔夜收益, 因此第一项填成0
        if self.do_cash_pa:
            self.cash_returns_cum = self.cash_returns.mul(port_nav.shift(1)).fillna(0).cumsum()
        # 最后计算组合的累计简单收益，以做参考
        self.port_returns_cum = port_nav.sub(1)


    # 进行风险归因
    def analyze_pa_risk_outcome(self):
        # 在每个时间点上, 用过去能得到的所有因子收益率序列来计算因子收益率的波动率
        # 并没有用加权的方法来估计波动率, 因为归因是对组合的历史波动率归因,
        # 计算历史波动率时, 并不加权, 而是进行回溯的等权计算
        # 注意, 由于波动选取的窗口为21个交易日, 因此序列最开始的前20个交易日的数据默认都填成0
        self.pa_sigma = self.pa_returns.expanding(min_periods=21).std().fillna(0.0)
        # 同样的, 在每个时间点上, 用过去能得到的所有因子收益率序列, 和组合的收益率序列来计算相关系数
        self.pa_corr = self.pa_returns.expanding(min_periods=21).corr(self.port_returns).fillna(0.0)

        # 根据barra: risk contribution = exposure * volatility * correlation 的框架,
        # 将风险贡献归因到风格, 行业, 国家因子, 以及残余部分上去, 详细的方法见barra文档.
        # 首先将每个小因子的风险贡献部分计算出来, 注意, 与收益归因一样, 暴露同样是要用上一期的暴露
        # 将第一天的nan也填成0, 保证风险前20个交易日都是0
        self.port_pa_risks = (self.port_expo.shift(1) * self.pa_sigma * self.pa_corr).fillna(0.0)
        # 然后根据因子分类, 将风险贡献分配到不同的因子类型上去
        self.style_factor_risks = self.port_pa_risks.ix[:, 0:self.base.n_style].sum(1)
        self.industry_factor_risks = self.port_pa_risks.ix[:,
                                     self.base.n_style:(self.base.n_style+self.base.n_indus)].sum(1)
        self.country_factor_risks = self.port_pa_risks.ix[:,
                                   (self.base.n_style+self.base.n_indus)].fillna(0.0)

        # 残余收益贡献的风险, 由组合总风险减去已经归因的风险得到
        # 计算组合的风险
        self.port_risks = self.port_returns.expanding(min_periods=21).std().fillna(0.0)
        # 残余收益贡献的风险
        self.residual_risks = self.port_risks - (self.style_factor_risks + self.industry_factor_risks
                                                 + self.country_factor_risks)

        # 为了保证风险归因的真实性, 如果有手续费在, 需要计算手续费对组合风险的影响
        if isinstance(self.trans_cost, pd.Series):
            # 计算手续费的风险
            self.trans_cost_sigma = self.trans_cost.expanding(min_periods=21).std().fillna(0)
            # 计算手续费与组合的相关系数
            self.trans_cost_corr = self.trans_cost.expanding(min_periods=21).corr(self.port_returns).fillna(0.0)
            # 默认手续费暴露是1, 计算手续费的风险贡献
            self.trans_cost_risks = self.trans_cost_sigma * self.trans_cost_corr
            # 于是, 残余风险要减去手续费的风险, 即将手续费的风险贡献从残余收益风险贡献中剥离出去
            self.residual_risks -= self.trans_cost_risks
        # 如果要做现金部分的归因，还需要计算现金部分的收益对组合风险的影响
        if self.do_cash_pa:
            # 现金的风险
            self.cash_sigma = self.risk_free_rate_simple.expanding(min_periods=21).std().fillna(0.0)
            # 现金与组合的相关系数
            self.cash_corr = self.risk_free_rate_simple.expanding(min_periods=21).corr(self.port_returns).fillna(0.0)
            # 计算现金部分的风险贡献
            self.cash_risks = (self.pa_position.cash.shift(1) * self.cash_sigma * self.cash_corr).fillna(0.0)
            # 残余风险要减去现金部分的风险
            self.residual_risks -= self.cash_risks

    # 处理那些没有归因的股票，即有些股票被策略选入，但因没有因子暴露值，而无法纳入归因的股票
    # 此dataframe处理这些股票，储存每期这些股票的个数，以及它们在策略中的持仓权重
    # 注意，此类股票的出现必然导致归因的不准确，因为它们归入到了组合总收益中，但不会被归入到缺少暴露值的因子收益中，因此进入到残余收益中
    # 这样不仅会使得残余收益含入因子收益，而且使得残余收益与因子收益之间具有显著相关性
    # 如果这样暴露缺失的股票比例很大，则使得归因不具有参考价值
    def handle_discarded_stocks(self, *, foldername=''):
        self.discarded_stocks_num = self.pa_returns.mul(0)
        self.discarded_stocks_wgt = self.pa_returns.mul(0)
        # 因子暴露有缺失值，没有参与归因的股票
        if_discarded = self.base.base_data.factor_expo.reindex(major_axis=self.pa_position.holding_matrix.index).isnull()
        # 没有参与归因，同时还持有了
        discarded_and_held = if_discarded.mul(self.pa_position.holding_matrix.fillna(0), axis='items').astype(bool)
        # 各个因子没有参与归因的股票个数与持仓比例
        self.discarded_stocks_num = discarded_and_held.sum(2)
        # 注意：如果有benchmark传入，则持仓为负数，这时为了反应绝对量，持仓比例要取绝对值
        self.discarded_stocks_wgt = discarded_and_held.mul(self.pa_position.holding_matrix, axis='items').abs().sum(2)
        # 计算总数
        self.discarded_stocks_num['total'] = self.discarded_stocks_num.sum(1)
        self.discarded_stocks_wgt['total'] = self.discarded_stocks_wgt.sum(1)

        # 循环输出警告
        if self.show_warning:
            for time, temp_data in self.discarded_stocks_num.iterrows():
                # 一旦没有归因的股票数超过总持股数的100%，或其权重超过100%，则输出警告
                if temp_data.ix['total'] >= 1*((self.pa_position.holding_matrix.ix[time] != 0).sum()) or \
                self.discarded_stocks_wgt.ix[time, 'total'] >= 1:
                    print('At time: {0}, the number of stocks(*discarded times) held but discarded in performance attribution '
                          'is: {1}, the weight of these stocks(*discarded times) is: {2}.\nThus the outcome of performance '
                          'attribution at this time can be significantly distorted. Please check discarded_stocks_num and '
                          'discarded_stocks_wgt for more information.\n'.format(time, temp_data.ix['total'],
                                                                                self.discarded_stocks_wgt.ix[time, 'total']))
        # 输出总的缺失情况：
        target_str = 'The average number of stocks(*discarded times) held but discarded in the pa is: {0}, \n' \
                     'the weight of these stocks(*discarded times) is: {1}.\n'.format(
                             self.discarded_stocks_num['total'].mean(), self.discarded_stocks_wgt['total'].mean())
        print(target_str)
        # 将输出写到txt中
        with open(str(os.path.abspath('.'))+'/'+foldername+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)

    # 进行画图
    def plot_performance_attribution(self, foldername='', pdfs=None):
        self.plot_pa_return(foldername=foldername, pdfs=pdfs)
        self.plot_pa_risk(foldername=foldername, pdfs=pdfs)


    # 对收益归因的结果进行画图
    def plot_pa_return(self, *, foldername='', pdfs=None):
        # 处理中文图例的字体文件
        from matplotlib.font_manager import FontProperties
        # chifont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
        chifont = FontProperties(fname=str(os.path.abspath('.'))+'/ResearchData/华文细黑.ttf')
        
        # 第一张图分解组合的累计收益来源
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.plot(self.style_factor_returns_cum*100, label='style')
        plt.plot(self.industry_factor_returns_cum*100, label='industry')
        plt.plot(self.country_factor_returns_cum*100, label='country')
        plt.plot(self.residual_returns_cum*100, label='residual')
        # 如果有手续费序列, 则画出手续费序列
        if isinstance(self.trans_cost, pd.Series):
            plt.plot(self.trans_cost_cum*100, label='trans_cost')
        # 如果有现金序列, 则画出现金序列
        if self.do_cash_pa:
            plt.plot(self.cash_returns_cum*100, label='cash')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Simple Return (%)')
        ax1.set_title('The Cumulative Simple Return of Factor Groups')
        ax1.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_RetSource.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第二张图分解组合的累计风格收益
        f2 = plt.figure()
        ax2 = f2.add_subplot(1,1,1)
        plt.plot((self.port_pa_returns_cum.ix[:, 0:self.base.n_style]*100))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Simple Return (%)')
        ax2.set_title('The Cumulative Simple Return of Style Factors')
        ax2.legend(self.port_pa_returns.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_CumRetStyle.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第三张图分解组合的累计行业收益
        # 行业图示只给出最大和最小的5个行业
        # 当前的有效行业数
        valid_indus = self.pa_returns.iloc[:, self.base.n_style:(self.base.n_style+self.base.n_indus)].\
            dropna(axis=1, how='all').shape[1]
        if valid_indus<=10:
            qualified_rank = [i for i in range(1, valid_indus+1)]
        else:
            part1 = [i for i in range(1, 6)]
            part2 = [j for j in range(valid_indus, valid_indus-5, -1)]
            qualified_rank = part1+part2
        f3 = plt.figure()
        ax3 = f3.add_subplot(1, 1, 1)
        indus_rank = self.port_pa_returns_cum.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)]. \
            iloc[-1].rank(ascending=False)
        for i, j in enumerate(self.port_pa_returns_cum.ix[:,
                              self.base.n_style:(self.base.n_style+self.base.n_indus)].columns):
            if indus_rank[j] in qualified_rank:
                plt.plot((self.port_pa_returns_cum.ix[:, j] * 100), label=j+str(indus_rank[j]))
            else:
                plt.plot((self.port_pa_returns_cum.ix[:, j] * 100), label='_nolegend_')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Cumulative Simple Return (%)')
        ax3.set_title('The Cumulative Simple Return of Industrial Factors')
        ax3.legend(loc='best', prop=chifont, bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_CumRetIndus.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第四张图画组合的累计风格暴露
        f4 = plt.figure()
        ax4 = f4.add_subplot(1, 1, 1)
        plt.plot(self.port_expo.ix[:, 0:self.base.n_style].cumsum(0))
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Cumulative Factor Exposures')
        ax4.set_title('The Cumulative Style Factor Exposures of the Portfolio')
        ax4.legend(self.port_expo.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_CumExpoStyle.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第五张图画组合的累计行业暴露
        f5 = plt.figure()
        ax5 = f5.add_subplot(1, 1, 1)
        # 累计暴露最大和最小的5个行业
        indus_rank = self.port_expo.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)]. \
            cumsum(0).ix[-1].rank(ascending=False)
        for i, j in enumerate(self.port_expo.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)].columns):
            if indus_rank[j] in qualified_rank:
                plt.plot((self.port_expo.ix[:, j].cumsum(0)), label=j+str(indus_rank[j]))
            else:
                plt.plot((self.port_expo.ix[:, j].cumsum(0)), label='_nolegend_')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Cumulative Factor Exposures')
        ax5.set_title('The Cumulative Industrial Factor Exposures of the Portfolio')
        ax5.legend(loc='best', prop=chifont, bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_CumExpoIndus.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第六张图画组合的每日风格暴露
        f6 = plt.figure()
        ax6 = f6.add_subplot(1, 1, 1)
        plt.plot(self.port_expo.ix[:, 0:self.base.n_style])
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Factor Exposures')
        ax6.set_title('The Style Factor Exposures of the Portfolio')
        ax6.legend(self.port_expo.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_ExpoStyle.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第七张图画组合的每日行业暴露
        f7 = plt.figure()
        ax7 = f7.add_subplot(1, 1, 1)
        # 平均暴露最大和最小的5个行业
        indus_rank = self.port_expo.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)]. \
            mean(0).rank(ascending=False)
        for i, j in enumerate(self.port_expo.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)].columns):
            if indus_rank[j] in qualified_rank:
                plt.plot((self.port_expo.ix[:, j]), label=j+str(indus_rank[j]))
            else:
                plt.plot((self.port_expo.ix[:, j]), label='_nolegend_')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Factor Exposures')
        ax7.set_title('The Industrial Factor Exposures of the Portfolio')
        ax7.legend(loc='best', prop=chifont, bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_ExpoIndus.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第八张图画用于归因的base的风格因子的纯因子收益率，即回归得到的因子收益率，仅供参考
        f8 = plt.figure()
        ax8 = f8.add_subplot(1, 1, 1)
        plt.plot(self.pa_returns.ix[:, 0:self.base.n_style].add(1).cumprod(0).sub(1)*100)
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Cumulative Simple Return (%)')
        ax8.set_title('The Cumulative Simple Return of Pure Style Factors Through Regression')
        ax8.legend(self.pa_returns.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_PureStyleFactorRet.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

    # 对风险归因的结果进行画图
    def plot_pa_risk(self, *, foldername='', pdfs=None):
        # 处理中文图例的字体文件
        from matplotlib.font_manager import FontProperties
        # chifont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
        chifont = FontProperties(fname=str(os.path.abspath('.'))+'/华文细黑.ttf')

        # 第一张图分解组合的风险来源
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.plot(self.style_factor_risks*100, label='style')
        plt.plot(self.industry_factor_risks*100, label='industry')
        plt.plot(self.country_factor_risks*100, label='country')
        plt.plot(self.residual_risks*100, label='residual')
        # 如果有手续费序列, 则画出手续费序列
        if isinstance(self.trans_cost, pd.Series):
            plt.plot(self.trans_cost_risks*100, label='trans_cost')
        # 如果对现金做归因，则画出现金序列
        if self.do_cash_pa:
            plt.plot(self.cash_risks*100, label='cash')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Volatility Simple Return (%)')
        ax1.set_title('The Volatility of Simple Return of Factor Groups')
        ax1.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_RiskSource.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第二张图分解组合的风格风险贡献
        f2 = plt.figure()
        ax2 = f2.add_subplot(1, 1, 1)
        plt.plot((self.port_pa_risks.ix[:, 0:self.base.n_style] * 100))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Volatility of Simple Return (%)')
        ax2.set_title('The Volatility of Simple Return of Style Factors')
        ax2.legend(self.port_pa_risks.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_RiskStyle.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第三张图分解组合的行业风险贡献
        # 行业图示只给出最大和最小的5个行业
        # 当前的有效行业数
        valid_indus = self.pa_returns.iloc[:, self.base.n_style:(self.base.n_style+self.base.n_indus)].\
            dropna(axis=1, how='all').shape[1]
        if valid_indus<=10:
            qualified_rank = [i for i in range(1, valid_indus+1)]
        else:
            part1 = [i for i in range(1, 6)]
            part2 = [j for j in range(valid_indus, valid_indus-5, -1)]
            qualified_rank = part1+part2
        f3 = plt.figure()
        ax3 = f3.add_subplot(1, 1, 1)
        indus_rank = self.port_pa_risks.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)]. \
            mean(0).rank(ascending=False)
        for i, j in enumerate(self.port_pa_risks.ix[:, self.base.n_style:(self.base.n_style+self.base.n_indus)].columns):
            if indus_rank[j] in qualified_rank:
                plt.plot((self.port_pa_risks.ix[:, j] * 100), label=j+str(indus_rank[j]))
            else:
                plt.plot((self.port_pa_risks.ix[:, j] * 100), label='_nolegend_')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Volatility of Simple Return (%)')
        ax3.set_title('The Volatility of Simple Return of Industrial Factors')
        ax3.legend(loc='best', prop=chifont, bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/PA_RiskIndus.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第四张图画用于归因的base的风格因子纯因子收益率的波动率，以供参考
        # 注意, 根据barra的风险归因模型, 因子的收益率的波动率, 以及因子的收益率与组合收益率的相关系数(下图)
        # 是影响风险贡献的因素, 两者相乘再乘以因子暴露, 即可得到因子的风险贡献
        f4 = plt.figure()
        ax4 = f4.add_subplot(1, 1, 1)
        plt.plot(self.pa_sigma.ix[:, 0:self.base.n_style] * 100)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Volatility of Simple Return (%)')
        ax4.set_title('The Standalone Volatility of Style Factors Simple Return')
        ax4.legend(self.port_pa_risks.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_PureStyleFactorVol.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

        # 第五张图画用于归因的base的风格因子纯因子收益率与组合收益的相关系数，以供参考
        f5 = plt.figure()
        ax5 = f5.add_subplot(1, 1, 1)
        plt.plot(self.pa_corr.ix[:, 0:self.base.n_style])
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Correlation Coefficient')
        ax5.set_title('The Corr Between Style Factors Simple Return and Portfolio Simple Return')
        ax5.legend(self.port_pa_risks.columns[0:self.base.n_style], loc='best', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=30)
        plt.grid()
        plt.savefig(str(os.path.abspath('.')) + '/' + foldername + '/PA_PureStyleFactorCorr.png', dpi=1200,
                    bbox_inches='tight')
        if isinstance(pdfs, PdfPages):
            plt.savefig(pdfs, format='pdf', bbox_inches='tight')

    # 进行业绩归因
    def execute_performance_attribution(self, *, outside_base=None, base_stock_pool='all', discard_factor=(),
                                        foldername='', enable_read_pa_return=False):
        self.construct_base(outside_base=outside_base, base_stock_pool= base_stock_pool)
        self.get_pa_return(discard_factor=discard_factor, enable_read_pa_return=enable_read_pa_return)
        self.analyze_pa_return_outcome()
        self.analyze_pa_risk_outcome()
        self.handle_discarded_stocks(foldername=foldername)





























































































































































































































