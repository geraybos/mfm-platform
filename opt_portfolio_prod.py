import sys, os
import numpy as np
import pandas as pd
from datetime import datetime

from strategy_data import strategy_data
from db_engine import db_engine
from optimizer_utility import optimizer_utility

# 生产系统中生成最优化持仓的类

class opt_portfolio_prod(object):
    """ This is the class which generates optimized portfolio in production system.

    foo
    """
    def __init__(self, *, date=datetime.now().date().strftime('%Y-%m-%d'), benchmark=None,
                 stock_pool=None, read_json=None, output_dir=os.path.abspath('.')):
        # 首先初始化当前日期, 这个日期会作为最后解出的优化组合的日期存入数据库, 作为这一天的目标持仓
        try:
            pd.Timestamp(date)
        except ValueError:
            raise ValueError('Please enter valid datetime string for date argument!\n')
        self.date = pd.Timestamp(date)
        # benchmark和stock pool必须在指定的范围内, 否则报错
        assert benchmark in ('hs300', 'zz500'), 'Benchmark argument is not valid!\n'
        assert stock_pool in ('all', 'hs300', 'zz500'), 'Stock pool argument is not valid!\n'
        # 目前来说, 如果投资域和benchmark不一样, 则投资域必须是全市场
        assert np.logical_or(benchmark==stock_pool, stock_pool=='all'), 'If benchmark and stock pool' \
            'are different, stock pool must be all for now!\n'
        self.benchmark = benchmark
        self.stock_pool = stock_pool
        # 读取数据的数据库相关信息的json文件路径
        assert read_json is not None, 'Please specify the directory of json file!\n'
        self.read_json = read_json
        # 写优化组合的json文件的路径
        self.output_dir = output_dir
        # 读取股票alpha的数据库引擎
        self.alpha_engine = None
        # 风险模型的数据库引擎
        self.rm_engine = None
        # 根据server type选择的sql alchemy支持的driver
        self.driver_map = {'mssql': 'pymssql', 'postgresql': 'psycopg2'}

        # 用于解优化的优化器
        self.optimizer = optimizer_utility()
        # 用于储存数据的数据类
        self.data = strategy_data()
        self.data.set_stock_pool(stock_pool)
        # 储存股票的alpha值
        self.alpha = None
        # 储存风险预测
        self.cov_mat = None
        self.spec_var = None

        # 优化器的条件
        self.opt_config = {}

    # 读取json文件, 建立读取alpha的数据库引擎
    def initialize_alpha_engine(self):
        json = pd.read_json(self.read_json, orient='index')
        # json的数据库连接信息里面必须要有取alpha的smart quant数据库的连接信息
        assert 'SmartQuant' in json.index, 'SmartQuant database is not in the json file!\n'
        sq = json.ix['SmartQuant', :]
        # 初始化数据库
        self.alpha_engine = db_engine(server_type=sq['serverType'], driver=self.driver_map[sq['serverType']],
            username=sq['user'], password=sq['password'], server_ip=sq['host'], port=str(sq['port']),
            db_name=sq['database'], add_info='')
        pass

    # 读取json文件, 建立读取risk model数据库的引擎
    def initialize_rm_engine(self):
        json = pd.read_json(self.read_json, orient='index')
        # json的数据库连接信息里面必须要有risk model数据库的连接信息
        assert 'RiskModel' in json.index, 'RiskModel database is not in the json file!\n'
        rm = json.ix['RiskModel', :]
        # 初始化数据库
        self.rm_engine = db_engine(server_type=rm['serverType'], driver=self.driver_map[rm['serverType']],
            username=rm['user'], password=rm['password'], server_ip=rm['host'], port=str(rm['port']),
            db_name=rm['database'], add_info='')
        pass

    # 取指定交易日的前一天, 即要用到的数据的那一天
    def get_former_day(self):
        sql = "select top 1 DataDate from ( " \
              "select distinct DataDate from RawData where DataDate < '" + str(self.date) + \
              "' ) temp order by DataDate desc"
        self.former_date = self.rm_engine.get_original_data(sql).iloc[0, 0]

    # 读取优化中要用到risk model数据库中的数据
    def read_rm_data(self):
        # 首先取raw data中的数据
        # 需要用到的数据为if tradable相关的标记数据, benchmark权重数据
        data_needed = ('is_enlisted', 'is_delisted', 'is_suspended', 'Weight_'+self.benchmark)
        sql_raw_data = "select * from RawData where DataDate = '" + str(self.former_date) + "' and " \
                       "DataName in " + str(data_needed)
        raw_data = self.rm_engine.get_original_data(sql_raw_data)
        raw_data = raw_data.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel().transpose(2, 1, 0)
        # 分配数据
        self.data.if_tradable = raw_data.ix[['is_enlisted', 'is_delisted', 'is_suspended']]
        self.data.benchmark_price = raw_data.ix[['Weight_'+self.benchmark]]
        # 生成可交易, 投资域相关的标记
        self.data.generate_if_tradable()
        self.data.handle_stock_pool()

        # 取因子暴露数据
        sql_expo = "select * from BarraBaseFactorExpo where DataDate = '" + str(self.former_date) + \
                   "' and StockPool = '" + str(self.stock_pool) + "' "
        expo = self.rm_engine.get_original_data(sql_expo)
        expo = expo.pivot_table(index='SecuCode', columns='FactorName', values='Value', aggfunc='first')
        # 取行业标记数据
        sql_industry = "select * from IndustryMark where DataDate = '" + str(self.former_date) + "' "
        industry = self.rm_engine.get_original_data(sql_industry)
        industry = industry.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                        aggfunc='first').squeeze()
        industry_expo = pd.get_dummies(industry, prefix='Industry')
        # 将因子暴露数据粘在一起
        total_expo = pd.concat([expo, industry_expo, pd.Series(1, index=industry_expo.index,
                                                               name='country_factor')], axis=1)
        self.data.factor_expo = pd.Panel({self.former_date: total_expo}).transpose(2, 0, 1)

        # 取因子协方差矩阵和股票特定风险的数据
        sql_covmat = "select * from BarraBaseCovMat where DataDate = '" + str(self.former_date) + \
                     "' and StockPool = '" + str(self.stock_pool) + "' "
        covmat = self.rm_engine.get_original_data(sql_covmat)
        covmat = covmat.pivot_table(index='FactorName1', columns='FactorName2', values='Value')
        self.cov_mat = covmat.reindex(index=self.data.factor_expo.items, columns=self.data.factor_expo.items)
        sql_specvar = "select * from BarraBaseSpecVar where DataDate = '" + str(self.former_date) + \
                      "' and StockPool = '" + str(self.stock_pool) + "' "
        specvar = self.rm_engine.get_original_data(sql_specvar)
        specvar = specvar.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                      aggfunc='first').squeeze()
        self.spec_var = specvar
        pass

    # 读取alpha数据的函数
    def read_alpha(self):
        sql_alpha = "select runnerdate as DataDate, stockticker as SecuCode, value as Value " \
                    "from RunnerValue where runnerdate = '" + str(self.former_date) + "' and " \
                    "runnerid = 63 order by DataDate, SecuCode "
        alpha = self.alpha_engine.get_original_data(sql_alpha)
        alpha = alpha.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                  aggfunc='first').squeeze()
        self.alpha = alpha

    # 处理数据
    def prepare_data(self):
        # 股票索引取if_tradable中的股票索引和alpha的股票索引的并集
        # 注意, if tradable和benchmark price的股票索引是一样的, 因为是一起取出来的
        all_stocks = self.data.if_tradable.minor_axis.union(self.alpha.index)
        # 其他的数据都重索引到这个并集上去, 之所以alpha需要fillna(0), 是因为下一步中,
        # 是通过对alpha dropna来得到最终的股票索引, 而dropna这一步实际要做的是, 是想把
        # 不在投资域的股票数据去除掉, 如果有股票在投资域中, 但不在alpha中, 对alpha reindex后,
        # 数据会变成nan, 然后会被dropna掉, 此时对benchmark weight进行reindex, 会把这只股票扔掉
        # 因此这里fillna是为了下一步dropna中被drop掉的一定是不在投资域内的股票
        self.data.if_tradable = self.data.if_tradable.reindex(minor_axis=all_stocks)
        self.alpha = self.alpha.reindex(index=all_stocks).fillna(0)

        # 将alpha中不可投资的数据变成nan, 去除nan后, 之后的数据都以这个alpha的股票索引作为索引
        self.alpha = self.alpha.where(self.data.if_tradable.ix['if_inv', 0, :], np.nan).dropna()
        self.data.if_tradable = self.data.if_tradable.reindex(minor_axis=self.alpha.index)
        self.data.benchmark_price = self.data.benchmark_price.reindex(minor_axis=self.alpha.index)
        self.data.factor_expo = self.data.factor_expo.reindex(minor_axis=self.alpha.index)
        self.spec_var = self.spec_var.reindex(index=self.alpha.index)
        # 最后对cov mat重索引, 使得cov mat的因子顺序和factor expo的因子顺序一样, 因此,
        # factor expo一定要保持风格因子, 行业因子, 国家因子的顺序存放
        self.cov_mat = self.cov_mat.reindex(index=self.data.factor_expo.items,
                                            columns=self.data.factor_expo.items)

    # 解优化
    def solve_opt_portfolio(self):
        # 从panel中取数据, 将数据变成dataframe, 注意factor expo是行为因子, 列为股票
        curr_benchmark_weight = self.data.benchmark_price.ix['Weight_'+self.benchmark, 0, :]
        curr_factor_expo = self.data.factor_expo.ix[:, 0, :].T

        # 首先看是否有风险厌恶系数的设置
        if 'cov_risk_aversion' in self.opt_config.keys():
            self.optimizer.set_risk_aversion(cov_risk_aversion=self.opt_config['cov_risk_aversion'])
        if 'spec_risk_aversion' in self.opt_config.keys():
            self.optimizer.set_risk_aversion(spec_risk_aversion=self.opt_config['spec_risk_aversion'])

        # 看是否有行业中性的限制条件, 默认行业中性=True
        if 'indus_neutral' in self.opt_config.keys():
            indus_neutral = self.opt_config['indus_neutral']
        else:
            indus_neutral = True


        ################################################################################################
        # 因子暴露, 交易费用, 换手率, 全投资, 现金比率, 单个资产上下限的限制条件暂时不加
        ################################################################################################

        # 如果进行行业中性
        if indus_neutral:
            # 首先判断风格因子和行业因子的个数
            industry = self.data.factor_expo.items.str.startswith('Industry')
            n_indus = industry[industry].size
            n_styles = self.data.factor_expo.shape[0] - n_indus - 1
            # 行业因子的因子名
            indus_name = self.data.factor_expo.items[n_styles:(n_styles+n_indus)]
            indus_cons = pd.DataFrame(indus_name.values, columns=['factor'])
            indus_cons['if_eq'] = True
            indus_cons['if_lower_bound'] = True
            indus_cons['limit'] = 0
            # 如果这一期, 某个行业的所有股票都是0暴露, 则在行业限制中去除这个行业
            empty_indus = curr_factor_expo[n_styles:(n_styles+n_indus)].sum(1) == 0
            indus_cons = indus_cons[np.logical_not(empty_indus.values)]
            enable_full_inv_cons = False
        else:
            indus_cons = None
            enable_full_inv_cons = True

        # 解优化组合
        self.optimizer.solve_optimization(curr_benchmark_weight, curr_factor_expo, self.cov_mat,
            residual_return=self.alpha, specific_var=self.spec_var, factor_expo_cons=indus_cons,
            enable_full_inv_cons=enable_full_inv_cons)

        pass



if __name__ == '__main__':
    test = opt_portfolio_prod(benchmark='zz500', stock_pool='zz500',
                              read_json=os.path.abspath('.')+'/dbuser.txt')
    test.initialize_alpha_engine()
    test.initialize_rm_engine()
    test.get_former_day()
    test.read_rm_data()
    test.read_alpha()
    test.prepare_data()
    import time
    start_time = time.time()
    test.solve_opt_portfolio()
    print("time: {0} seconds\n".format(time.time()-start_time))





