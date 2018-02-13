import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

from strategy_data import strategy_data
from db_engine import db_engine
from optimizer_utility import optimizer_utility

# 生产系统中生成最优化持仓的类

class opt_portfolio_prod(object):
    """ This is the class which generates optimized portfolio in production system.

    foo
    """
    def __init__(self, *, date=datetime.now().date().strftime('%Y-%m-%d'), benchmark=None,
                 stock_pool=None, read_json=None, output_dir=os.path.abspath('.'),
                 read_opt_config=None):
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
        self.data.set_benchmark(benchmark)
        # 储存股票的alpha值
        self.alpha = None
        # 储存风险预测
        self.cov_mat = None
        self.spec_var = None

        # 优化器的条件
        self.opt_config = pd.Series()
        # 需要读取的优化条件的配置文件
        self.read_opt_config = read_opt_config

    # 读取json文件, 建立读取alpha的数据库引擎
    def initialize_alpha_engine(self):
        json = pd.read_json(self.read_json, orient='index')
        # json的数据库连接信息里面必须要有取alpha的smart quant数据库的连接信息
        assert 'Alpha' in json.index, 'Alpha database is not in the json file!\n'
        sq = json.ix['Alpha', :]
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
        # 如果stock pool与benchmark不同, 且stock pool不是all, 则也要取stock pool的权重数据
        if self.stock_pool != self.benchmark and self.stock_pool != 'all':
            data_needed = data_needed + ('Weight_'+self.stock_pool, )
        sql_raw_data = "select * from RawData where DataDate = '" + str(self.former_date) + "' and " \
                       "DataName in " + str(data_needed)
        raw_data = self.rm_engine.get_original_data(sql_raw_data)
        raw_data = raw_data.pivot_table(index=['DataDate', 'DataName'], columns='SecuCode',
            values='Value', aggfunc='first').to_panel().transpose(2, 1, 0)
        # 分配数据
        self.data.if_tradable = raw_data.ix[['is_enlisted', 'is_delisted', 'is_suspended']]
        if self.stock_pool != self.benchmark and self.stock_pool != 'all':
            self.data.benchmark_price = raw_data.ix[['Weight_'+self.benchmark, 'Weight_'+self.stock_pool]]
        else:
            self.data.benchmark_price = raw_data.ix[['Weight_'+self.benchmark]]
        # 生成可交易, 投资域相关的标记
        self.data.generate_if_tradable()
        self.data.handle_stock_pool()
        self.data.validate_benchmark_stockpool()

        # 取因子暴露数据
        # sql_expo = "select * from BarraBaseFactorExpo where DataDate = '" + str(self.former_date) + \
        #            "' and StockPool = '" + str(self.stock_pool) + "' "
        # 无论是什么投资域, 都用全市场的数据
        sql_expo = "select * from BarraBaseFactorExpo where DataDate = '" + str(self.former_date) + \
                   "' and StockPool = '" + str('all') + "' "
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
        # sql_covmat = "select * from BarraBaseCovMat where DataDate = '" + str(self.former_date) + \
        #              "' and StockPool = '" + str(self.stock_pool) + "' "
        # 无论是什么投资域, 都用全市场的数据
        sql_covmat = "select * from BarraBaseCovMat where DataDate = '" + str(self.former_date) + \
                     "' and StockPool = '" + str('all') + "' "
        covmat = self.rm_engine.get_original_data(sql_covmat)
        covmat = covmat.pivot_table(index='FactorName1', columns='FactorName2', values='Value')
        self.cov_mat = covmat.reindex(index=self.data.factor_expo.items, columns=self.data.factor_expo.items)
        # sql_specvar = "select * from BarraBaseSpecVar where DataDate = '" + str(self.former_date) + \
        #               "' and StockPool = '" + str(self.stock_pool) + "' "
        # 无论是什么投资域, 都用全市场的数据
        sql_specvar = "select * from BarraBaseSpecVar where DataDate = '" + str(self.former_date) + \
                      "' and StockPool = '" + str('all') + "' "
        specvar = self.rm_engine.get_original_data(sql_specvar)
        specvar = specvar.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                      aggfunc='first').squeeze()
        self.spec_var = specvar
        pass

    # 读取alpha数据的函数
    def read_alpha(self):
        # alpha数据是根据alpha的名字来匹配, 默认是Signal_Main
        if 'alpha_name' not in self.opt_config.index:
            self.opt_config['alpha_name'] = 'Signal_Main'
        sql_alpha = "select runnerdate as \"DataDate\", stockticker as \"SecuCode\", value as \"Value\" " \
                    "from RunnerValue where runnerdate = '" + str(self.former_date) + "' and " \
                    "runnerid in (select distinct runnerid from RunnerInfo where runnername = '" + \
                    str(self.opt_config['alpha_name']) + "') order by \"DataDate\", \"SecuCode\" "
        alpha = self.alpha_engine.get_original_data(sql_alpha)
        alpha = alpha.pivot_table(index='DataDate', columns='SecuCode', values='Value',
                                  aggfunc='first').squeeze()
        # 将alpha除以标准差, 使得其是一个标准差为1的series, 暂时不对其均值做处理
        self.alpha = alpha.div(alpha.std())

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

    # 设置优化的限制条件
    def set_opt_config(self):
        # 如果有配置文件, 读取配置文件
        if self.read_opt_config is not None:
            self.opt_config = pd.read_json(self.read_opt_config, orient='index', typ='series')

        # 首先看是否有风险厌恶系数的设置, 没有的话, 就用优化器的默认设置
        if 'cov_risk_aversion' in self.opt_config.index:
            self.optimizer.set_risk_aversion(cov_risk_aversion=self.opt_config['cov_risk_aversion'])
        else:
            self.opt_config['cov_risk_aversion'] = self.optimizer.cov_risk_aversion / 100
        if 'spec_risk_aversion' in self.opt_config.index:
            self.optimizer.set_risk_aversion(spec_risk_aversion=self.opt_config['spec_risk_aversion'])
        else:
            self.opt_config['spec_risk_aversion'] = self.optimizer.spec_risk_aversion / 100

        # 看是否有优化器包的设置, 没有的话, 默认设置为cvxopt
        if 'opt_package' in self.opt_config.index:
            self.optimizer.set_opt_package(self.opt_config['opt_package'])
            # 把opt_config里的opt_package写发改成标准写法
            self.opt_config['opt_package'] = self.optimizer.opt_package
        else:
            self.opt_config['opt_package'] = self.optimizer.opt_package

        # 看是否有行业中性的限制条件, 默认行业中性=True
        if 'indus_neutral' not in self.opt_config.index:
            self.opt_config['indus_neutral'] = True

        # 全投资的限制条件, 默认为True, 注意如果有行业中性, 全投资的限制条件会被改成False
        if 'enable_full_inv_cons' not in self.opt_config.index:
            self.opt_config['enable_full_inv_cons'] = True

        # 股票的上下限, 下限只能选择是否为0, 上限有一个asset cap, 还有现金比例限制
        if 'long_only' not in self.opt_config.index:
            self.opt_config['long_only'] = True
        if 'asset_cap' not in self.opt_config.index:
            self.opt_config['asset_cap'] = None
        if 'cash_ratio' not in self.opt_config.index:
            self.opt_config['cash_ratio'] = 0

        # 因子暴露限制条件
        if 'factor_expo_cons' not in self.opt_config.index:
            self.opt_config['factor_expo_cons'] = None

        # 上期持仓
        if 'old_w' not in self.opt_config.index:
            self.opt_config['old_w'] = None

        # 交易费用的限制条件
        if 'enable_trans_cost' not in self.opt_config.index:
            self.opt_config['enable_trans_cost'] = False
        if 'buy_cost' not in self.opt_config.index:
            self.opt_config['buy_cost'] = 1.5/1000
        if 'sell_cost' not in self.opt_config.index:
            self.opt_config['sell_cost'] = 1.5/1000

        # 换手率限制条件
        if 'enable_turnover_cons' not in self.opt_config.index:
            self.opt_config['enable_turnover_cons'] = False
        if 'turnover_cap' not in self.opt_config.index:
            self.opt_config['turnover_cap'] = 1.0

    # 解优化
    def solve_opt_portfolio(self):
        # 从panel中取数据, 将数据变成dataframe, 注意factor expo是行为因子, 列为股票
        curr_benchmark_weight = self.data.benchmark_price.ix['Weight_'+self.benchmark, 0, :]
        curr_factor_expo = self.data.factor_expo.ix[:, 0, :].T

        ################################################################################################
        # 交易费用, 换手率, 现金比率, 单个资产上限的限制条件暂时不加
        ################################################################################################

        # 首先考虑因子暴露限制条件, 注意, 如果有行业中性, 行业中性的限制条件不要写在这里
        if self.opt_config['factor_expo_cons'] is not None:
            # 从json读入的限制条件多半是dict, 将其转为dataframe
            factor_expo_cons = pd.DataFrame(self.opt_config['factor_expo_cons']).T
        else:
            factor_expo_cons = None

        # 如果进行行业中性
        if self.opt_config['indus_neutral']:
            # 首先判断风格因子和行业因子的个数
            industry = self.data.factor_expo.items.str.startswith('Industry')
            n_indus = industry[industry].size
            n_styles = self.data.factor_expo.shape[0] - n_indus - 1
            # 行业因子的因子名
            indus_name = self.data.factor_expo.items[n_styles:(n_styles+n_indus)]
            indus_cons = pd.DataFrame(indus_name.values, columns=['factor'])
            indus_cons['if_eq'] = True
            indus_cons['if_lower_bound'] = True
            indus_cons['limit'] = 0.0
            # 如果这一期, 某个行业的所有股票都是0暴露, 则在行业限制中去除这个行业
            empty_indus = curr_factor_expo[n_styles:(n_styles+n_indus)].sum(1) == 0
            indus_cons = indus_cons[np.logical_not(empty_indus.values)]
            # 将indus_cons和factor_expo_cons拼到一起, 形成完整的因子暴露限制条件
            if factor_expo_cons is not None:
                factor_expo_cons = pd.concat([factor_expo_cons, indus_cons], axis=0).drop_duplicates()
            else:
                factor_expo_cons = indus_cons
            self.opt_config['enable_full_inv_cons'] = False

        # 解优化组合
        self.optimizer.solve_optimization(curr_benchmark_weight, curr_factor_expo, self.cov_mat,
            residual_return=self.alpha, specific_var=self.spec_var, factor_expo_cons=factor_expo_cons,
            enable_full_inv_cons=self.opt_config['enable_full_inv_cons'],
            long_only=self.opt_config['long_only'], asset_cap=self.opt_config['asset_cap'],
            cash_ratio=self.opt_config['cash_ratio'], old_w=self.opt_config['old_w'],
            enable_trans_cost=self.opt_config['enable_trans_cost'], buy_cost=self.opt_config['buy_cost'],
            sell_cost=self.opt_config['sell_cost'], turnover_cap=self.opt_config['turnover_cap'],
            enable_turnover_cons=self.opt_config['enable_turnover_cons'])

        pass

    # 储存优化结果
    def save_opt_outcome(self):
        # 输出的优化持仓时间, 以及所用数据的时间
        self.opt_config['ValueDate'] = str(self.date)
        self.opt_config['DataDate'] = str(self.former_date)
        # 优化的投资域和基准
        self.opt_config['benchmark'] = self.benchmark
        self.opt_config['stock_pool'] = self.stock_pool
        # 优化结果的输出信息
        self.opt_config['OptResult_success'] = self.optimizer.opt_result['success']
        self.opt_config['OptResult_status'] = self.optimizer.opt_result['status']
        self.opt_config['OptResult_message'] = self.optimizer.opt_result['message']
        self.opt_config['obj_func_value'] = self.optimizer.opt_result['fun']
        self.opt_config['forecasted_vol'] = self.optimizer.forecasted_vol

        # 由于数据库没有bool类型, 因此将true false替换成1 0来储存
        self.opt_config = self.opt_config.replace(True, 1).replace(False, 0)
        self.opt_config.to_json(self.output_dir + '/OptimizationProfile.json')

        # 判断优化器结果是否成功
        if self.optimizer.opt_result['success'] and self.optimizer.opt_result['status'] == 0:
            # 持仓绝对值小于1e-4的股票都不要
            output_holding = self.optimizer.optimized_weight.mask(
                self.optimizer.optimized_weight.abs() < 1e-4, np.nan).dropna()
            # 按照要求的格式, 股票代码叫StockTicker, 持仓是Value
            output_holding = output_holding.reset_index().rename(
                columns={'SecuCode': 'StockTicker', 0: 'Value'})
            output_holding.to_json(self.output_dir + '/Value.json', orient='records')
        else:
            raise ValueError('The optimization is not terminated successfully, portfolio holding '
                             'shall not be written!\n')
        pass

    # 执行优化系统的函数, 即把流程都走一遍得到结果的函数
    def execute_opt(self):
        start_time = time.time()
        self.initialize_alpha_engine()
        print('Alpha database engines has been initialized...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.initialize_rm_engine()
        print('RiskModel database engines has been initialized...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.get_former_day()
        self.read_alpha()
        print('Alpha data has been successfully read...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.read_rm_data()
        print('RiskModel data has been successfully read...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.prepare_data()
        print('Data preparation for optimization has been completed...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.set_opt_config()
        print('Optimization configuration has been set, try to solve optimization...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.solve_opt_portfolio()
        print('Optimization has been solved...\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)
        self.save_opt_outcome()
        print('Optimization outcome has been successfully saved!\n', flush=True)
        print("time: {0} seconds\n".format(time.time() - start_time), flush=True)


if __name__ == '__main__':
    test = opt_portfolio_prod(benchmark='hs300', stock_pool='hs300',
                              read_json=os.path.abspath('.')+'/dbuser.txt',
                              date='2018-01-29',
                              read_opt_config=os.path.abspath('.')+'/opt_config.txt')
    # test.initialize_alpha_engine()
    # test.initialize_rm_engine()
    # test.get_former_day()
    # test.read_rm_data()
    # test.read_alpha()
    # test.prepare_data()
    # test.opt_config['indus_neutral'] = True
    # test.set_opt_config()
    import time
    start_time = time.time()
    # test.solve_opt_portfolio()
    test.execute_opt()
    print("time: {0} seconds\n".format(time.time()-start_time))
    # test.save_opt_outcome()





