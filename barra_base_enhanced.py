import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import copy
import pathos.multiprocessing as mp

from data import data
from strategy_data import strategy_data
from position import position
from factor_base import factor_base
from barra_base import barra_base

# barra base加强类, 仍然基于barra的那几个风格因子, 再添加几个因子后增加r squared

class barra_base_enhanced(barra_base):
    """Enchaned barra base, with some factors added to increase overall r squared.

    foo
    """
    def __init__(self, *, stock_pool='all'):
        barra_base.__init__(self, stock_pool=stock_pool)

    # 计算short term reversal因子, 参考momentum因子
    def get_short_reversal(self):
        if os.path.isfile(os.path.abspath('.')+'/ResearchData/short_rev'+self.filename_appendix) \
                and not self.is_update and self.try_to_read:
            self.base_data.factor['short_rev'] = data.read_data(['short_rev'+self.filename_appendix],
                                                                item_name=['short_rev'])
        # 没有就进行计算
        else:
            # rolling求sum, 21个交易日, 半衰期为10天
            exponential_weights = strategy_data.construct_expo_weights(10, 21)
            # 定义reversal的函数
            def func_rev(df, *, weights):
                iweights = pd.Series(weights, index=df.index)
                # 将权重乘在原始数据上, 然后加和计算reversal
                weighted_return = strategy_data.multiply_weights(df, iweights, multiply_power=1.0)
                rev = weighted_return.sum(0)
                # 设定阈值, 表示至少过去21个交易日中有多少数据才能有momentum因子
                threshold_condition = df.notnull().sum(0) >= 5
                rev = rev.where(threshold_condition, np.nan)
                return rev
            reversal = self.base_data.stock_price.ix['daily_excess_log_return'] * np.nan
            for cursor, date in enumerate(self.complete_base_data.stock_price. \
                                                  ix['daily_excess_log_return'].index):
                # 至少第21期才开始计算
                if cursor < 20:
                    continue
                curr_data = self.complete_base_data.stock_price.ix['daily_excess_log_return',
                                cursor-20:cursor+1, :]
                temp = func_rev(curr_data, weights=exponential_weights)
                reversal.ix[cursor, :] = temp
            self.base_data.factor['short_rev'] = reversal


    # 考虑使用总市值

    # 计算市值对数因子，市值需要第一个计算以确保各个panel有index和column
    def get_lncap(self):
        # 如果有文件，则直接读取
        if os.path.isfile(os.path.abspath('.') + '/ResearchData/lncap_total' + self.filename_appendix) \
                and not self.is_update and self.try_to_read:
            self.base_data.factor = data.read_data(['lncap' + self.filename_appendix], item_name=['lncap'])
        # 没有就用市值进行计算
        else:
            self.base_data.factor = pd.Panel(
                {'lncap': np.log(self.base_data.stock_price.ix['MarketValue'])},
                major_axis=self.base_data.stock_price.major_axis,
                minor_axis=self.base_data.stock_price.minor_axis)

    # 计算nonlinear size
    def get_nonlinear_size(self):
        if os.path.isfile(os.path.abspath('.')+'/ResearchData/nls_total'+self.filename_appendix) \
                and not self.is_update and self.try_to_read:
            self.base_data.factor['nls'] = data.read_data('nls'+self.filename_appendix)
        else:
            # 计算市值因子的暴露，注意解释变量需要为一个panel
            x = pd.Panel({'lncap_expo': strategy_data.get_cap_wgt_exposure(self.base_data.factor.ix['lncap'],
                         self.base_data.stock_price.ix['FreeMarketValue'])})
            # 将市值因子暴露取3次方, 得到size cube
            y = x['lncap_expo'] ** 3
            # 将size cube对市值因子做正交化
            new_nls = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.base_data.stock_price.ix['FreeMarketValue']))[0]
            self.base_data.factor['nls'] = new_nls

    # 计算短期beta因子, 计算过去21个交易日的beta值
    def get_short_beta_parallel(self):
        if os.path.isfile(os.path.abspath('.') + '/ResearchData/short_beta' + self.filename_appendix) \
                and not self.is_update and self.try_to_read:
            self.base_data.factor['short_beta'] = data.read_data('short_beta' + self.filename_appendix)
        else:
            # 所有股票的日简单收益的市值加权，加权用前一交易日的市值数据进行加权
            cap_wgt_universe_return = self.base_data.stock_price.ix['daily_excess_simple_return'].mul(
                self.base_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
                self.base_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)

            # 回归函数
            def reg_func(y, *, x):
                # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
                if y.notnull().sum() <= 5:
                    return np.nan
                x = sm.add_constant(x)
                model = sm.OLS(y, x, missing='drop')
                results = model.fit()
                return results.params[1]

            # 按照Barra的方法进行回归
            # 股票收益的数据
            complete_return_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return']

            # 计算每期beta的函数
            def one_time_beta(cursor):
                # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据
                curr_data = complete_return_data.ix[cursor - 20:cursor + 1, :]
                curr_x = cap_wgt_universe_return.ix[cursor - 20:cursor + 1]
                temp = curr_data.apply(reg_func, x=curr_x)
                print(cursor)
                return temp

            ncpus = 20
            p = mp.ProcessPool(ncpus)
            p.close()
            p.restart()
            # 一般情况下, 是从21期开始计算beta因子
            # 注意, 在更新的时候, 为了节约时间, 不会算第21到第524个交易日的beta值
            # 因此在更新的时候, 会从第525期开始计算
            if self.is_update:
                start_cursor = 524
            else:
                start_cursor = 20
            data_size = np.arange(start_cursor,
                                  self.base_data.stock_price.ix['daily_excess_simple_return'].shape[0])
            chunksize = int(len(data_size) / ncpus)
            results = p.map(one_time_beta, data_size, chunksize=chunksize)
            # 储存结果i
            beta = pd.concat([i for i in results], axis=1).T
            p.close()
            p.join()
            # 两个数据对应的日期，为原始数据的日期减去20，因为前20期的数据并没有计算
            # 在更新的时候, 则是原始数据日期减去524, 原因同理
            data_index = self.base_data.stock_price.iloc[:,
                         start_cursor - self.base_data.stock_price.shape[1]:, :].major_axis
            beta = beta.set_index(data_index)
            self.base_data.factor['short_beta'] = beta

    # def get_beta_parallel(self):
    #     # if os.path.isfile(os.path.abspath('.')+'/ResearchData/beta'+self.filename_appendix) \
    #     #         and not self.is_update and self.try_to_read:
    #     #     self.base_data.factor['beta'] = data.read_data('beta'+self.filename_appendix)
    #     if False:
    #         pass
    #     else:
    #         # 所有股票的日简单收益的市值加权，加权用前一交易日的市值数据进行加权
    #         cap_wgt_universe_return = self.base_data.stock_price.ix['daily_excess_simple_return'].mul(
    #             self.base_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
    #             self.base_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)
    #
    #         # 回归函数
    #         def reg_func(y, *, x):
    #             # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
    #             if y.notnull().sum() <= 63:
    #                 return pd.Series({'beta': np.nan, 'hsigma': np.nan})
    #             x = sm.add_constant(x)
    #             model = sm.OLS(y, x, missing='drop')
    #             results = model.fit()
    #             resid = results.resid.reindex(index=y.index)
    #             # 在这里提前计算hsigma----------------------------------------------------------------------
    #             # 求std，252个交易日，63的半衰期
    #             exponential_weights_h = strategy_data.construct_expo_weights(63, 252)
    #             # 给weights加上index以索引resid
    #             exponential_weights_h = pd.Series(exponential_weights_h, index=y.index)
    #             # 给resid直接乘以权重, 然后按照权重计算加权的std
    #             weighted_resid = strategy_data.multiply_weights(resid, exponential_weights_h, multiply_power=0.5)
    #             hsigma = weighted_resid.std()
    #             # ----------------------------------------------------------------------------------------
    #             return pd.Series({'beta': results.params[1], 'hsigma': hsigma})
    #         # 按照Barra的方法进行回归
    #         # 股票收益的数据
    #         complete_return_data = self.complete_base_data.stock_price.ix['daily_excess_simple_return']
    #         # 计算每期beta的函数
    #         def one_time_beta(cursor):
    #             # 注意, 这里的股票收益因为要用过去一段时间的数据, 因此要用完整的数据
    #             curr_data = complete_return_data.ix[cursor - 251:cursor + 1, :]
    #             curr_x = cap_wgt_universe_return.ix[cursor - 251:cursor + 1]
    #             temp = curr_data.apply(reg_func, x=curr_x)
    #             print(cursor)
    #             return temp
    #
    #         ncpus = 20
    #         p = mp.ProcessPool(ncpus)
    #         p.close()
    #         p.restart()
    #         # 一般情况下, 是从252期开始计算beta因子
    #         # 注意, 在更新的时候, 为了节约时间, 不会算第252到第524个交易日的beta值
    #         # 因此在更新的时候, 会从第525期开始计算
    #         if self.is_update:
    #             start_cursor = 524
    #         else:
    #             start_cursor = 251
    #         data_size = np.arange(start_cursor, self.base_data.stock_price.ix['daily_excess_simple_return'].shape[0])
    #         chunksize = int(len(data_size)/ncpus)
    #         results = p.map(one_time_beta, data_size, chunksize=chunksize)
    #         # 储存结果
    #         beta = pd.concat([i.ix['beta'] for i in results], axis=1).T
    #         hsigma = pd.concat([i.ix['hsigma'] for i in results], axis=1).T
    #         p.close()
    #         p.join()
    #         # 两个数据对应的日期，为原始数据的日期减去251，因为前251期的数据并没有计算
    #         # 在更新的时候, 则是原始数据日期减去524, 原因同理
    #         data_index = self.base_data.stock_price.iloc[:,
    #                      start_cursor - self.base_data.stock_price.shape[1]:, :].major_axis
    #         beta = beta.set_index(data_index)
    #         hsigma = hsigma.set_index(data_index)
    #         self.base_data.factor['beta'] = beta
    #         self.temp_hsigma = hsigma.reindex(self.base_data.stock_price.major_axis)

    # 考虑把growth因子改成fundamental momentum
    def get_growth(self):
        self.base_data.factor['growth'] = data.read_data('growth_bbe')
        # from db_engine import db_engine
        # sq_db = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
        #                   server_ip='192.168.66.12', port='1433', db_name='SmartQuant', add_info='')
        # sue = sq_db.get_original_data("select * from RunnerValue where runnerid=36 ")
        # sue['runnerdate'] = pd.to_datetime(sue['runnerdate'])
        # sue = sue.pivot_table(index='runnerdate', columns='stockticker', values='value')
        # sue = sue.reindex(index=self.base_data.stock_price.major_axis,
        #                   columns=self.base_data.stock_price.minor_axis)
        # eps_rev = sq_db.get_original_data("select * from RunnerValue where runnerid=5 ")
        # eps_rev['runnerdate'] = pd.to_datetime(eps_rev['runnerdate'])
        # eps_rev = eps_rev.pivot_table(index='runnerdate', columns='stockticker', values='value')
        # eps_rev = eps_rev.reindex(index=self.base_data.stock_price.major_axis,
        #                   columns=self.base_data.stock_price.minor_axis)
        # # 标准化
        # self.base_data.raw_data['sue_expo'] = strategy_data.get_cap_wgt_exposure(sue,
        #     self.base_data.stock_price.ix['FreeMarketValue'])
        # self.base_data.raw_data['eps_rev_expo'] = strategy_data.get_cap_wgt_exposure(eps_rev,
        #     self.base_data.stock_price.ix['FreeMarketValue'])
        # # 用等权将两个因子加起来
        # self.base_data.factor['growth'] = 0.5 * self.base_data.raw_data['sue_expo'] + \
        #                                   0.5 * self.base_data.raw_data['eps_rev_expo']

    # 计算风格因子的因子暴露
    def get_style_factor_exposure(self):
        # 给因子暴露panel加上索引
        self.base_data.factor_expo = pd.Panel(data=None, major_axis=self.base_data.factor.major_axis,
                                            minor_axis=self.base_data.factor.minor_axis)
        # 循环计算暴露
        for item, df in self.base_data.factor.iteritems():
            # 通过内部因子加总得到的因子，或已经计算过一次暴露的因子（如正交化过），不再需要去极值
            if item in ['rv', 'liquidity', 'ey', 'growth', 'nls']:
                self.base_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.base_data.stock_price.ix['FreeMarketValue'], percentile=0)
            else:
                self.base_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.base_data.stock_price.ix['FreeMarketValue'])


    # 构建barra base enhanced的所有因子
    def construct_factor_base(self, *, if_save=False):
        # 读取数据，更新数据则不用读取，因为已经存在
        if not self.is_update:
            self.read_original_data()
        # 构建风格因子前, 要设置读取文件的名称, 有可能会使用不同股票池下的因子定义
        self.construct_reading_file_appendix()

        # 首先检验是否有现成的本地的暴露文件可以读取
        if os.path.isfile(os.path.abspath('.') + '/ResearchData/bbe_factor_expo' + self.filename_appendix) \
                and not self.is_update and self.try_to_read:
            self.base_data.factor_expo = data.read_data('bbe_factor_expo'+self.filename_appendix)
            print('Barra base enhanced factor exposure data has been successfully read from local file!\n')
        else:
            # 创建风格因子
            self.get_lncap()
            print('get lncap completed...\n')
            self.get_beta_parallel()
            print('get beta completed...\n')
            self.get_momentum()
            print('get momentum completed...\n')
            self.get_residual_volatility()
            print('get rv completed...\n')
            self.get_nonlinear_size()
            print('get nls completed...\n')
            self.get_bp()
            print('get bp completed...\n')
            self.get_liquidity()
            print('get liquidity completed...\n')
            self.get_earnings_yeild()
            print('get ey completed...\n')
            self.get_growth()
            print('get growth completed...\n')
            self.get_leverage()
            print('get leverage completed...\n')
            self.get_short_reversal()
            print('get short reversal completed...\n')
            # self.get_short_beta_parallel()
            # print('get short beta completed...\n')
            # 计算风格因子暴露之前再过滤一次
            self.base_data.discard_uninv_data()
            # 计算风格因子暴露
            self.get_style_factor_exposure()
            print('get style factor expo completed...\n')
            # 加入行业暴露
            self.get_industry_factor()
            print('get indus factor expo completed...\n')
            # 添加国家因子
            self.add_country_factor()
            # 计算的最后，过滤数据
            self.base_data.discard_uninv_data()

        # 判定风格因子和行业因子的数量
        self.get_factor_group_count()

        # 如果显示指定了储存数据且股票池为所有股票，则储存因子值数据
        if not self.is_update and if_save:
            # 如果要储存因子数据, 必须保证当前股票池和文件后缀是完全一致的, 否则报错
            assert self.filename_appendix[1:] == self.base_data.stock_pool, 'Error: The stock ' \
                'pool of base is different from filename appendix, in order to avoid possible ' \
                'data loss, the saving procedure has been terminated! \n'
            # 根据股票池, 生成要储存的文件名
            written_filename = []
            for i in self.base_data.factor.items:
                written_filename.append(i+self.filename_appendix)
            data.write_data(self.base_data.factor, file_name=written_filename, separate=True)
            print('Style factor data have been saved!\n')
            # 储存因子暴露
            data.write_data(self.base_data.factor_expo, file_name='bbe_factor_expo'+self.filename_appendix)
            print('factor exposure data has been saved!\n')





if __name__ == '__main__':
    bbe = barra_base_enhanced(stock_pool='all')
    bbe.construct_factor_base(if_save=False)
    bbe.get_base_factor_return(if_save=False)
    bbe.style_factor_significance(freq='d')
    bbe.style_factor_similarity()
    # bbe.get_growth()










































































