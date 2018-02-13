import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
from datetime import datetime
import os
import statsmodels.api as sm
import copy
from linearmodels import FamaMacBeth

from single_factor_strategy import single_factor_strategy
from database import database
from data import data
from strategy_data import strategy_data

# 构造titman文章中的结果\

class intangible_info(single_factor_strategy):
    def __init__(self, *, time_range=252*3):
        single_factor_strategy.__init__(self)
        # 用于取数据的database类, 并且初始化数据
        self.db = database(start_date=pd.Timestamp('2007-01-01'), end_date=pd.Timestamp('2017-06-21'))
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 表示数据一共要算多久的, 因为论文里用的是5年, 国内数据没有这么多, 默认取3年
        self.time_range = time_range

    # 读取需要用到的数据
    def prepare_data(self, *, price='ClosePrice'):
        price_adj = price + '_adj'
        self.strategy_data.raw_data = data.read_data([price_adj, 'vwap_adj', price, 'TotalEquity',
            'FreeShares','FreeMarketValue'], shift=True)
        # 计算一些可能会用到的数据
        self.strategy_data.raw_data['lbm'] = np.log(self.strategy_data.raw_data['TotalEquity']/\
            self.strategy_data.raw_data['FreeMarketValue'])
        self.strategy_data.raw_data['delta_lbm'] = self.strategy_data.raw_data['lbm'].\
            sub(self.strategy_data.raw_data['lbm'].shift(1))
        self.strategy_data.raw_data['lb'] = np.log(self.strategy_data.raw_data['TotalEquity'])
        self.strategy_data.raw_data['delta_lb'] = self.strategy_data.raw_data['lb'].\
            sub(self.strategy_data.raw_data['lb'].shift(1))
        self.strategy_data.raw_data['daily_return'] = np.log(self.strategy_data.raw_data[price_adj]/\
            self.strategy_data.raw_data[price_adj].shift(1))
        # 价格变化要用股数调整后的价格变化
        self.strategy_data.raw_data['share_adj_price'] = self.strategy_data.raw_data[price] * \
            self.strategy_data.raw_data['FreeShares']
        self.strategy_data.raw_data['price_change_adj'] = np.log(self.strategy_data.raw_data['share_adj_price']/\
            self.strategy_data.raw_data['share_adj_price'].shift(1))

    # 使用两种方法分别计算book value return
    # 使用第一种, 直接法, 用book value的变化加上share adjust factor 计算
    def get_bv_return_direct(self):
        # 计算bookvalue的累计变化
        cum_delta_lb = self.strategy_data.raw_data['delta_lb'].rolling(self.time_range).sum()

        # 计算这段时间的累计对数收益
        cum_log_r = self.strategy_data.raw_data['daily_return'].rolling(self.time_range).sum()
        # 计算这段时间的价格变化(非复权)
        cum_delta_p = self.strategy_data.raw_data['price_change_adj'].rolling(self.time_range).sum()
        # 计算share adjust factor
        share_adjust_factor = cum_log_r - cum_delta_p

        # 用share adjust factor 与 cum delta lb相加, 得到bv return
        self.bv_return_direct = cum_delta_lb + share_adjust_factor
        pass

    # 使用第二种方法, 间接法, 用bm的变化加上收益率计算
    def get_bv_return_indirect(self):
        # 计算这段时间的对数bp的累计变化
        cum_delta_lbm = self.strategy_data.raw_data['delta_lbm'].rolling(self.time_range).sum()
        # 计算这段时间的累计收益
        cum_log_r = self.strategy_data.raw_data['daily_return'].rolling(self.time_range).sum()

        # 用这段时间的对数bp变化加上对数收益, 得到bv return
        self.bv_return_indirect = cum_delta_lbm + cum_log_r
        pass

    # 检查对比两种方法计算出的bv return是否相差不大
    def check_two_bv_returns(self):
        diff = self.bv_return_direct - self.bv_return_indirect
        diff_mean = diff.mean()
        if diff_mean.mean() >= 1e-4:
            print('Warning: BV returns get from two methods are not consistent! \n')
        else:
            self.bv_return = self.bv_return_direct
        pass

    # 用收益进行回归, 取残差, 得到intangible return
    def get_intangible_return(self):
        # 这段时间之前的bm
        lagged_bm = self.strategy_data.raw_data['lbm'].shift(self.time_range)
        # x为这段时间之前的bm, 以及这段时间的bv return
        indep = pd.Panel({'lagged_bm':lagged_bm, 'bv_return':self.bv_return})

        # 定义一次回归的函数
        def reg_func(y, x):
            x = sm.add_constant(x)
            # 如果只有小于等于一个有效数据, 则返回nan序列
            if pd.concat([y,x], axis=1).dropna().shape[0] <= 1:
                return 'empty'
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            tan_r = x.mul(results.params, axis=1).sum(1, skipna=False)
            intan_r = results.resid.reindex(index=x.index)
            return [tan_r, intan_r, results.params, results.rsquared_adj]

        intan_return = lagged_bm * np.nan
        tan_return = lagged_bm * np.nan
        stats = pd.DataFrame(np.nan, index=tan_return.index, columns=['const', 'bv_return',
                                                                       'lagged_bm', 'rsquared_adj'])
        cum_log_return = self.strategy_data.raw_data.ix['daily_return'].rolling(self.time_range).sum()
        # 循环进行回归
        for curcor, date in enumerate(indep.major_axis):
            curr_results = reg_func(cum_log_return.ix[date, :], indep.ix[:, date, :])
            if type(curr_results) != str:
                tan_return.ix[date, :] = curr_results[0]
                intan_return.ix[date, :] = curr_results[1]
                stats.ix[date, :] = curr_results[2]
                stats.ix[date, 'rsquared_adj'] = curr_results[3]
                pass
        self.intangible_return = intan_return
        self.tangible_return = tan_return
        pass

    # 按照论文的table3制作, 看结果如何
    def get_table3(self, *, freq='m'):
        # 首先需要按照频率生成holdingdays
        self.generate_holding_days(holding_freq=freq, loc=-1)
        # 按照频率算收益率, 和holdingdays同步, 论文用月, 我们一般用w
        r = self.strategy_data.raw_data['daily_return'].resample('m').sum()
        # 因为r的index为月末, 但是月末不一定是交易日, 因此将r的index重置为holding days
        r = r.set_index(self.holding_days)
        # 注意, 回归的左边是未来一期的收益率, 因此要shift(-1), 即用到未来数据
        r = r.shift(-1)
        # 用于回归的右边
        reg_panel = pd.Panel({'lbm':self.strategy_data.raw_data['lbm'],
                              'lagged_lbm':self.strategy_data.raw_data['lbm'].shift(self.time_range),
                              'bv_return':self.bv_return,
                              'lagged_return':self.strategy_data.raw_data['daily_return'].
                                rolling(self.time_range).sum()})

        # 储存table3的结果
        table3 = pd.Panel(items=['coef', 't_stats'], major_axis=np.arange(5),
                          minor_axis=['intercept', 'lbm', 'lagged_lbm', 'bv_return', 'lagged_return'])
        # 使用holding days中的日期进行回归,
        # 1. 用lagged lbm回归
        results1 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_lbm'], self.holding_days, :])
        table3.ix['coef', 0, :] = results1[0]
        table3.ix['t_stats', 0, :] = results1[1]
        # 2. 使用bv return回归
        results2 = strategy_data.fama_macbeth(r, reg_panel.ix[['bv_return'], self.holding_days, :])
        table3.ix['coef', 1, :] = results2[0]
        table3.ix['t_stats', 1, :] = results2[1]
        # 3. 使用lagged return回归
        results3 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_return'], self.holding_days, :])
        table3.ix['coef', 2, :] = results3[0]
        table3.ix['t_stats', 2, :] = results3[1]
        # 4. 使用lagged lbm与bv return回归
        results4 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_lbm', 'bv_return'], self.holding_days, :])
        table3.ix['coef', 3, :] = results4[0]
        table3.ix['t_stats', 3, :] = results4[1]
        # 5. 使用lagged lbm, bv return, lagged return一起回归
        results5 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_lbm', 'bv_return', 'lagged_return'],
                                                 self.holding_days, :])
        table3.ix['coef', 4, :] = results5[0]
        table3.ix['t_stats', 4, :] = results5[1]

        # 储存信息
        table3.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                 '/' + 'Table3_coef.csv', na_rep='N/A', encoding='GB18030')
        table3.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                    '/' + 'Table3_t_stats.csv', na_rep='N/A', encoding='GB18030')
        pass


class intangible_info_earnings(intangible_info):
    def __init__(self, *, time_range=252*2):
        intangible_info.__init__(self, time_range=time_range)

    def prepare_data(self, *, price='ClosePrice'):
        intangible_info.prepare_data(self, price=price)
        add_data = data.read_data(['NetIncome_ttm', 'runner_value_36'],
                                  item_name=['NetIncome_ttm', 'sue'], shift=True)
        self.strategy_data.raw_data['NetIncome_ttm'] = add_data['NetIncome_ttm']
        self.strategy_data.raw_data['sue'] = add_data['sue']
        self.strategy_data.raw_data['ep_ttm'] = self.strategy_data.raw_data['NetIncome_ttm']/\
            self.strategy_data.raw_data['FreeMarketValue']
        # self.strategy_data.raw_data['ep_ttm'] = 0

    def get_intangible_return(self):
        lagged_ep = self.strategy_data.raw_data['ep_ttm'].shift(self.time_range)
        indep = pd.Panel({'lagged_ep':lagged_ep, 'sue':self.strategy_data.raw_data['sue']})

        # 定义一次回归的函数
        def reg_func(y, x):
            x = sm.add_constant(x)
            # 如果只有小于等于一个有效数据, 则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 1:
                return 'empty'
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            tan_r = x.mul(results.params, axis=1).sum(1, skipna=False)
            intan_r = results.resid.reindex(index=x.index)
            return [tan_r, intan_r, results.params, results.rsquared_adj]

        intan_return = lagged_ep * np.nan
        tan_return = lagged_ep * np.nan
        stats = pd.DataFrame(np.nan, index=tan_return.index, columns=['const', 'sue',
                                                                      'lagged_ep', 'rsquared_adj'])
        cum_log_return = self.strategy_data.raw_data.ix['daily_return'].rolling(self.time_range).sum()
        # 循环进行回归
        for curcor, date in enumerate(indep.major_axis):
            curr_results = reg_func(cum_log_return.ix[date, :], indep.ix[:, date, :])
            if type(curr_results) != str:
                tan_return.ix[date, :] = curr_results[0]
                intan_return.ix[date, :] = curr_results[1]
                stats.ix[date, :] = curr_results[2]
                stats.ix[date, 'rsquared_adj'] = curr_results[3]
                pass
        self.intangible_return = intan_return
        self.tangible_return = tan_return
        pass

        # 按照论文的table3制作, 看结果如何

    def get_table3(self, *, freq='m'):
        # 首先需要按照频率生成holdingdays
        self.generate_holding_days(holding_freq=freq, loc=-1)
        # 按照频率算收益率, 和holdingdays同步, 论文用月, 我们一般用w
        r = self.strategy_data.raw_data['daily_return'].resample('m').sum()
        # 因为r的index为月末, 但是月末不一定是交易日, 因此将r的index重置为holding days
        r = r.set_index(self.holding_days)
        # 注意, 回归的左边是未来一期的收益率, 因此要shift(-1), 即用到未来数据
        r = r.shift(-1)
        # 用于回归的右边
        reg_panel = pd.Panel({'ep_ttm': self.strategy_data.raw_data['ep_ttm'],
                              'lagged_ep': self.strategy_data.raw_data['ep_ttm'].shift(self.time_range),
                              'sue': self.strategy_data.raw_data['sue'],
                              'lagged_return': self.strategy_data.raw_data['daily_return'].
                             rolling(self.time_range).sum()})

        # 储存table3的结果
        table3 = pd.Panel(items=['coef', 't_stats'], major_axis=np.arange(5),
                          minor_axis=['intercept', 'ep_ttm', 'lagged_ep', 'sue', 'lagged_return'])
        # 使用holding days中的日期进行回归,
        # 1. 用lagged lbm回归
        results1 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep'], self.holding_days, :])
        table3.ix['coef', 0, :] = results1[0]
        table3.ix['t_stats', 0, :] = results1[1]
        # 2. 使用bv return回归
        results2 = strategy_data.fama_macbeth(r, reg_panel.ix[['sue'], self.holding_days, :])
        table3.ix['coef', 1, :] = results2[0]
        table3.ix['t_stats', 1, :] = results2[1]
        # 3. 使用lagged return回归
        results3 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_return'], self.holding_days, :])
        table3.ix['coef', 2, :] = results3[0]
        table3.ix['t_stats', 2, :] = results3[1]
        # 4. 使用lagged lbm与bv return回归
        results4 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep', 'sue'], self.holding_days, :])
        table3.ix['coef', 3, :] = results4[0]
        table3.ix['t_stats', 3, :] = results4[1]
        # 5. 使用lagged lbm, bv return, lagged return一起回归
        results5 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep', 'sue', 'lagged_return'],
                                                 self.holding_days, :])
        table3.ix['coef', 4, :] = results5[0]
        table3.ix['t_stats', 4, :] = results5[1]

        # 储存信息
        table3.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                     '/' + 'Table3_coef.csv', na_rep='N/A', encoding='GB18030')
        table3.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                        '/' + 'Table3_t_stats.csv', na_rep='N/A', encoding='GB18030')
        pass

    def data_description(self):
        self.prepare_data()
        self.get_table3()

    def construct_factor(self):
        price_data = data.read_data(['ClosePrice_adj', 'OpenPrice_adj', 'vwap_adj'],
                                    item_name=['ClosePrice_adj', 'OpenPrice_adj', 'vwap_adj'],
                                    shift=True)
        ret = np.log(price_data['ClosePrice_adj'] / price_data['ClosePrice_adj'].shift(1))
        ret = ret.fillna(0)
        exp_w = strategy_data.construct_expo_weights(126, 504)
        mom = ret.rolling(504).apply(lambda x: (x * exp_w).sum())

        # mom = data.read_data(['runner_value_8'], shift=True)
        # mom = - mom['runner_value_8']

        bb = data.read_data(['rv', 'liquidity', 'lncap', 'runner_value_36'], shift=True)
        # bb = data.read_data(['runner_value_36'], shift=True)

        # 过滤数据
        self.strategy_data.handle_stock_pool()
        mom = mom.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        for item, df in bb.iteritems():
            bb[item] = df.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)

        # 进行回归
        # orth_mom = strategy_data.simple_orth_gs(mom, bb)
        # orth_mom = - orth_mom[0]

        self.strategy_data.factor = pd.Panel({'mom':-mom})


# 重新审视这个因子 2018年2月1日
class reversal_new(intangible_info_earnings):
    def __init__(self, *, time_range=252*1):
        intangible_info_earnings.__init__(self, time_range=time_range)

    # 读取数据
    def construct_factor(self):
        # 直接使用runner value中的rv8, rv36
        self.strategy_data.raw_data = data.read_data(['runner_value_8', 'runner_value_36'],
                                                     item_name=['rv8', 'rv36'])
        self.strategy_data.stock_price = data.read_data(['FreeMarketValue'])

        # # 在投资域内进行标准化及回归
        # self.strategy_data.discard_uninv_data()

        # 由于rv没有进行标准化, 在这里进行标准化, 进行市值加权标准化
        self.strategy_data.raw_data['rv8'] = strategy_data.get_cap_wgt_exposure(
            self.strategy_data.raw_data['rv8'], self.strategy_data.stock_price['FreeMarketValue'])
        self.strategy_data.raw_data['rv36'] = strategy_data.get_cap_wgt_exposure(
            self.strategy_data.raw_data['rv36'], self.strategy_data.stock_price['FreeMarketValue'])

        # 等权回归
        reg_weight = None
        # 以根号市值作为回归权重
        # reg_weight = np.sqrt(self.strategy_data.stock_price['FreeMarketValue'])
        outcome = strategy_data.simple_orth_gs(self.strategy_data.raw_data['rv8'],
            self.strategy_data.raw_data[['rv36']], weights=reg_weight)

        new_factor = outcome[0]
        pvalues = outcome[1]
        rsquared = outcome[2]

        # 储存因子
        self.strategy_data.factor = pd.Panel({'new_reversal': new_factor.shift(1)})
        pass

    def get_table3(self, *, freq='w', foldername=None, startdate=None, enddate=None):
        # 首先需要按照频率生成holdingdays
        self.generate_holding_days(holding_freq=freq, loc=-1, start_date=startdate,
                                   end_date=enddate)
        # 读取数据
        self.strategy_data.raw_data['rv3'] = data.read_data('runner_value_3')
        # 对rv3进行标准化
        self.strategy_data.raw_data['rv3'] = strategy_data.get_cap_wgt_exposure(
            self.strategy_data.raw_data['rv3'], self.strategy_data.stock_price['FreeMarketValue'])

        self.strategy_data.stock_price['ClosePrice_adj'] = data.read_data('ClosePrice_adj')
        self.strategy_data.stock_price['daily_return'] = \
            self.strategy_data.stock_price['ClosePrice_adj'].pct_change()
        # 按照频率算收益率, 和holdingdays同步, 论文用月, 我们一般用w
        r = self.strategy_data.stock_price['daily_return', startdate:enddate, :].\
            resample(freq).sum()
        # 注意, 回归的左边是未来一期的收益率, 因此要shift(-1), 即用到未来数据
        r = r.shift(-1).dropna(how='all')
        # 因为r的index为月末, 但是月末不一定是交易日, 因此将r的index重置为holding days
        r = r.set_index(self.holding_days)
        # 用于回归的右边
        reg_panel = pd.Panel({'lagged_ep': self.strategy_data.raw_data['rv3'].shift(0),
                              'sue': self.strategy_data.raw_data['rv36'],
                              'reversal': self.strategy_data.raw_data['rv8']})
        # 储存table3的结果
        table3 = pd.Panel(items=['coef', 't_stats'], major_axis=np.arange(5),
                          minor_axis=['intercept', 'lagged_ep', 'sue', 'reversal'])
        # 使用holding days中的日期进行回归,
        # 1. 用lagged lbm回归
        results1 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep'], self.holding_days, :])
        table3.ix['coef', 0, :] = results1[0]
        table3.ix['t_stats', 0, :] = results1[1]
        # 2. 使用bv return回归
        results2 = strategy_data.fama_macbeth(r, reg_panel.ix[['sue'], self.holding_days, :])
        table3.ix['coef', 1, :] = results2[0]
        table3.ix['t_stats', 1, :] = results2[1]
        # 3. 使用lagged return回归
        results3 = strategy_data.fama_macbeth(r, reg_panel.ix[['reversal'], self.holding_days, :])
        table3.ix['coef', 2, :] = results3[0]
        table3.ix['t_stats', 2, :] = results3[1]
        # 4. 使用lagged lbm与bv return回归
        results4 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep', 'reversal'], self.holding_days, :])
        table3.ix['coef', 3, :] = results4[0]
        table3.ix['t_stats', 3, :] = results4[1]
        # 5. 使用lagged lbm, bv return, lagged return一起回归
        results5 = strategy_data.fama_macbeth(r, reg_panel.ix[['lagged_ep', 'sue', 'reversal'],
                                                 self.holding_days, :], nw_lags=0)
        table3.ix['coef', 4, :] = results5[0]
        table3.ix['t_stats', 4, :] = results5[1]

        # # 储存信息
        # if foldername is None:
        #     table3.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #                                  '/' + 'Table3_coef.csv', na_rep='N/A', encoding='GB18030')
        #     table3.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #                                     '/' + 'Table3_t_stats.csv', na_rep='N/A', encoding='GB18030')
        # else:
        #     table3.ix['coef'].to_csv(foldername +
        #                              '/' + 'Table3_coef.csv', na_rep='N/A', encoding='GB18030')
        #     table3.ix['t_stats'].to_csv(foldername +
        #                                 '/' + 'Table3_t_stats.csv', na_rep='N/A', encoding='GB18030')
        # pass

        # 尝试使用linearmodels包
        reg_panel['const'] = 1.0
        fm_result1 = FamaMacBeth(r, reg_panel.ix[['lagged_ep', 'const'], self.holding_days, :]).fit()
        fm_result2 = FamaMacBeth(r, reg_panel.ix[['sue', 'const'], self.holding_days, :]).fit()
        fm_result3 = FamaMacBeth(r, reg_panel.ix[['reversal', 'const'], self.holding_days, :]).fit()
        fm_result4 = FamaMacBeth(r, reg_panel.ix[['reversal', 'lagged_ep', 'const'],
                                    self.holding_days, :]).fit()
        fm_result5 = FamaMacBeth(r, reg_panel.ix[['reversal', 'sue', 'const'],
                                    self.holding_days, :]).fit()
        fm_result6 = FamaMacBeth(r, reg_panel.ix[:, self.holding_days, :]).fit()

        pass



if __name__ == '__main__':
    rn = reversal_new()
    rn.construct_factor()
    rn.get_table3(startdate=pd.Timestamp('2009-05-04'), enddate=pd.Timestamp('2018-01-16'))
    # ii = intangible_info_earnings()
    # ii.strategy_data.stock_pool = 'cyb'
    # ii.construct_factor()
    # ii.prepare_data()
    # ii.get_bv_return_direct()
    # ii.get_bv_return_indirect()
    # ii.check_two_bv_returns()
    # ii.get_intangible_return()
    # ii.get_table3()







































































