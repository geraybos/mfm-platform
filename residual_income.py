import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix
from pandas.tools.plotting import table
from pandas.stats.fama_macbeth import fama_macbeth
from statsmodels.discrete.discrete_model import Poisson

from single_factor_strategy import single_factor_strategy
from database import database
from data import data
from strategy_data import strategy_data
from strategy import strategy

# 测试用residual income valuation算出的股票估值因子

class residual_income(single_factor_strategy):
    """ Residual income valuation, single factor strategy class

    foo
    """
    def __init__(self):
        single_factor_strategy.__init__(self)
        # 用于取数据的database类, 并且初始化数据
        self.db = database(start_date='2007-01-01', end_date='2017-06-21')
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

    # 取dividend的数据
    def get_dividend(self, *, rolling_days=400):
        sql_query = "select a.SecuCode, b.ExDiviDate, b.TotalCashDiviComRMB from " \
                    "(select distinct InnerCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select InnerCode, ExDiviDate, TotalCashDiviComRMB FROM LC_Dividend) b " \
                    "on a.InnerCode=b.InnerCode " \
                    "order by SecuCode, ExDiviDate "
        dividend_data = self.db.jydb_engine.get_original_data(sql_query)
        dividend_table = dividend_data.pivot_table(index='ExDiviDate', columns='SecuCode',
                                                   values='TotalCashDiviComRMB')
        # 储存最终数据的dataframe
        dividend = dividend_table * np.nan
        # 循环每天的数据
        for cursor, time in enumerate(dividend_table.index):
            # 取最近的400天(默认)
            end_time = time
            start_time = end_time - pd.DateOffset(days=rolling_days-1)
            # 最近400天的项
            condition = np.logical_and(dividend_table.index>=start_time,
                                       dividend_table.index<=end_time)
            # 最近400天的总红利之和
            dividend.ix[time, :] = dividend_table[condition].sum()
        # reindex后存在raw data里
        self.strategy_data.raw_data = pd.Panel({'dividend':dividend}, major_axis=self.strategy_data. \
            stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)
        # 先向前填充, 然后将缺失数据填成0
        self.strategy_data.raw_data['dividend'] = self.strategy_data.raw_data['dividend']. \
            fillna(method='ffill').fillna(0)
        pass

    # 取其他要用的数据
    def get_reg_data(self):
        # 读取数据
        original_data = data.read_data(['TotalAssets', 'NetIncome_ttm', 'CFO_ttm', 'Shares'], shift=True)
        # 计算accrual
        accrual = original_data['NetIncome_ttm'].sub(original_data['CFO_ttm'])

        # 将数据除以总股数, 以得到每股的数据
        original_data_ps = pd.Panel(major_axis=original_data.major_axis, minor_axis=original_data.minor_axis)
        original_data_ps['AssetsPS'] = original_data['TotalAssets'].div(original_data['Shares'])
        original_data_ps['EPS'] = original_data['NetIncome_ttm'].div(original_data['Shares'])
        original_data_ps['AccrualPS'] = accrual.div(original_data['Shares'])
        original_data_ps['DividendPS'] = self.strategy_data.raw_data['dividend'].shift(1). \
            div(original_data['Shares'])
        # 将inf替换成nan, 主要是防止shares中有0
        original_data_ps = original_data_ps.replace(np.inf, np.nan)

        # 生成是否有dividend和eps是否为负的的虚拟变量
        dd = accrual * np.nan
        negEPS = accrual * np.nan
        for cursor, time in enumerate(original_data_ps.major_axis):
            # 是否有dividend
            divi = original_data_ps['DividendPS', time, :]
            if_dd = pd.get_dummies(divi>0)
            # 如果有true的那一列, 则取true的那一列为虚拟变量, 否则是一个全为0的序列
            if True in if_dd.columns:
                dd.ix[time, :] = if_dd[True]
            else:
                dd.ix[time, :] = 0
            # 是否是负的eps
            eps = original_data_ps['EPS', time, :]
            if_neg = pd.get_dummies(eps<0)
            # 如果有true的那一列, 则取true的那一列为虚拟变量, 否则是一个全为0的序列
            if True in if_neg.columns:
                negEPS.ix[time, :] = if_neg[True]
            else:
                negEPS.ix[time, :] = 0
        # 循环结束, 储存数据
        original_data_ps['dd'] = dd
        original_data_ps['negEPS'] = negEPS

        # 储存所有需要用到的回归数据
        self.reg_data = original_data_ps
        # 在reg data中加入ni ttm用来判断财报期
        self.reg_data['NetIncome_ttm'] = original_data['NetIncome_ttm']
        pass

    # 进行回归, 拟合回归模型, 然后预测之后的earnings
    # 注意, 无论投资域是什么, 这里都应当用全市场的数据来进行模型的拟合, 因为不在投资域里的股票也能提供预测信息
    # 而算因子收益的时候, 不在投资域的股票无法构建代表因子收益的纯因子组合, 因此两者不同
    def predict_eps(self):
        # 排除那些不能交易的股票的数据
        for item, df in self.reg_data.iteritems():
            self.reg_data[item] = self.reg_data[item].where(self.strategy_data.if_tradable['if_tradable'],
                                                            np.nan)
        # 按照论文的方法排出金融股和上市两年之内的股票
        self.exclude_financial_and_new()

        # # 首先计算暴露值, 可以计算barra的市值加权暴露, 也可计算普通的暴露
        # for items, df in self.reg_data.iteritems():
        #     # 注意虚拟变量无需进行标准化
        #     if items not in ['dd', 'negEPS']:
        #         self.reg_data[items] = strategy_data.get_cap_wgt_exposure(df,
        #                                 self.strategy_data.stock_price['FreeMarketValue'])

        # 储存预测的eps的df
        temp_df = self.reg_data.iloc[0] * np.nan
        predicted_eps_all = pd.Panel({'y1':temp_df, 'y2':temp_df, 'y3':temp_df})
        # 根据调仓日, 进行模型的拟合, 先循环调仓日
        for cursor, time in enumerate(self.holding_days):
            # 取当前调仓日前的所有数据, 用所有的数据来进行模型拟合
            # 原始论文中使用了过去10年的数据, 但是由于我们数据一共也才10年, 因此全部使用上
            for h in [4, 8, 12]:
                item_str = 'y'+ str(int(h/4))
                [y, x] = self.align_reg_data(time, horizon=h)
                if y.empty or x.empty:
                    predicted_eps_all.ix[item_str, time, :] = np.nan
                else:
                    model, reg_results = residual_income.execute_reg(y, x)
                    # 储存回归结果
                    reg_csv = pd.DataFrame({'params':reg_results.params, 'pvalues':reg_results.pvalues})
                    reg_csv.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                '/' + item_str + 'reg_results.csv', na_rep='N/A', encoding='GB18030')
                    # 利用拟合好的模型进行预测
                    curr_data = self.reg_data[:, time, :].drop('NetIncome_ttm', axis=1).dropna(how='any')
                    # 对curr_data进行1%的winsorize, 去掉极值
                    lower_curr_data = curr_data.quantile(0.01, axis=0)
                    upper_curr_data = curr_data.quantile(0.99, axis=0)
                    curr_data = curr_data.where(curr_data>=lower_curr_data, lower_curr_data, axis=1). \
                        where(curr_data<=upper_curr_data, upper_curr_data, axis=1)

                    curr_data = sm.add_constant(curr_data)
                    curr_predicted_eps = curr_data.mul(reg_results.params).sum(1)
                    predicted_eps_all.ix[item_str, time, :] = curr_predicted_eps

                pass
            print(time)
        # data.write_data(predicted_eps_all)
        pass

    # 预测eps的并行版本
    def predict_eps_parallel(self):
        # 排除那些不能交易的股票的数据
        for item, df in self.reg_data.iteritems():
            self.reg_data[item] = self.reg_data[item].where(self.strategy_data.if_tradable['if_tradable'],
                                                            np.nan)
        # 按照论文的方法排出金融股和上市两年之内的股票
        self.exclude_financial_and_new()

        # 储存预测的eps的df
        temp_df = self.reg_data.iloc[0] * np.nan
        predicted_eps_all = pd.Panel({'y1':temp_df, 'y2':temp_df, 'y3':temp_df})

        # 将数据取出, 因为并行中使用的数据不能在self中
        reg_data = self.reg_data * 1
        holding_days = self.holding_days

        # 定义计算单次时间的函数
        def one_time_eps(cursor):
            # 取当前调仓日前的所有数据, 用所有的数据来进行模型拟合
            # 原始论文中使用了过去10年的数据, 但是由于我们数据一共也才10年, 因此全部使用上
            time = holding_days[cursor]
            # 储存一期数据的df
            one_time_data = pd.DataFrame(np.nan, index=reg_data.minor_axis, columns=['y1', 'y2', 'y3'])
            for horizon in [4, 8, 12]:
                item_str = 'y'+ str(int(horizon/4))
                # 将align_reg_data函数在这里展开
                # [y, x] = self.align_reg_data(time, horizon=h)
                ###########################################################################################
                # 用ni_ttm的数据来判断财报的年份
                curr_ni = reg_data['NetIncome_ttm', :time, :]
                # 每支股票一共有多少期财报的数据
                unique_terms = curr_ni.apply(lambda x: x.nunique(), axis=0)
                # 储存y和x
                y = pd.Series()
                x = pd.DataFrame()

                # 开始循环每支股票
                for cursor, code in enumerate(unique_terms.index):
                    curr_unique_terms = unique_terms[code]
                    # 如果当前股票期数不足
                    if curr_unique_terms < horizon + 1:
                        continue
                    curr_stock_ni = curr_ni.ix[:, code]
                    # 丢弃重复数据, 只留第一期, 即用每个财报周期的第一天的数据来对下个财报周期的第一天的数据进行预测
                    unique_ni = curr_stock_ni.drop_duplicates(keep='first').dropna()

                    # 开始循环, 将满足当前回归的数据放入y和x中
                    i = 1
                    while i + horizon <= curr_unique_terms:
                        # 发出预测的那个时间点, 及其对应的数据
                        predict_time = unique_ni.index[i - 1]
                        predict_data = reg_data.ix[:, predict_time, code].drop('NetIncome_ttm')
                        real_time = unique_ni.index[i + horizon - 1]
                        real_data = reg_data.ix['EPS', real_time, code]
                        x = x.append(predict_data)
                        y = y.append(pd.Series(real_data, index=[code]))
                        i += 1
                ###########################################################################################
                model, reg_results = residual_income.execute_reg(y, x)
                # 利用拟合好的模型进行预测
                curr_data = reg_data[:, time, :].drop('NetIncome_ttm', axis=1).dropna(how='any')
                # 对curr_data进行1%的winsorize, 去掉极值
                lower_curr_data = curr_data.quantile(0.01, axis=0)
                upper_curr_data = curr_data.quantile(0.99, axis=0)
                curr_data = curr_data.where(curr_data>=lower_curr_data, lower_curr_data, axis=1). \
                    where(curr_data<=upper_curr_data, upper_curr_data, axis=1)

                curr_data = sm.add_constant(curr_data)
                curr_predicted_eps = curr_data.mul(reg_results.params).sum(1)
                one_time_data[item_str] = curr_predicted_eps
            print(time)
            return one_time_data

        # 进行并行计算
        import pathos.multiprocessing as mp
        if __name__ == '__main__':
            ncpus = 20
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(holding_days.shape[0])
            chunksize = int(len(data_size)/ncpus)
            results = p.map(one_time_eps, data_size, chunksize=chunksize)
            predicted_eps_y1 = pd.concat([i['y1'] for i in results], axis=1).T
            predicted_eps_y2 = pd.concat([i['y2'] for i in results], axis=1).T
            predicted_eps_y3 = pd.concat([i['y3'] for i in results], axis=1).T
            predicted_eps_y1 = predicted_eps_y1.set_index(self.holding_days).reindex(
                index=predicted_eps_all.major_axis, method='ffill')
            predicted_eps_all.ix['y1'] = predicted_eps_y1
            predicted_eps_y2 = predicted_eps_y2.set_index(self.holding_days).reindex(
                index=predicted_eps_all.major_axis, method='ffill')
            predicted_eps_all.ix['y2'] = predicted_eps_y2
            predicted_eps_y3 = predicted_eps_y3.set_index(self.holding_days).reindex(
                index=predicted_eps_all.major_axis, method='ffill')
            predicted_eps_all.ix['y3'] = predicted_eps_y3

            # 向前填充
            predicted_eps_all = predicted_eps_all.fillna(method='ffill', axis=1)

            # 储存数据
            data.write_data(predicted_eps_all)


    # 根据传入的截至当前的数据, 来判断每支股票过去的所有财报期数据, 然后用这些数据来进行回归
    # horizon表示回归数据的预测期为多少季度, 按照论文的做法, 要分别对4, 8, 12季度的数据做回归拟合
    def align_reg_data(self, curr_time, *, horizon=4):
        # 用ni_ttm的数据来判断财报的年份
        curr_ni = self.reg_data['NetIncome_ttm', :curr_time, :]
        # 每支股票一共有多少期财报的数据
        unique_terms = curr_ni.apply(lambda x:x.nunique(), axis=0)
        # 储存y和x
        y = pd.Series()
        x = pd.DataFrame()

        # 开始循环每支股票
        for cursor, code in enumerate(unique_terms.index):
            curr_unique_terms = unique_terms[code]
            # 如果当前股票期数不足
            if curr_unique_terms < horizon+1:
                continue
            curr_stock_ni = curr_ni.ix[:, code]
            # 丢弃重复数据, 只留第一期, 即用每个财报周期的第一天的数据来对下个财报周期的第一天的数据进行预测
            unique_ni = curr_stock_ni.drop_duplicates(keep='first').dropna()

            # 开始循环, 将满足当前回归的数据放入y和x中
            i = 1
            while i+horizon <= curr_unique_terms:
                # 发出预测的那个时间点, 及其对应的数据
                predict_time = unique_ni.index[i-1]
                predict_data = self.reg_data.ix[:, predict_time, code].drop('NetIncome_ttm')
                real_time = unique_ni.index[i+horizon-1]
                real_data = self.reg_data.ix['EPS', real_time, code]
                x = x.append(predict_data)
                y = y.append(pd.Series(real_data, index=[code]))
                i += 1
            pass
        pass
        return [y, x]

    # 执行回归的函数
    @staticmethod
    def execute_reg(y, x):
        # 首先对数据进行1%的winsorize, 由于目前看来对每年做有一点难度, 因此对历史上所有数据做
        lower_y = y.quantile(0.01)
        upper_y = y.quantile(0.99)
        y = y.where(y>=lower_y, lower_y).where(y<=upper_y, upper_y)
        lower_x = x.quantile(0.01, axis=0)
        upper_x = x.quantile(0.99, axis=0)
        x = x.where(x>=lower_x, lower_x, axis=1).where(x<=upper_x, upper_x, axis=1)

        x = sm.add_constant(x)
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()

        return [model, results]

    # 计算折现率的函数
    def get_discount_factor(self):
        # 读取beta
        beta = data.read_data(['beta'])
        beta = beta['beta']
        # 读取行业数据
        indus = data.read_data(['Industry'])
        indus = indus['Industry']
        # 计算每天的行业平均beta
        indus_mean_beta = pd.DataFrame(np.nan, index=beta.index, columns=indus.iloc[-1, :].unique())
        for cursor, time in enumerate(beta.index):
            curr_beta = beta.ix[time, :]
            curr_indus = indus.ix[time, :]
            indus_mean_beta.ix[time, :] = curr_beta.groupby(curr_indus).mean()

        # 将每天的行业平均beta根据每支股票的行业分配到每支股票上去
        stocks_with_indus_mean_beta = beta * np.nan
        for cursor, time in enumerate(beta.index):
            if time >= pd.Timestamp('2009-04-01'):
                curr_indus = indus.ix[time, :]
                curr_indus_mean_beta = indus_mean_beta.ix[time, :]
                stocks_with_indus_mean_beta.ix[time, :] = curr_indus.replace(curr_indus_mean_beta.to_dict())
                pass

        # 由于使用的是日beta, 因此算一个过去252个交易日的平均beta作为年beta
        stocks_with_indus_mean_beta_annual = stocks_with_indus_mean_beta.rolling(
            252, min_periods=63).apply(lambda x:np.nanmean(x))
        # 暂时设无风险利率为0
        # 论文中, 市场的超额收益直接设定在了6%的常数, 这个的合理性还有待检验
        # 还可以考虑使用历史数据得出市场超额收益等, 现在先暂时使用6%这个常数
        discount_factor = stocks_with_indus_mean_beta_annual * 0.06
        self.discount_factor = discount_factor
        pass

    # 获取未来的book value
    def predict_book_value(self):
        # 读取total equity, shares等数据
        original_data = data.read_data(['TotalEquity', 'Shares', 'NetIncome_ttm'], shift=True)
        predicted_eps = data.read_data(['y1', 'y2', 'y3'], shift=False)

        # 可以采用当前股息率的方法来估计payout ratio, 然后来计算未来的book value
        dividend = self.strategy_data.raw_data['dividend'].shift(1)
        # 用dividend除以当前的净利润, 得到估计的payout ratio
        payout_ratio = dividend.div(original_data['NetIncome_ttm'])
        # payout ratio为负的, 即当前净利润为0的, 应当改为0
        payout_ratio = payout_ratio.where(payout_ratio >= 0, 0)
        # 计算得到当前的book value per share
        bps = original_data['TotalEquity'].div(original_data['Shares'])

        # 于是, 便可以通过公式计算未来的bps
        bps_y1 = (1 - payout_ratio) * predicted_eps['y1'] + bps
        # 注意, 如果预测的eps为负数, 则预测的bps直接为减去预测的eps
        bps_y1 = bps_y1.where(predicted_eps['y1'] >= 0, bps + predicted_eps['y1'])
        bps_y2 = (1 - payout_ratio) * predicted_eps['y2'] + bps
        bps_y2 = bps_y2.where(predicted_eps['y2'] >= 0, bps_y1 + predicted_eps['y2'])

        self.bps = pd.Panel({'bps':bps, 'bps_y1':bps_y1, 'bps_y2':bps_y2})

    # 根据预测的eps和discount factor, 用residual income model对股票估值
    def get_value_using_rim(self):
        # 读取存在本地的预测的eps
        predicted_eps = data.read_data(['y1', 'y2', 'y3'], shift=False)
        price = data.read_data(['ClosePrice'], shift=True)

        # 根据公式, 计算ri1, ri2, ri3
        ri1 = (predicted_eps['y1'] - self.discount_factor * self.bps['bps']) / (1 + self.discount_factor)
        ri2 = (predicted_eps['y2'] - self.discount_factor * self.bps['bps_y1']) / \
              (1 + self.discount_factor) ** 2
        ri3 = (predicted_eps['y3'] - self.discount_factor * self.bps['bps_y2']) / \
              (1 + self.discount_factor) ** 3

        # 终值是假设3年之后的ri3可以一直持续, 成长率为0
        terminal_value = ri3.div((1 + self.discount_factor) ** 3).div(self.discount_factor)

        # 最终的价值
        rim_value = self.bps['bps'] + ri1 + ri2 + ri3 + terminal_value

        # 用rim value除以当天的收盘价, 得到估值水平
        vp = rim_value.div(price['ClosePrice'])
        self.strategy_data.factor = pd.Panel({'vp':vp})

    # 按照论文删除样本的方法, 金融类的股票和上市不足两年的股票全部排出在样本之外
    def exclude_financial_and_new(self):
        industry = pd.read_csv('Industry.csv', index_col=0, parse_dates=True, encoding='GB18030')
        # shift一天
        industry = industry.shift(1)
        # 第一个条件, 金融股全部排出
        bank = pd.DataFrame('银行', index=industry.index, columns=industry.columns)
        non_bank_financial = pd.DataFrame('非银金融', index=industry.index, columns=industry.columns)
        condition_finance = np.logical_or(industry == bank, industry == non_bank_financial)

        self.get_enlisted_data()
        # 第二个条件, 上市两年以内的公司都排除, 用504个交易日来代替
        enlisted_days = self.is_enlisted.cumsum(0)
        condition_new = enlisted_days <= 504
        condition_new = condition_new.reindex(index=self.reg_data.major_axis)

        # 排除的股票是只要是金融股或者上市2年内的都排出, 这里取的时候没有排出的, 即加一个not
        not_exluded_condition = np.logical_not(np.logical_or(condition_finance, condition_new))

        # 被排出的股票都被设置为nan
        for item, df in self.reg_data.iteritems():
            self.reg_data[item] = self.reg_data[item].where(not_exluded_condition, np.nan)
        pass

    # 取上市数据的函数, 照搬database中的函数
    def get_enlisted_data(self):
        sql_query = "select a.SecuCode, b.ChangeDate, b.ChangeType from "\
                    "(select distinct InnerCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select ChangeDate, ChangeType, InnerCode from LC_ListStatus where SecuMarket in " \
                    "(83,90) and  ChangeDate<='" + \
                    str(self.reg_data.major_axis[-1]) + "') b on a.InnerCode=b.InnerCode "\
                    " order by SecuCode, ChangeDate"
        list_status = self.db.jydb_engine.get_original_data(sql_query)
        list_status = list_status.pivot_table(index='ChangeDate',columns='SecuCode',values='ChangeType',
                                              aggfunc='first')
        # 向前填充
        list_status = list_status.fillna(method='ffill')

        # 同时需要取交易日信息
        sql_query2 = "select TradingDate as trading_days from QT_TradingDayNew where SecuMarket=83 " \
                     "and IfTradingDay=1"
        trading_days = self.db.jydb_engine.get_original_data(sql_query2)
        trading_days = trading_days['trading_days']
        # 截取交易日, 从第一家公司的上市日期, 到reg data的最后一天
        trading_days = trading_days[trading_days>=list_status.index[0]]
        trading_days = trading_days[trading_days<=self.reg_data.major_axis[-1]]

        # 上市标记为1，找到那些为1的，然后将false全改为nan，再向前填充true，即可得到is_enlisted
        # 即一旦上市后，之后的is_enlisted都为true
        is_enlisted = list_status == 1
        is_enlisted = is_enlisted.replace(False, np.nan)
        is_enlisted = is_enlisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_enlisted = is_enlisted.reindex(trading_days, method='ffill')
        # 将股票索引对其，以保证fillna时可以填充所有的股票
        is_enlisted = is_enlisted.reindex(columns=self.reg_data.minor_axis)
        is_enlisted = is_enlisted.fillna(0).astype(np.int)
        self.is_enlisted = is_enlisted

    # 按照论文的方式, 取得一些数据的统计量, 在data description中给我们一些信息
    def describe_reg_data(self):
        # 对dd的平均数做一个统计
        mean_dd = self.reg_data['dd'].mean(1).mean()
        mean_negE = self.reg_data['negEPS'].mean(1).mean()
        mean_vp = self.strategy_data.factor.iloc[0].mean(1).mean()
        data_mean = pd.Series({'mean_dd':mean_dd, 'mean_negEPS':mean_negE,
                               'mean_vp':mean_vp})
        std_vp = self.strategy_data.factor.iloc[0].std(1).std()
        data_std = pd.Series({'std_dd':np.nan, 'mean_negEPS':np.nan,
                              'mean_vp':std_vp})
        data_describe = pd.DataFrame({'mean':data_mean, 'std':data_std})
        data_describe.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'data_describe.csv', na_rep='N/A', encoding='GB18030')
        pass

    # 描述数据
    def data_description(self):
        self.get_dividend()
        self.get_reg_data()
        # 过滤股票池
        for item, df in self.reg_data.iteritems():
            self.reg_data[item] = self.reg_data[item].where(self.strategy_data.if_tradable['if_inv'],
                                                            np.nan)
        self.strategy_data.discard_uninv_data()
        self.describe_reg_data()
    
    # 构建因子, 直接读取
    def construct_factor(self):
        vp = data.read_data(['vp'], shift=False)
        self.strategy_data.factor = vp








if __name__ == '__main__':
    ri = residual_income()
    ri.get_dividend()
    ri.get_reg_data()
    ri.generate_holding_days(holding_freq='w', start_date='2017-06-14', end_date='2017-06-21', loc=-1)
    ri.predict_eps()
    # ri.predict_eps_parallel()
    ri.get_discount_factor()
    ri.predict_book_value()
    ri.get_value_using_rim()
    data.write_data(ri.strategy_data.factor)




































































































































