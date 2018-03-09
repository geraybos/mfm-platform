#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:13:36 2017

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from db_engine import db_engine

# 维护数据库的类

class database(object):
    """ This is the class of handling database, including fetch and update data from database

    foo
    """
    def __init__(self, *, start_date=None, end_date=pd.Timestamp(datetime.now().date().strftime('%Y-%m-%d')),
                 market="83"):
        # 储存交易日表
        self.trading_days = pd.Series()
        # 数据库取出来后整理成型的数据
        self.data = data()
        # 聚源数据库的引擎
        self.initialize_jydb()
        # smart quant数据库引擎，取常用的行情数据
        self.initialize_sq()
        # smart quant数据库中取出的数据
        self.sq_data = pd.DataFrame()
        # 朝阳永续数据库引擎，取分析师预期数据
        self.initialize_gg()
        self.initialize_gg_new()
        # 所取数据的开始、截止日期，市场代码
        self.start_date = start_date
        self.end_date = end_date
        self.market = market
        # 标记本次取数据是否为更新数据
        self.is_update = False

    # 初始化jydb
    def initialize_jydb(self):
        self.jydb_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='JYDB', add_info='')

    # 初始化sq
    def initialize_sq(self):
        self.sq_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                   server_ip='192.168.66.12', port='1433', db_name='SmartQuant', add_info='')

    # 初始化朝阳永续
    def initialize_gg(self):
        self.gg_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='GOGOAL', add_info='')
    # 初始化朝阳永续2.0版本的数据库
    def initialize_gg_new(self):
        self.gg_new_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='GOGOAL2', add_info='')

    # 取交易日表
    def get_trading_days(self):
        sql_query = "select TradingDate as trading_days from QT_TradingDayNew where SecuMarket=" +\
                    self.market +" and IfTradingDay=1 "
        # 如果指定了开始结束日期，则选取开始结束日期之间的交易日
        if isinstance(self.start_date, pd.Timestamp):
            sql_query = sql_query + "and TradingDate>=" + "'" + str(self.start_date) + "' "
        if isinstance(self.end_date, pd.Timestamp):
            sql_query = sql_query + "and TradingDate<=" + "'" + str(self.end_date) + "' "
        sql_query = sql_query + 'order by trading_days'

        # 取数据
        trading_days = self.jydb_engine.get_original_data(sql_query)
        self.trading_days = trading_days['trading_days']

    # 设定数据的index和columns，index以交易日表为准，columns以sq中的return daily里的股票为准
    def get_labels(self):
        sql_query = "select distinct SecuCode from ReturnDaily where TradingDay <= '" + \
                    str(self.trading_days.iloc[-1]) + "' order by SecuCode"
        column_label = self.sq_engine.get_original_data(sql_query)
        column_label = column_label.ix[:, 0]
        index_label = self.trading_days

        # data中的所有交易日和股票数据都以这两个label为准，包括benchmark
        self.data.stock_price = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.raw_data = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.benchmark_price = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.if_tradable = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.const_data = pd.DataFrame(index=index_label)

    # 取ClosePrice_adj数据，将data中的panel数据index和columns都设置为ClosePrice_adj的index和columns
    # 先将所有的数据都取出来，之后不用再次从sq中取
    def get_sq_data(self):
        sql_query = "select TradingDay, SecuCode, PrevClosePrice, "\
                    "TurnoverVolume as Volume, TurnoverValue, TotalShares as Shares, " \
                    "NonRestrictedShares as FreeShares, MarketCap as MarketValue, " \
                    "FloatMarketCap as FreeMarketValue, IndustryNameNew as Industry, "\
                    "IfSuspended as is_suspended "\
                    "from ReturnDaily where "\
                    "IfTradingDay=1 and TradingDay>='" + str(self.trading_days.iloc[0]) + "' and TradingDay<='" + \
                    str(self.trading_days.iloc[-1]) + "' order by TradingDay, SecuCode"
        self.sq_data = self.sq_engine.get_original_data(sql_query)

        # 提取sq_data里所需要的各种数据
        self.get_PrevClosePrice()
        self.get_Volume()
        self.get_value_and_vwap()
        self.get_total_and_free_mv()
        self.get_total_and_free_shares()
        self.get_Industry()
        self.get_is_suspended()

    # 取PrevClosePrice, 可用来算涨跌停价格, 也可用来算日收益率(后复权)
    def get_PrevClosePrice(self):
        PrevClosePrice = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                  values='PrevClosePrice', aggfunc='first')
        self.data.stock_price['PrevClosePrice'] = PrevClosePrice

    # 取volumne
    def get_Volume(self):
        Volume = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                          values='Volume', aggfunc='first')
        self.data.stock_price['Volume'] = Volume

    # 取turnover value以及vwap
    def get_value_and_vwap(self):
        TurnoverValue = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                 values='TurnoverValue', aggfunc='first')
        self.data.stock_price['TurnoverValue'] = TurnoverValue
        vwap = self.data.stock_price['TurnoverValue'].div(self.data.stock_price['Volume'])
        self.data.stock_price['vwap'] = vwap

    # 取total shares和free shares
    def get_total_and_free_shares(self):
        Shares = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                          values='Shares', aggfunc='first')
        self.data.stock_price['Shares'] = Shares
        FreeShares = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                              values='FreeShares', aggfunc='first')
        self.data.stock_price['FreeShares'] = FreeShares

    # 取total mv和free mv
    def get_total_and_free_mv(self):
        MarketValue = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                               values='MarketValue', aggfunc='first')
        self.data.stock_price['MarketValue'] = MarketValue
        FreeMarketValue = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                   values='FreeMarketValue', aggfunc='first')
        self.data.stock_price['FreeMarketValue'] = FreeMarketValue

    # 取行业标签
    def get_Industry(self):
        Industry = self.sq_data.pivot(index='TradingDay', columns='SecuCode', values='Industry')
        self.data.raw_data['Industry'] = Industry

    # 取是否停牌
    def get_is_suspended(self):
        is_suspended = self.sq_data.pivot(index='TradingDay', columns='SecuCode', values='is_suspended')
        # self.data.if_tradable['is_suspended'] = is_suspended.fillna(0).astype(np.bool)
        self.data.if_tradable['is_suspended'] = is_suspended

    # 取open，close， high， low的价格数据, 注意, 由于聚源数据库在停牌期间会修改收盘价数据, 因此
    # 这里的价格数据全部按照复权因子一样的算法进行计算, 即停牌期间的数据一律不要, 用停牌前最后一天的数据来填充
    def get_ochl(self, *, first_date=pd.Timestamp('1900-01-01')):
        # 去高开低收数据, 以及停牌标记数据, 与复权因子数据一样, 要从first_date开始取
        sql_query = "select TradingDay, SecuCode, OpenPrice, HighPrice, LowPrice, ClosePrice, " \
                    "IfSuspended as is_suspended from ReturnDaily where " \
                    "IfTradingDay=1 and TradingDay>='" + str(first_date) + "' and TradingDay<='" + \
                    str(self.trading_days.iloc[-1]) + "' order by TradingDay, SecuCode"
        ochl_sus_data = self.sq_engine.get_original_data(sql_query)
        suspended_mark = ochl_sus_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                   values='is_suspended', aggfunc='first')
        # 循环高开低收数据, 每一次循环的流程和处理复权因子数据时一模一样
        ochl = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice']
        for data_name in ochl:
            curr_price_data = ochl_sus_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                        values=data_name, aggfunc='first')
            # 要处理更新数据的时候可能出现的空数据的情况
            if curr_price_data.empty:
                self.data.stock_price[data_name] = np.nan
            # 如果是有数据, 但是是在更新数据, 就不在这里进行计算, 而是在更新函数里衔接了数据后再计算
            elif self.is_update:
                self.data.stock_price[data_name] = curr_price_data.fillna(method='ffill')
                self.data.stock_price[data_name] = self.data.stock_price[data_name].fillna(method='ffill')
            else:
                # 同复权因子的计算一样, 将停牌期间的数据抹去, 然后用停牌前最后一天的数据来填充
                # 首先将价格数据向前填充, 然后reindex到停牌数据上去
                curr_price_data = curr_price_data.fillna(method='ffill').reindex(index=suspended_mark.index,
                            columns=suspended_mark.columns, method='ffill')
                # 对于停牌期间的股票, 要将它们的价格数据都改为nan
                # suspened_mark是nan的股票, 都是那些根本没上市或者退市的股票, 因此如何改都没有影响
                curr_price_data = curr_price_data.where(np.logical_not(suspended_mark), np.nan)
                # 然后用停牌前最后一天的价格数据向前填充, 将停牌期间的价格数据都设置为停牌前最后一天的价格数据
                curr_price_data = curr_price_data.fillna(method='ffill')

                self.data.stock_price[data_name] = curr_price_data

    # 从聚源数据库里取复权因子
    def get_AdjustFactor(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select b.ExDiviDate, b.RatioAdjustingFactor, a.SecuCode from " \
                    "(select distinct InnerCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select * from JYDB.dbo.QT_AdjustingFactor where ExDiviDate >='" + \
                    str(first_date) + "' and ExDiviDate <='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode " \
                    "order by SecuCode, ExDiviDate"
        AdjustFactor = self.jydb_engine.get_original_data(sql_query)
        AdjustFactor = AdjustFactor.pivot_table(index='ExDiviDate', columns='SecuCode', values='RatioAdjustingFactor',
                                                aggfunc='first')

        # 要处理更新数据的时候可能出现的空数据的情况
        if AdjustFactor.empty:
            self.data.stock_price['AdjustFactor'] = np.nan
        # 如果是有数据, 但是是在更新数据, 就不在这里计算复权因子, 因为更新数据时,
        # 在不衔接新旧停牌数据的情况下计算的复权因子是不对的, 因此要在衔接后重新计算, 这里就不用再浪费时间计算了
        elif self.is_update:
            self.data.stock_price['AdjustFactor'] = AdjustFactor.fillna(method='ffill')
            self.data.stock_price['AdjustFactor'] = self.data.stock_price['AdjustFactor'].fillna(method='ffill')
        else:
            # 为了应对停牌期间复权因子发生变化的情况, 这里要取所有的停牌标记
            # 因为如果停牌标记只从取数据那天开始(2007-01-04), 则如果在此之前也一直在停牌,
            # 并且停牌期间复权因子已经发生变化的股票, 仍然会用已经变化的复权因子来进行计算(因为停牌标记只从07-01-04开始)
            # 尽管这些股票, 因为一来买不进去的缘故, 最终不会影响策略的回测结果, 但是它们错误的当天收益率,
            # 可能会影响一些其他指标的计算(例如, beta的计算)
            # 总之, 最后要实现的是把停牌期间因为复权因子变化带来的收益都挪到复牌第一天实现
            sql_query_sus = "select TradingDay, SecuCode, IfSuspended as is_suspended from ReturnDaily where " \
                            "IfTradingDay=1 and TradingDay >= '" + str(first_date) + "' and TradingDay <= ' " + \
                            str(self.trading_days.iloc[-1]) + "' order by TradingDay, SecuCode"
            suspended_mark = self.sq_engine.get_original_data(sql_query_sus)
            suspended_mark = suspended_mark.pivot_table(index='TradingDay', columns='SecuCode',
                                                        values='is_suspended', aggfunc='first')
            # 首先将复权因子向前填充, 然后reindex到停牌数据上去
            AdjustFactor = AdjustFactor.fillna(method='ffill').reindex(index=suspended_mark.index,
                            columns=suspended_mark.columns, method='ffill')
            # 对于停牌期间的股票, 要将它们的复权因子数据都改为nan
            # suspened_mark是nan的股票, 都是那些根本没上市或者退市的股票, 因此如何改都没有影响
            AdjustFactor = AdjustFactor.where(np.logical_not(suspended_mark), np.nan)
            # 然后用停牌前最后一天的复权因子向前填充, 将停牌期间的复权因子都设置为停牌前最后一天的复权因子
            AdjustFactor = AdjustFactor.fillna(method='ffill')

            # 将数据直接储存进data.stock_price, 可以自动reindex
            # 不需要再将数据填成1, 在算adj price的时候进行fill value就可以了
            self.data.stock_price['AdjustFactor'] = AdjustFactor
        pass

    # 复权因子后, 计算调整后的价格
    def get_ochl_vwap_adj(self):
        ochl = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'vwap']
        for data_name in ochl:
            curr_data_adj = self.data.stock_price[data_name].mul(self.data.stock_price['AdjustFactor'].fillna(1))
            self.data.stock_price[data_name + '_adj'] = curr_data_adj
        pass


    # 取上市退市标记，即if_enlisted & if_delisted
    # 如果是第一次取数据（非更新数据）一些数据（包括财务数据）的起始日期并不是第一个交易日，
    # 即第一个交易日的数据在数据库里并不是标记为这个交易日的数据
    # 而是之前的数据，因此在非更新数据的情况下，起始日期选取为一个最小日期，以保证取到所有数据
    def get_list_status(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select a.SecuCode, b.ChangeDate, b.ChangeType from "\
                    "(select distinct InnerCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select ChangeDate, ChangeType, InnerCode from LC_ListStatus where SecuMarket in " \
                    "(83,90) and ChangeDate>='" + str(first_date) + "' and ChangeDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.InnerCode=b.InnerCode "\
                    " order by SecuCode, ChangeDate"
        list_status = self.jydb_engine.get_original_data(sql_query)

        ################################################################################################
        ######################## SPECIAL TREATMENT #####################################################
        # A股历史上有两个更换股票代码的例子, 一个是上海医药, 一个是招商地产变成招商蛇口
        # 后者由于InnerCode发生了变化, 因此在list status上有原先的InnerCode的退市标记
        # 因为list status是根据InnerCode来做的, 而上海医药更改了股票代码, 但是未更改InnerCode
        # 这就使得list status上没有这支股票原来代码的退市标记, 因此这支股票会被系统视为未退市
        # 由于历史上只有这一支股票是这种情况, 因此将其在这里进行单独修改, 修改方式是, 如果数据库日期在
        # 股票代码变更发生时间段, 将上海医药旧代码的那支股票手动加上退市标记. 注意, 不能通过InnerCode添加退市标记,
        # 否则, 上海医药新代码的那支股票也会被标记为退市(因为使用了一样的InnerCode), 一定要通过股票代码来添加

        # 判断取数据的时间段是否包含上海医药的代码变更期 2010年3月5号
        if first_date <= pd.Timestamp('2010-03-05') and \
            self.trading_days.iloc[-1] >= pd.Timestamp('2010-03-05'):
            # 上海医药的老代码是600849, 变化时间是2010年3月5日, 退市的变化类型是4
            list_status = list_status.append(pd.Series({'SecuCode': '600849',
                'ChangeDate': pd.Timestamp('2010-03-05'), 'ChangeType': 4}), ignore_index=True)
        ################################################################################################
        ################################################################################################

        list_status = list_status.pivot_table(index='ChangeDate',columns='SecuCode',values='ChangeType',
                                              aggfunc='first')
        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        if list_status.empty:
            list_status = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        # 向前填充
        list_status = list_status.fillna(method='ffill')

        # 上市标记为1, 恢复上市标记为3, 退市准备期为6, 其他标记为9, 6和9一般都在退市标记4之前, 且数量很少
        # 以上标记都处于上市状态, 2为暂停上市, 虽然暂停上市为非上市状态, 但是由于持有的股票无法清盘
        # 因此对于投资来说, 暂停上市和停牌更相近, 因此将暂停上市也标记为上市状态, 而暂停上市的股票会没有交易数据
        # 因此停牌标记会缺失, 在生成可交易标记的函数中, 停牌标记会把nan填成True, 因此暂停上市可以被正确的视作停牌
        # 详情见data.generate_if_tradable. 因此, 状态标记为1, 2, 3, 6, 9的股票都是上市状态
        # 另外注意, 新版的上市标记和以前不同, 并非只要上市, 之后的标记都是True, 而是退市后会变成False
        is_enlisted = list_status.isin((1, 2, 3, 6, 9))
        # is_enlisted = is_enlisted.replace(False, np.nan)
        # is_enlisted = is_enlisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_enlisted = is_enlisted.reindex(self.data.stock_price.major_axis, method='ffill')

        # 退市标记为4，只有为4的时候, 才认为股票真的退市了. 一般情况下, 股票一旦退市, 之后就不会有状态变化了,
        # 即之后的状态都会是4, 一旦退市, 之后退市标记都是True. 唯一的例外是600018(上港集团)这支股票
        # 由于进行重组, 股票被标记为先退市, 之后又上市了, 而股票代码SecuCode没变, 但是实际上InnerCode是发生了变化的
        # 因此对于InnerCode而言, 实际上是两支证券, 一支退市了, 另一只上市了, 但是对于SecuCode, 体现为这支股票先退市
        # 然后又恢复了上市
        is_delisted = list_status == 4
        # is_delisted = is_delisted.replace(False, np.nan)
        # is_delisted = is_delisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_delisted = is_delisted.reindex(self.data.stock_price.major_axis, method='ffill')

        # 当first_date不为默认值时, 一般认为此时在更新数据, 因此, 更新时间段内的list_status要做额外处理
        # 最重要的是在第一个状态标记前的那些nan的值不能被当做布尔值, 必须保留为nan, 然后在更新数据函数中
        # 利用以前的老数据的状态标记进行填充, 譬如更新区间段并未涉及到某股票的上市时间, 没有上市标记
        # 则is_enlisted会都是false, 则退市之前的那段时间都会被错误的标记成未上市.
        # 解决办法是将list_status的时间索引先reindex到is_enlisted(is_delisted)上去,
        # 然后新的list_status的为nan的地方一律改为nan即可
        if first_date != pd.Timestamp('1900-01-01'):
            new_list_status = list_status.reindex(self.data.stock_price.major_axis, method='ffill')
            is_enlisted = is_enlisted.where(new_list_status.notnull(), np.nan)
            is_delisted = is_delisted.where(new_list_status.notnull(), np.nan)

        # 将股票索引对其, 只有first_date不为默认值时才会没有涵盖所有股票, 在更新数据时
        # 没有出现的股票都是更新时间段状态没有变化的股票, 因此这里不能填nan
        is_enlisted = is_enlisted.reindex(columns=self.data.stock_price.minor_axis)
        is_delisted = is_delisted.reindex(columns=self.data.stock_price.minor_axis)

        self.data.if_tradable['is_enlisted'] = is_enlisted
        self.data.if_tradable['is_delisted'] = is_delisted

    # 取总资产，总负债和所有者权益
    # 取合并报表，即if_merged = 1
    # 报表会进行调整因此每个时间点上可能会有多个不同时间段的报表，类似于前复权
    def get_asset_liability_equity(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select b.InfoPublDate, b.EndDate, a.SecuCode, b.TotalAssets, b.TotalLiability, "\
                    "b.TotalEquity from ("\
                    "select distinct CompanyCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select InfoPublDate, EndDate, CompanyCode, TotalAssets, TotalLiability, " \
                    "TotalShareholderEquity as TotalEquity from LC_BalanceSheetAll where IfMerged=1 "\
                    "and InfoPublDate>='" + str(first_date) + "' and InfoPublDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.CompanyCode=b.CompanyCode "\
                    " order by InfoPublDate, SecuCode, EndDate"
        balance_sheet_data = self.jydb_engine.get_original_data(sql_query)

        # 对资产负债和所有者权益，只取每个时间点上最近的那一期报告，
        # 因为每个时间点上只会使用当前时间点的最新值，不是涉及变化率的计算
        recent_data = balance_sheet_data.groupby(['InfoPublDate', 'SecuCode'],as_index=False).nth(-1)

        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        TotalAssets = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalAssets',
                                              aggfunc='first')
        if TotalAssets.empty:
            TotalAssets = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
            TotalAssets = TotalAssets.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalLiability = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalLiability',
                                                 aggfunc='first')
        if TotalLiability.empty:
            TotalLiability = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
            TotalLiability = TotalLiability.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalEquity = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalEquity',
                                              aggfunc='first')
        if TotalEquity.empty:
            TotalEquity = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
            TotalEquity = TotalEquity.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['TotalAssets'] = TotalAssets
        self.data.raw_data['TotalLiability'] = TotalLiability
        self.data.raw_data['TotalEquity'] = TotalEquity

    # 计算pb
    def get_pb(self):
        # TotalEquity有可能是0, 因此为了不让pb成为inf, 将0改为nan, 即book value为0的股票, pb设为nan
        # 还有一些book value为负的情况, 就不做更改了, 这些股票如果比较bp, 将是最差的股票, 因此不会有影响
        # 特别注意, 由于有这些为负的情况出现, pb和pe一样, 最好都不要直接排序pb或pe, 而是排bp和ep
        pb = self.data.stock_price.ix['MarketValue'].div(self.data.raw_data.ix['TotalEquity'].replace(0, np.nan))
        self.data.raw_data['PB'] = pb

    # 取一致预期净利润
    def get_ni_fy1_fy2(self):
        sql_query = "select STOCK_CODE, CON_DATE, C4*10000 as NI, "\
                    "ROW_NUMBER() over (partition by stock_code, con_date order by rpt_date) as fy from "\
                    "CON_FORECAST_STK where C4_TYPE!=0 and con_date>='" + str(self.trading_days.iloc[0]) + \
                    "' and con_date<='" + str(self.trading_days.iloc[-1]) + \
                    "' order by stock_code, con_date, rpt_date"
        forecast_ni = self.gg_engine.get_original_data(sql_query)
        grouped_data = forecast_ni.groupby(['CON_DATE', 'STOCK_CODE'], as_index=False)
        fy1_data = grouped_data.nth(0)
        fy2_data = grouped_data.nth(1)
        ni_fy1 = fy1_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='NI', aggfunc='first')
        ni_fy2 = fy2_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='NI', aggfunc='first')
        ni_fy1 = ni_fy1.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        ni_fy2 = ni_fy2.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['NetIncome_fy1'] = ni_fy1
        self.data.raw_data['NetIncome_fy2'] = ni_fy2

    # 取一致预期eps
    def get_eps_fy1_fy2(self):
        sql_query = "select STOCK_CODE, CON_DATE, C1 as EPS, " \
                    "ROW_NUMBER() over (partition by stock_code, con_date order by rpt_date) as fy from " \
                    "CON_FORECAST_STK where CON_TYPE!=0 and con_date>='" + str(self.trading_days.iloc[0]) + \
                    "' and con_date<='" + str(self.trading_days.iloc[-1]) + \
                    "' order by stock_code, con_date, rpt_date"
        forecast_eps = self.gg_engine.get_original_data(sql_query)
        grouped_data = forecast_eps.groupby(['CON_DATE', 'STOCK_CODE'], as_index=False)
        fy1_data = grouped_data.nth(0)
        fy2_data = grouped_data.nth(1)
        eps_fy1 = fy1_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='EPS', aggfunc='first')
        eps_fy2 = fy2_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='EPS', aggfunc='first')
        eps_fy1 = eps_fy1.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        eps_fy2 = eps_fy2.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['EPS_fy1'] = eps_fy1
        self.data.raw_data['EPS_fy2'] = eps_fy2

    # 取cash earnings ttm
    def get_cash_related_ttm(self):
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.cash_earnings_ttm, b.cfo_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a left join " \
                    "(select DataDate, CashEquivalentIncrease as cash_earnings_ttm, InnerCode, " \
                    "NetOperateCashFlow as cfo_ttm from " \
                    "TTM_LC_CashFlowStatementAll where DataDate>='" + str(self.trading_days.iloc[0]) + \
                    "' and DataDate<='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode order by DataDate, SecuCode"
        ttm_data = self.sq_engine.get_original_data(sql_query)
        cash_earnings_ttm = ttm_data.pivot_table(index='DataDate', columns='SecuCode',
                                                 values='cash_earnings_ttm', aggfunc='first')
        cash_earnings_ttm = cash_earnings_ttm.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                             method='ffill')
        self.data.raw_data['CashEarnings_ttm'] = cash_earnings_ttm
        cfo_ttm = ttm_data.pivot_table(index='DataDate', columns='SecuCode',
                                       values='cfo_ttm', aggfunc='first')
        cfo_ttm = cfo_ttm.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        self.data.raw_data['CFO_ttm'] = cfo_ttm

    # 取net income ttm
    def get_ni_ttm(self):
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.ni_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a left join " \
                    "(select DataDate, NetProfit as ni_ttm, InnerCode from TTM_LC_IncomeStatementAll " \
                    "where DataDate>='" + str(self.trading_days.iloc[0]) + "' and DataDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.InnerCode=b.InnerCode " \
                    "order by DataDate, SecuCode"
        ttm_data = self.sq_engine.get_original_data(sql_query)
        ni_ttm = ttm_data.pivot_table(index='DataDate', columns='SecuCode',
                                      values='ni_ttm', aggfunc='first')
        ni_ttm = ni_ttm.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                             method='ffill')
        self.data.raw_data['NetIncome_ttm'] = ni_ttm

    # 计算pe ttm
    def get_pe_ttm(self):
        pe_ttm = self.data.stock_price.ix['MarketValue'].div(self.data.raw_data.ix['NetIncome_ttm'])
        self.data.raw_data['PE_ttm'] = pe_ttm

    # 取ni ttm, revenue ttm, eps_ttm的两年增长率
    def get_ni_revenue_eps_growth(self):
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.EndDate, b.ni_ttm, b.revenue_ttm, b.eps_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a " \
                    "left join (select DataDate, InnerCode, EndDate, NetProfit as ni_ttm, " \
                    "TotalOperatingRevenue as revenue_ttm, BasicEPS as eps_ttm from TTM_LC_IncomeStatementAll_8Q " \
                    "where DataDate>='" + str(self.trading_days.iloc[0]) + \
                    "' and DataDate<='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode order by DataDate, SecuCode, EndDate"
        ttm_data_8q = self.sq_engine.get_original_data(sql_query)
        # 两年增长率，直接用每个时间点上的当前quarter的ttm数据除以8q以前的ttm数据减一
        grouped_data = ttm_data_8q.groupby(['DataDate', 'SecuCode'])
        # 定义计算两年增长率的函数
        from strategy_data import strategy_data
        def calc_growth(s):
            # 数据的期数, 有些数据可能并没有8期
            no_of_data = s.shape[0]
            # 根据数据的期数计算annualized term
            ann_term = no_of_data/4
            growth = strategy_data.get_ni_growth(s, lag=no_of_data-1, annualize_term=ann_term)
            return growth.iloc[-1]
        growth_data = grouped_data['ni_ttm','revenue_ttm','eps_ttm'].apply(calc_growth)
        time_index = growth_data.index.get_level_values(0)
        stock_index = growth_data.index.get_level_values(1)

        ni_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index,
                                                   values='ni_ttm', aggfunc='first')
        ni_ttm_growth_8q = ni_ttm_growth_8q.fillna(method='ffill').\
            reindex(self.data.stock_price.major_axis, method='ffill').replace(np.inf, np.nan)
        revenue_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index,
                                                        values='revenue_ttm', aggfunc='first')
        revenue_ttm_growth_8q = revenue_ttm_growth_8q.fillna(method='ffill').\
            reindex(self.data.stock_price.major_axis, method='ffill').replace(np.inf, np.nan)
        eps_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index,
                                                    values='eps_ttm', aggfunc='first')
        eps_ttm_growth_8q = eps_ttm_growth_8q.fillna(method='ffill').\
            reindex(self.data.stock_price.major_axis, method='ffill').replace(np.inf, np.nan)
        self.data.raw_data['NetIncome_ttm_growth_8q'] = ni_ttm_growth_8q
        self.data.raw_data['Revenue_ttm_growth_8q'] = revenue_ttm_growth_8q

        self.data.raw_data['EPS_ttm_growth_8q'] = eps_ttm_growth_8q

    # 取指数行情数据
    def get_index_price(self):
        sql_query = "select b.TradingDay, a.SecuCode, b.ClosePrice, b.OpenPrice from "\
                    "(select distinct InnerCode, SecuCode from SecuMain "\
                    "where SecuCode in ('000016','000300','000902','000905','000906','H00016','H00300'," \
                    "'H00905','H00906', '399005', '399006', '399333', '399606') and SecuCategory=4) a "\
                    "left join (select InnerCode, TradingDay, ClosePrice, OpenPrice from QT_IndexQuote "\
                    "where TradingDay>='" + str(self.trading_days.iloc[0]) + "' and TradingDay<='" + \
                    str(self.trading_days.iloc[-1]) + "') b "\
                    "on a.InnerCode=b.InnerCode order by TradingDay, SecuCode"
        index_data = self.jydb_engine.get_original_data(sql_query)
        index_close_price = index_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                   values='ClosePrice', aggfunc='first')
        index_close_price = index_close_price.reindex(self.data.stock_price.major_axis)
        index_open_price = index_data.pivot_table(index='TradingDay', columns='SecuCode',
                                                  values='OpenPrice', aggfunc='first')
        index_open_price = index_open_price.reindex(self.data.stock_price.major_axis)
        # 鉴于指数行情的特殊性，将指数行情都存在benchmark price中的每个item的第一列
        index_name = {'000016': 'sz50', '000300': 'hs300', '000902': 'zzlt',
                      '000905': 'zz500', '000906': 'zz800', '399005': 'zxb', '399006': 'cyb'}
        for key in index_name:
            self.data.benchmark_price.ix['ClosePrice_'+index_name[key], :, 0] = index_close_price[key].values
            self.data.benchmark_price.ix['OpenPrice_'+index_name[key], :, 0] = index_open_price[key].values
        # 全收益指数只有收盘价, 而且没有中证流通指数的全收益
        index_adj_name = {'H00016':'adj_sz50', 'H00300':'adj_hs300', 'H00905':'adj_zz500', 'H00906':'adj_zz800',
                          '399333':'adj_zxb', '399606': 'adj_cyb'}
        for key in index_adj_name:
            self.data.benchmark_price.ix['ClosePrice_'+index_adj_name[key], :, 0] = index_close_price[key].values
        pass

    # 取指数权重数据
    def get_index_weight(self, *, first_date=pd.Timestamp('1900-01-01')):
        # 一些不那么重要的指数的权重数据,如中小板, 创业板指数, 没有放入SmartQuant里, 因此还是需要从这里取
        # 这些不重要的指数主要是用作研究用, 实际上并不能做空这些指数
        sql_query_minor = "select b.EndDate, a.SecuCode as index_code, c.SecuCode as comp_code, " \
                    "b.Weight/100 as Weight from (select distinct InnerCode, SecuCode from SecuMain "\
                    "where SecuCode in ('399005', '399006') and SecuCategory=4) a "\
                    "left join (select EndDate, IndexCode, InnerCode, Weight from LC_IndexComponentsWeight "\
                    "where EndDate>='" + str(first_date) + "' and EndDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b "\
                    "on a.InnerCode=b.IndexCode "\
                    "left join (select distinct InnerCode, SecuCode from SecuMain) c "\
                    "on b.InnerCode=c.InnerCode "\
                    "order by EndDate, index_code, comp_code "
        weight_data_minor = self.jydb_engine.get_original_data(sql_query_minor)

        # 从衔接了聚源数据和国泰安数据(2015年后)的表中取指数权重数据), 这个数据从2015年开始就有日度的指数权重数据了
        sql_query = "select b.EndDate, a.SecuCode as index_code, c.SecuCode as comp_code, " \
                    "b.Weight/100 as Weight from (select distinct InnerCode, SecuCode from " \
                    "SecuMain where SecuCode in ('000016','000300','000902','000905','000906') " \
                    "and SecuCategory=4) a left join (select EndDate, IndexInnerCode, SecuInnerCode, " \
                    "Weight from SmartQuant.dbo.IndexComponentWeight where EndDate>='" + \
                    str(first_date) + "' and EndDate<='" + str(self.trading_days.iloc[-1]) + \
                    "') b on a.InnerCode=b.IndexInnerCode left join (select distinct InnerCode, " \
                    "SecuCode from SecuMain) c on b.SecuInnerCode=c.InnerCode " \
                    "order by EndDate, index_code, comp_code"
        weight_data = self.jydb_engine.get_original_data(sql_query)
        # 把两组数据衔接起来一起使用
        weight_data = weight_data.append(weight_data_minor)
        index_weight = weight_data.pivot_table(index='EndDate', columns=['index_code', 'comp_code'],
                                               values='Weight', aggfunc='first')

        index_name = {'000016': 'sz50', '000300': 'hs300', '000902': 'zzlt',
                      '000905': 'zz500', '000906': 'zz800', '399005': 'zxb', '399006': 'cyb'}
        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        if index_weight.empty:
            index_weight = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,columns=
                                        pd.MultiIndex.from_product([list(index_name.keys()), self.data.stock_price.minor_axis]))

        # 对指数进行循环储存
        for i in index_weight.columns.get_level_values(0).drop_duplicates():
            # 因为这里的数据，一行是所有指数数据都在一起，就会出现，如沪深300在这一期有数据，而中证500一个数据都没有的情况
            # 这种情况在其他数据中不会出现，其他数据是分开储存，如果一期都没有，那么原始数据就不会存在这一期，于是需要分开处理
            curr_weight = index_weight.ix[:, i]
            # 上面提到的更新数据时，出现的没有新数据的情况，不能dropna再填充，因为dropna会变成空dataframe
            if curr_weight.isnull().all().all():
                pass
            else:
                curr_weight = curr_weight.dropna(axis=0, how='all').reindex(curr_weight.index, method='ffill').\
                    reindex(self.data.stock_price.major_axis, method='ffill')
            # 无论是否更新, 都对指数权重做归一化处理, 更新的时候那些某一天全是nan的, 做完归一化处理后也是nan,
            # 因此没有问题, 这些数据会在更新的时候被ffill, 因此同样也是归一化后的, 因为久的数据都是已经被归一化的
            curr_weight = curr_weight.div(curr_weight.sum(1).replace(0, np.nan), axis=0)
            self.data.benchmark_price['Weight_'+index_name[i]] = curr_weight
            # # 将权重数据的nan填上0
            # # 如果为更新数据，则一行全是nan的情况不填，一行有数据的情况才将nan填成0
            # if self.is_update:
            #     self.data.benchmark_price['Weight_'+index_name[i]] = self.data.benchmark_price['Weight_'+index_name[i]].\
            #         apply(lambda x:x if x.isnull().all() else x.fillna(0), axis=1)
            # else:
            #     self.data.benchmark_price['Weight_'+index_name[i]] = \
            #         self.data.benchmark_price['Weight_'+index_name[i]].fillna(0)
            # pass

    # 从现在的因子库里取因子数据
    def get_runner_value(self, runner_id):
        sql_query = "select runnerdate as TradingDay, stockticker as SecuCode, value as factor_value " \
                    "from RunnerValue where runnerdate>='" + str(self.trading_days.iloc[0]) + "' and " \
                    "runnerdate<='" + str(self.trading_days.iloc[-1]) + "' and runnerid=" + str(runner_id) + " " \
                    "order by TradingDay, SecuCode"
        runner_value_data = self.sq_engine.get_original_data(sql_query)
        runner_value = runner_value_data.pivot_table(index='TradingDay', columns='SecuCode', values='factor_value')
        # 处理TradingDay数据类型不对的问题, 将其变为datetime
        runner_value = runner_value.set_index(pd.to_datetime(runner_value.index))

        # 储存数据
        self.data.stock_price['runner_value_'+str(runner_id)] = runner_value

    # 储存数据文件
    def save_data(self):
        data.write_data(self.data.stock_price, separate=True)
        data.write_data(self.data.raw_data, separate=True)
        data.write_data(self.data.benchmark_price, separate=True)
        data.write_data(self.data.if_tradable, separate=True)
        data.write_data(self.data.const_data, file_name='const_data')

    # 取数据的主函数
    # update_time为default时，则为首次取数据，需要更新数据时，传入更新的第一个交易日的时间给update_time即可
    def get_data_from_db(self, *, update_time=pd.Timestamp('1900-01-01')):
        self.get_trading_days()
        self.get_labels()
        self.get_sq_data()
        self.get_ochl(first_date=update_time)
        self.get_AdjustFactor(first_date=update_time)
        self.get_ochl_vwap_adj()
        print('get sq data has been completed...\n')
        self.get_list_status(first_date=update_time)
        print('get list status has been completed...\n')
        self.get_asset_liability_equity(first_date=update_time)
        print('get balancesheet data has been completed...\n')
        self.get_pb()
        print('get pb has been completed...\n')
        self.get_ni_fy1_fy2()
        self.get_eps_fy1_fy2()
        print('get forecast data has been completed...\n')
        self.get_cash_related_ttm()
        print('get cash related ttm has been completed...\n')
        self.get_ni_ttm()
        print('get netincome ttm has been completed...\n')
        self.get_pe_ttm()
        print('get pe_ttm has been completed...\n')
        self.get_ni_revenue_eps_growth()
        print('get growth ttm has been completed...\n')
        self.get_index_price()
        self.get_index_weight(first_date=update_time)
        print('get index data has been completed...\n')

        # 更新数据的情况先不能储存数据，只有非更新的情况才能储存
        if not self.is_update:
           self.save_data()

    # 更新数据的主函数
    def update_data_from_db(self, *, start_date=None, end_date=None):
        # 更新标记
        self.is_update = True
        # 首先读取ClosePrice_adj数据，将其当作更新数据时的参照标签
        data_mark = data.read_data('ClosePrice_adj')
        # 更新的第一天为之前数据标签日期的最后一天
        # 因为有可能当时更新数据的时候，还没有得到那次的数据
        # 因此为了统一，更新的第一天都默认设置为那一天
        last_day = data_mark.iloc[-1].name
        # 如果有传入的指定更新数据的开始时间, 则选取last_day和指定开始时间更早的那天
        if isinstance(start_date, pd.Timestamp):
            self.start_date = min(start_date, last_day)
        else:
            self.start_date = last_day
        # 当衔接新旧数据的时候, 需要丢弃的数据是从开始更新的那一天, 一直到last_day
        dropped_time = data_mark.ix[self.start_date:last_day, :].index

        # 可以设置更新数据的更新截止日，默认为更新到当天
        if isinstance(end_date, pd.Timestamp):
            self.end_date = end_date

        # 更新数据
        self.get_data_from_db(update_time=self.start_date)

        # 读取以前的老数据
        stock_price_name_list = ['ClosePrice_adj', 'OpenPrice_adj', 'HighPrice_adj', 'LowPrice_adj',
                                 'vwap', 'OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice',
                                 'vwap_adj', 'PrevClosePrice', 'AdjustFactor', 'Volume', 'TurnoverValue',
                                 'Shares', 'FreeShares', 'MarketValue', 'FreeMarketValue']
        raw_data_name_list = ['Industry', 'TotalAssets', 'TotalLiability', 'TotalEquity', 'PB', 'NetIncome_fy1',
                              'NetIncome_fy2', 'EPS_fy1', 'EPS_fy2', 'CashEarnings_ttm', 'CFO_ttm', 'NetIncome_ttm',
                              'PE_ttm', 'NetIncome_ttm_growth_8q', 'Revenue_ttm_growth_8q', 'EPS_ttm_growth_8q']
        if_tradable_name_list = ['is_suspended', 'is_enlisted', 'is_delisted']
        benchmark_index_name = ['sz50', 'hs300', 'zzlt', 'zz500', 'zz800', 'zxb', 'cyb']
        benchmark_data_type = ['ClosePrice', 'OpenPrice', 'Weight', 'ClosePrice_adj']
        benchmark_price_name_list = [a+'_'+b for a in benchmark_data_type for b in benchmark_index_name]
        # 注意中证流通指数没有closeprice adj, 即全收益数据
        benchmark_price_name_list.remove('ClosePrice_adj_zzlt')

        old_stock_price = data.read_data(stock_price_name_list, item_name=stock_price_name_list)
        old_raw_data = data.read_data(raw_data_name_list, item_name=raw_data_name_list)
        old_if_tradable = data.read_data(if_tradable_name_list, item_name=if_tradable_name_list)
        old_benchmark_price = data.read_data(benchmark_price_name_list, item_name=benchmark_price_name_list)
        old_const_data = data.read_data('const_data')

        # 新数据中的股票数可能与旧数据已经不同，要将旧数据中的股票索引换成新数据的索引
        new_stock_index = self.data.stock_price.minor_axis
        old_stock_price = old_stock_price.reindex(minor_axis=new_stock_index)
        old_raw_data = old_raw_data.reindex(minor_axis=new_stock_index)
        old_if_tradable = old_if_tradable.reindex(minor_axis=new_stock_index)
        old_benchmark_price = old_benchmark_price.reindex(minor_axis=new_stock_index)

        # 衔接新旧数据
        new_stock_price = pd.concat([old_stock_price.drop(dropped_time, axis=1).sort_index(),
                                     self.data.stock_price.sort_index()], axis=1)
        new_raw_data = pd.concat([old_raw_data.drop(dropped_time, axis=1).sort_index(),
                                  self.data.raw_data.sort_index()], axis=1)
        new_if_tradable = pd.concat([old_if_tradable.drop(dropped_time, axis=1).sort_index(),
                                     self.data.if_tradable.sort_index()], axis=1)
        new_benchmark_price = pd.concat([old_benchmark_price.drop(dropped_time, axis=1).sort_index(),
                                         self.data.benchmark_price.sort_index()], axis=1)
        new_const_data = pd.concat([old_const_data.drop(dropped_time, axis=0).sort_index(axis=1),
                                    self.data.const_data.sort_index(axis=1)], axis=0)

        self.data.stock_price = new_stock_price
        self.data.raw_data = new_raw_data
        self.data.if_tradable = new_if_tradable
        self.data.benchmark_price = new_benchmark_price
        self.data.const_data = new_const_data

        # 对需要的数据进行fillna，以及fillna后的重新计算
        # 主要是那些用到first_date参数的数据，以及涉及这些数据的衍生数据
        self.data.raw_data['TotalAssets'] = self.data.raw_data['TotalAssets'].fillna(method='ffill')
        self.data.raw_data['TotalLiability'] = self.data.raw_data['TotalLiability'].fillna(method='ffill')
        self.data.raw_data['TotalEquity'] = self.data.raw_data['TotalEquity'].fillna(method='ffill')
        self.get_pb()
        # # 注意这两个数据在用旧数据向前填na之后，还要再fill一次na，因为更新的时候出现的新股票，之前的旧数据因为重索引的关系，也是nan
        # self.data.if_tradable['is_enlisted'] = self.data.if_tradable['is_enlisted'].\
        #     fillna(method='ffill').fillna(0).astype(np.bool)
        # self.data.if_tradable['is_delisted'] = self.data.if_tradable['is_delisted'].\
        #     fillna(method='ffill').fillna(0).astype(np.bool)
        # # 指数权重数据也是这样, 在用旧数据向前填na之后, 还要再fill一次na, 原因与上面的上市退市数据是一样的
        # for index_name in benchmark_index_name:
        #     self.data.benchmark_price['Weight_'+index_name] = self.data.benchmark_price['Weight_'+index_name].\
        #         fillna(method='ffill').fillna(0.0)
        # 现在上市退市标记和指数权重数据只进行向前填充, 不再进行将所有nan填成0的步骤
        self.data.if_tradable['is_enlisted'] = self.data.if_tradable['is_enlisted'].\
            fillna(method='ffill')
        self.data.if_tradable['is_delisted'] = self.data.if_tradable['is_delisted'].\
            fillna(method='ffill')
        # 权重数据稍有不同, 只能整行整行的fillna, 而不能按个股fillna, 否则会把以前在指数中,
        # 现在不在指数中的那些股票的权重填到现在来, 这样是不对的, 只有当一行全是nan是才按行对这一行进行fillna
        for index_name in benchmark_index_name:
            self.data.benchmark_price['Weight_'+index_name] = self.data.benchmark_price['Weight_'+index_name].\
                dropna(how='all', axis=0).reindex(index=self.data.benchmark_price.major_axis, method='ffill')
        # 同下面的复权因子一样, 在衔接了停牌标记后, 在此进行高开低收价格数据的计算
        ochl = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice']
        for data_name in ochl:
            self.data.stock_price[data_name] = self.data.stock_price[data_name].where(np.logical_not(
                self.data.if_tradable['is_suspended']), np.nan).fillna(method='ffill')
        # 复权因子在更新的时候, 需要在衔接了停牌标记后, 在此进行复权因子(以及之后的后复权价格)的计算
        # 同直接取所有的复权因子数据时一样, 首先将停牌期间的复权因子设置为nan,
        # 然后使用停牌前最后一天的复权因子向前填充, 使得停牌期间的复权因子变化反映在复牌后第一天,
        # 不需要再将数据填成1, 在算adj price的时候进行fill value就可以了
        self.data.stock_price['AdjustFactor'] = self.data.stock_price['AdjustFactor'].where(
            np.logical_not(self.data.if_tradable['is_suspended']), np.nan).\
            fillna(method='ffill')
        # 因为衔接并向前填充了复权因子, 因此要重新计算后复权价格, 否则之前的后复权价格将是nan
        self.get_ochl_vwap_adj()

        self.save_data()

        # 重置标记
        self.is_update = False

if __name__ == '__main__':
    import time
    start_time = time.time()
    db = database(start_date=pd.Timestamp('2007-01-01'), end_date=pd.Timestamp('2018-01-17'))
    # db.is_update=False
    # db.get_data_from_db()
    # db.update_data_from_db()
    db.get_trading_days()
    db.get_labels()
    db.get_list_status(first_date=pd.Timestamp('20070101'))
    # db.get_AdjustFactor()
    # db.get_sq_data()
    # db.get_index_price()
    # db.get_index_weight()
    # data.write_data(db.data.benchmark_price)
    # for runner_id in [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,24,27,30,31,32,35,36]:
    # db.get_runner_value(3)
    # data.write_data(db.data.stock_price.ix['runner_value_3'], file_name='runner_value_3')
    # data.write_data(db.data.stock_price.ix['runner_value_63'], file_name='runner_value_63')
    print("time: {0} seconds\n".format(time.time()-start_time))













































































