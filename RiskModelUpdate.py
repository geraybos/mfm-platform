import sys
import pandas as pd
from datetime import datetime
from barra_base_prod import barra_base_prod
from database_prod import database_prod
from db_engine import db_engine

if __name__ == '__main__':
    # 打印当前日期
    print("###########################################################################\n\n"
          "Risk Model Data Update on {0}\n\n"
          "###########################################################################\n".
          format(datetime.now().date().strftime('%Y-%m-%d')))

    update_start_time = None
    stock_pool = ('all', 'hs300', 'zz500')

    # 建立risk model数据库引擎
    rm_engine = db_engine(server_type='mssql', driver='pymssql', username='rmreader', password='OP#567890',
                          server_ip='192.168.66.12', port='1433', db_name='RiskModel', add_info='')
    today = pd.Timestamp(format(datetime.now().date().strftime('%Y-%m-%d')))
    # RawData数据库里的最近和次近日期, 次新日期即为有效数据的日期
    latest_date = rm_engine.get_original_data("select max(DataDate) from RawData").iloc[0, 0]
    sub_latest_date = rm_engine.get_original_data("select max(DataDate) from RawData where "
        "DataDate < '" + str(latest_date) + "' ").iloc[0, 0]

    # 判断是否传入了更新开始日期
    if len(sys.argv) >= 2:
        try:
            pd.Timestamp(sys.argv[1])
        except ValueError:
            raise ValueError('Please enter valid datetime string for update_start_time argument!\n')
        update_start_time = pd.Timestamp(sys.argv[1])
        # 更新开始日, 为raw data数据库中最新日期和更新开始日的较小值, 代表了RawData开始更新的时间,
        # 也会是base data和risk forecast的更新开始日
        update_start_time = min(update_start_time, latest_date)
    else:
        update_start_time = latest_date

    # 如果潜在更新开始日是今天, 即此次更新将从今天更新到今天, 则可能造成潜在风险, 此时latest_date一定为今天
    # 因此需要将更新日期调整到次新日期, 一般来说, 即从上一个交易日开始更新.
    if update_start_time == today:
        print("Warning: potential update start date is today, in order to avoid potential running "
              "error, update start date is automatically reset to {0}, which is the " \
              "second-latest trading day in RawData database!\n".format(sub_latest_date))
        update_start_time = sub_latest_date

    # 确定更新开始日之后, 在指示数据是否更新的UpdateIndicator中, 将latest valid day改成更新开始这天的
    # 前一个交易日, 这样保证如果更新失败, 有效数据日期为更新时间段的前一天
    tradingday_before_update = rm_engine.get_original_data("select max(DataDate) from RawData where "
        "DataDate < '" + str(update_start_time) + "' ").iloc[0, 0]
    rm_engine.engine.execute("delete from UpdateIndicator where 1=1; insert into UpdateIndicator "
        "(latest_valid_date, update_date) values ('" + str(tradingday_before_update) + "', '" + \
        str(pd.Timestamp(format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))) + "') ")

    # 判断是否传入了投资域
    if len(sys.argv) >= 3:
        assert sys.argv[2] in ['all', 'hs300', 'zz500'], 'Please enter valid stock pool!\n'
        stock_pool = (sys.argv[2], )

    # 计算数据
    dbp = database_prod()
    dbp.update_data_from_db(start_date=update_start_time)
    for p in stock_pool:
        bbp1 = barra_base_prod(stock_pool=p)
        bbp2 = barra_base_prod(stock_pool=p)
        bbp1.update_factor_base_data(start_date=update_start_time)
        bbp2.update_risk_forecast(start_date=update_start_time)
        print('StockPool {0} has been updated!\n'.format(p))

    # 更新完成, 说明更新成功, 将UpdateIndicator中的latest valid date改成最近的有效更新日,
    # 如果latest date不是今天, 则有效更新日是以下两种情况:
    # 1. latest date, 此时对应每天第一次更新, 或者在非交易日时运行程序的情况.
    # 2. 更新好的数据库里, 小于today的第一天, 此时对应一次性更新了2天及2天以上数据的情况
    # 如果latest date是今天, 则有效更新日是sub latest date,
    # 此时对应的情况是, 同一天第二次(或以上)运行更新程序.
    # 无论是以上哪种情况, 最近有效更新日一定是数据库中小于today的第一天
    latest_valid_date = rm_engine.get_original_data("select max(DataDate) from RawData where "
                                                    "DataDate < '" + str(today) + "' ").iloc[0, 0]

    rm_engine.engine.execute("delete from UpdateIndicator where 1=1; insert into UpdateIndicator "
        "(latest_valid_date, update_date) values ('" + str(latest_valid_date) + "', '" + \
        str(pd.Timestamp(format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))) + "') ")

    pass

