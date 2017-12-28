import sys
import pandas as pd
from datetime import datetime
from barra_base_prod import barra_base_prod
from database_prod import database_prod

if __name__ == '__main__':
    # 打印当前日期
    print("###########################################################################\n\n"
          "Risk Model Data Update on {0}\n\n"
          "###########################################################################\n".
          format(datetime.now().date().strftime('%Y-%m-%d')))

    # 判断是否传入了更新开始日期
    if len(sys.argv) >= 2:
        try:
            pd.Timestamp(sys.argv[1])
        except ValueError:
            raise ValueError('Please enter valid datetime string for update_start_time argument!\n')
        update_start_time = pd.Timestamp(sys.argv[1])
        dbp = database_prod()
        dbp.update_data_from_db(start_date=update_start_time)
        # 判断是否传入了投资域
        if len(sys.argv) >= 3:
            assert sys.argv in ['all', 'hs300', 'zz500'], 'Please enter valid stock pool!\n'
            bbp1 = barra_base_prod(stock_pool=sys.argv[2])
            bbp2 = barra_base_prod(stock_pool=sys.argv[2])
            bbp1.update_factor_base_data(start_date=update_start_time)
            bbp2.update_risk_forecast(start_date=update_start_time)
        else:
            for p in ['all', 'hs300', 'zz500']:
                bbp1 = barra_base_prod(stock_pool=p)
                bbp2 = barra_base_prod(stock_pool=p)
                bbp1.update_factor_base_data(start_date=update_start_time)
                bbp2.update_risk_forecast(start_date=update_start_time)
                print('StockPool {0} has been updated!\n'.format(p))
    else:
        dbp = database_prod()
        dbp.update_data_from_db()
        for p in ['all', 'hs300', 'zz500']:
            bbp1 = barra_base_prod(stock_pool=p)
            bbp2 = barra_base_prod(stock_pool=p)
            bbp1.update_factor_base_data()
            bbp2.update_risk_forecast()
            print('StockPool {0} has been updated!\n'.format(p))