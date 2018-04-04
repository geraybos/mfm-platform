#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:06:26 2017

@author: lishiwang
"""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# 建立数据库引擎的类

class db_engine(object):
    """This is the class of database engine.
    
    foo
    """
    def __init__(self, *, server_type, driver, username, password, server_ip, port, db_name, 
                 add_info=''):
        # 创建引擎的string
        self.engine_str = server_type+'+'+driver+'://'+username+':'+password+'@'+server_ip+':'+ \
                          port+'/'+db_name 
        if add_info != '':
            self.engine_str = self.engine_str+'?'+add_info
        # 创建引擎
        self.engine = create_engine(self.engine_str)
        
    # 取数据
    def get_original_data(self, sql_query):
        """ Fetch data from this database engine using specified sql query.

        :param sql_query: (string) sql query which used to fetch data from this database engine
        :return: (pd.DataFrame) outcome data with time as index and fields as columns
        """
        data = pd.read_sql(sql_query, self.engine)
        return data

    # 向数据库中插入数据的函数
    def insert_df(self, df, table, *, rows_per_batch=1000):
        # 构建插入的sql语句的开头部分
        sql_prefix = "insert into " + table + " "
        # df的columns为表中的字段名
        sql_prefix += str(tuple(df.columns)).replace("'", '') + " "

        # 初始化插入数据需要用到的参数, 由于多次插入数据可能是在同一个db engine实例中,
        # 因此要保证每次调用insert df函数时, 插入数据的相关参数全部重置, 以免发生错误
        self._num_columns = None
        self._row_placeholders = None
        self._num_rows_previous = None
        self._all_placeholders = None
        self._sql_prefix = sql_prefix
        self._sql_insert = None

        row_count = 0
        param_list = list()
        for df_row in df.itertuples():
            param_list.append(tuple(df_row[1:]))
            row_count += 1
            if row_count >= rows_per_batch:
                self.send_insert(param_list)
                row_count = 0
                param_list = list()
        self.send_insert(param_list)

    # 向数据库里分段插入数据的函数, 由于数据库有每次插入的限制, 因此只能分段插入
    def send_insert(self, param_list):
        if len(param_list) > 0:
            if self._num_columns is None:
                self._num_columns = len(param_list[0])
                # pymssql的占位符是%s, 或者使用一模一样的%d, pyodbc的占位符为?
                self._row_placeholders = ','.join(['%s' for x in range(self._num_columns)])
            num_rows = len(param_list)
            if num_rows != self._num_rows_previous:
                self._all_placeholders = '({})'.\
                    format('),('.join([self._row_placeholders for x in range(num_rows)]))
                self._sql_insert = f'{self._sql_prefix} VALUES {self._all_placeholders}'
                self._num_rows_previous = num_rows
            params = [int(element) if isinstance(element, np.int64) else element
                      for row_tup in param_list for element in row_tup]
            self.engine.execute(self._sql_insert, params)
        
        
        
        
        
        
        
        
        
        
        
        
        













