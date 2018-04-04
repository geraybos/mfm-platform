import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import copy
import pathos.multiprocessing as mp
import warnings as warnings

from data import data
from strategy_data import strategy_data
from position import position
from factor_base import factor_base
from barra_base import barra_base

# 基于barra base的喜岳投资专用的xy_base, 改动包括加入short term reversal因子,
# 以及市值及nls使用总市值

class xy_base(barra_base):
    """ This is the class of multifactor model base for xy-inv, in addtion to barra base,
    it adds a short-term reversal factor, uses TOTAL market value for lncap and nls.

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


    # 使用总市值
    #######################################################################################################
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
    #######################################################################################################

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
            # written_filename = []
            # for i in self.base_data.factor.items:
            #     written_filename.append(i+self.filename_appendix)
            # xy_base主要是用作数据库更新, 数据库里面决定不存因子原始值了, 因此不再储存风格因子原始值
            # data.write_data(self.base_data.factor, file_name=written_filename, separate=True)
            # print('Style factor data have been saved!\n')
            # 储存因子暴露
            data.write_data(self.base_data.factor_expo, file_name='xy_factor_expo'+self.filename_appendix)
            print('factor exposure data has been saved!\n')

    # 回归计算各个基本因子的因子收益
    def get_base_factor_return(self, *, if_save=False):
        # 初始化储存因子收益的dataframe, 以及股票specific return的dataframe
        self.base_factor_return = pd.DataFrame(np.nan, index=self.base_data.factor_expo.major_axis,
                                             columns=self.base_data.factor_expo.items)
        self.specific_return = pd.DataFrame(np.nan, index=self.base_data.factor_expo.major_axis,
                                            columns=self.base_data.factor_expo.minor_axis)
        self.r_squared = pd.Series(np.nan, index=self.base_data.factor_expo.major_axis)
        # 因子暴露要用上一期的因子暴露，用来加权的市值要用上一期的市值
        lag_factor_expo = self.base_data.factor_expo.shift(1).reindex(
                          major_axis=self.base_data.factor_expo.major_axis)
        lag_mv = self.base_data.stock_price.ix['FreeMarketValue'].shift(1)
        # 循环回归，计算因子收益
        for time, temp_data in self.base_factor_return.iterrows():
            outcome = strategy_data.constrained_gls_barra_base(
                       self.base_data.stock_price.ix['daily_simple_return', time, :],
                       lag_factor_expo.ix[:, time, :],
                       weights = np.sqrt(lag_mv.ix[time, :]),
                       indus_ret_weights = lag_mv.ix[time, :],
                       n_style=self.n_style, n_indus=self.n_indus)
            self.base_factor_return.ix[time, :] = outcome[0]
            self.specific_return.ix[time, :] = outcome[1]
            self.r_squared.ix[time] = outcome[2]
        print('get xy factor return completed...\n')

        # 如果需要储存, 则储存因子收益数据
        # 如果要储存因子数据, 必须保证当前股票池和文件后缀是完全一致的, 否则报错
        if not self.is_update and if_save:
            assert self.filename_appendix[1:] == self.base_data.stock_pool, 'Error: The stock ' \
            'pool of base is different from filename appendix, in order to avoid possible ' \
            'data loss, the saving procedure has been terminated! \n'

            data.write_data(self.base_factor_return, file_name='xy_factor_return'+self.filename_appendix)
            data.write_data(self.specific_return, file_name='xy_specific_return' + self.filename_appendix)
            print('The xy factor return has been saved! \n')

    # xy_base主要是提供给数据库使用, 不进行本地的数据更新工作
    def update_factor_base_data(self, *, start_date=None):
        warnings.warn("XY Base does NOT have a local updating system since it's updated in SQL "
                      "database, instead you can reconstruct the base locally.")
        pass

    # 构建风险预测的函数
    def construct_risk_forecast_parallel(self, *, freq='a', covmat_sample_size=504, var_half_life=84,
            corr_half_life=504, var_nw_lag=5, corr_nw_lag=2, vra_sample_size=252, vra_half_life=42,
            eigen_adj_sims=1000, scaling_factor=1.4, specvol_sample_size=360, specvol_half_life=84,
            specvol_nw_lag=5, shrinkage_parameter=0.1):
        # 将freq转到forecast_step
        freq_map = {'a': 252, 'm': 21, 'w': 5}
        forecast_steps = freq_map[freq]

        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()

        # 估计原始的特征收益波动率
        self.get_initial_spec_vol_parallel(sample_size=specvol_sample_size, spec_var_half_life=specvol_half_life,
                                           nw_lag=specvol_nw_lag, forecast_steps=forecast_steps)

        # 对特征波动率进行vra
        self.get_vra_spec_vol(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 暂时不做bayesian shrinkage
        self.get_bayesian_shrinkage_spec_vol(shrinkage_parameter=shrinkage_parameter)

        # 将spec vol改成spec var
        self.spec_var = self.bs_spec_vol.pow(2)

        # 估计原始协方差矩阵
        self.get_initial_cov_mat_parallel(sample_size=covmat_sample_size, var_half_life=var_half_life,
                                          corr_half_life=corr_half_life, var_nw_lag=var_nw_lag,
                                          corr_nw_lag=corr_nw_lag, forecast_steps=forecast_steps)

        # 进行vra
        self.get_vra_cov_mat(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 进行特征值调整
        self.get_eigen_adjusted_cov_mat_parallel(n_of_sims=eigen_adj_sims, scaling_factor=scaling_factor,
                                                 simed_sample_size=covmat_sample_size)

        # 为了和更新数据兼容, 此时算出的协方差矩阵和特定风险的时间段都是只有更新时间段的
        # 由于更新时间段不会调用这个函数, 这个函数是研究的时候算全时间段使用的, 因此此时这里只有从
        # 第一个超过sample_size的那天开始的索引, 且一般协方差矩阵和特定风险的时间段还不一样(因为sample_size不同)
        # 因此为了在研究中方便使用, 要把其统一reindex成factor return的时间
        self.spec_var = self.spec_var.reindex(index=self.specific_return.index)
        self.eigen_adjusted_cov_mat = self.eigen_adjusted_cov_mat.reindex(items=self.base_factor_return.index)

        # 储存数据
        data.write_data(self.eigen_adjusted_cov_mat, file_name='xy_riskmodel_covmat_'+self.base_data.stock_pool)
        data.write_data(self.spec_var, file_name='xy_riskmodel_specvar_'+self.base_data.stock_pool)

        data.write_data(self.initial_daily_spec_vol.pow(2).reindex(index=self.specific_return.index),
                        file_name='xy_riskmodel_dailyspecvar_'+self.base_data.stock_pool)
        data.write_data(self.daily_var_forecast.reindex(index=self.base_factor_return.index),
                        file_name='xy_riskmodel_dailyfacvar_'+self.base_data.stock_pool)

if __name__ == '__main__':
    pool = 'zz500'
    xy = xy_base(stock_pool=pool)
    xy.try_to_read = False
    xy.construct_factor_base(if_save=True)
    xy.get_base_factor_return(if_save=True)
    xy.construct_risk_forecast_parallel(eigen_adj_sims=1000)






































































