import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
import pathos.multiprocessing as mp

from data import data
from strategy_data import strategy_data
from position import position

# 因子基的类, 里面储存factor base的一些基础函数, 以及风险模型的函数
# 风险模型的函数包括对因子协方差的预测和个股风险的预测

class factor_base(object):
    """ This is the class for factor base, which serves as the base class of any factor base.

    foo
    """

    def __init__(self, *, stock_pool='all'):
        self.base_data = strategy_data()
        self.base_factor_return = pd.DataFrame()
        self.specific_return = pd.DataFrame()
        # 提示factor base的股票池
        self.base_data.stock_pool = stock_pool

    def read_original_data(self):
        pass

    # 在完成全部因子暴露的计算或读取后, 得出该base中风格因子和行业因子的数量
    def get_factor_group_count(self):
        pass

    # 创建因子值读取文件名的函数, 一般来说, 文件名与股票池相对应
    # 但是, 如在多因子研究框架中说到的那样, 为了增强灵活性, 用户可以选择在当前股票池设定下,
    # 读取在其他股票池中计算出的因子值, 然后将这些因子值在股票池内标准化.
    # 只要将在不同股票池中算出的原始因子值, 理解为定义不同的因子就可以了.
    # 这个功能一般很少用到, 暂时放在这里增加灵活性
    def construct_reading_file_appendix(self, *, filename_appendix='default'):
        # 默认就是用到当前的股票池
        if filename_appendix == 'default':
            self.filename_appendix = '_' + self.base_data.stock_pool
        # 有可能会使用到与当前股票池不同的股票池下计算出的因子值
        else:
            self.filename_appendix = filename_appendix
            # 需要输出提示用户, 因为这个改动比较重要, 注意, 这个改动只会影响读取的因子值,
            # 不会影响算出来的因子值, 算出来的因子值还是与当前股票池一样
            print('Attention: The stock pool you specify under which the base factors are calculated '
                  'is different from your base stock pool. Please aware that this will make the definition'
                  'of your base factors different. Also note that this will only affect factors read from'
                  'files, factors which are calculated will not be affected! \n')

    def construct_factor_base(self, *, if_save=False):
        pass

    def get_base_factor_return(self, *, if_save=False):
        pass

    def update_factor_base_data(self):
        pass

    ######################################################################################################
    # 以下为对风险进行预测的部分

    # 对原始的协方差矩阵进行估计
    def get_initial_cov_mat(self, *, sample_size=504, var_half_life=84, corr_half_life=504,
                            var_nw_lag=5, corr_nw_lag=2, forecast_steps=21):

        # 初始化储存原始协方差矩阵的panel
        self.initial_cov_mat = pd.Panel(np.nan, items=self.base_factor_return.index,
            major_axis=self.base_factor_return.columns, minor_axis=self.base_factor_return.columns)
        # 初始化储存日度因子方差预测的dataframe, 用来做volatility regime adjustment
        self.daily_var_forecast = pd.DataFrame(np.nan, index=self.base_factor_return.index,
                                               columns=self.base_factor_return.columns)

        # 开始循环计算原始的cov_mat
        for cursor, time in enumerate(self.base_factor_return.index):
            # 注意, 根据sample size来决定从第几期开始计算
            # if cursor < sample_size - 1:
            if time < pd.Timestamp('2011-05-03'):
                continue

            estimated_outcome = factor_base.initial_cov_estimator(cursor, self.base_factor_return,
                                    sample_size=sample_size, var_half_life=var_half_life,
                                    corr_half_life=corr_half_life, var_nw_lag=var_nw_lag,
                                    corr_nw_lag=corr_nw_lag, forecast_steps=forecast_steps)

            # 将估计得到的var和corr结合起来, 形成当期估计的协方差矩阵
            # 用日数据转移到周或月数据, 需要乘以天数差
            self.initial_cov_mat.ix[time] = estimated_outcome[0]
            # 日度的因子方差预测值, 储存起来做vra的时候会用到
            self.daily_var_forecast.ix[time] = estimated_outcome[1]
            pass

        self.initial_cov_mat.to_hdf('bb_factor_covmat_all', '123')
        self.daily_var_forecast.to_hdf('bb_factor_var_all', '123')

    # 估计原始协方差矩阵的并行版本
    def get_initial_cov_mat_parallel(self, *, sample_size=504, var_half_life=84, corr_half_life=504,
                            var_nw_lag=5, corr_nw_lag=2, forecast_steps=21):
        base_factor_return = self.base_factor_return

        # 定义单次计算的函数
        def one_time_func(cursor):

            estimated_outcome = factor_base.initial_cov_estimator(cursor, base_factor_return,
                                    sample_size=sample_size, var_half_life=var_half_life,
                                    corr_half_life=corr_half_life, var_nw_lag=var_nw_lag,
                                    corr_nw_lag=corr_nw_lag, forecast_steps=forecast_steps)

            return estimated_outcome

        ncpus = 20
        p = mp.ProcessPool(ncpus)
        data_size = np.arange(base_factor_return.shape[0])
        chunksize = int(len(data_size)/ncpus)
        results = p.map(one_time_func, data_size, chunksize=chunksize)
        self.initial_cov_mat = pd.Panel({i: v[0] for i, v in zip(base_factor_return.index, results)})
        self.daily_var_forecast = pd.DataFrame({i: v[1] for i, v  in zip(base_factor_return.index, results)}).T
        # 将因子的排序改为习惯的排序, 而不是按照字母顺序
        self.initial_cov_mat = self.initial_cov_mat.reindex(major_axis=self.base_data.factor_expo.items,
                                                            minor_axis=self.base_data.factor_expo.items)
        self.daily_var_forecast = self.daily_var_forecast.reindex(columns=self.base_data.factor_expo.items)

    # 原始协方差矩阵的估计量函数, 采用statsmodels里面的函数改进性能
    @staticmethod
    def initial_cov_estimator(cursor, complete_factor_return_data, *, sample_size=504,
                              var_half_life=84, corr_half_life=504, var_nw_lag=5, corr_nw_lag=2,
                              forecast_steps=21):
        # 首先创建指数权重序列
        var_weight = strategy_data.construct_expo_weights(var_half_life, sample_size)
        corr_weight = strategy_data.construct_expo_weights(corr_half_life, sample_size)
        # 取当期的数据
        curr_data = complete_factor_return_data.iloc[cursor + 1 - sample_size:cursor + 1, :]
        if curr_data.dropna(axis=0, how='all').shape[0] < sample_size:
            return [pd.DataFrame(), pd.Series()]

        # 如果预测的步数为1, 则nw调整没有意义, 直接使用最简单的方法进行估计即可
        if forecast_steps == 1:
            var_weighted_curr_data = strategy_data.multiply_weights(curr_data, var_weight, multiply_power=0.5)
            curr_var = var_weighted_curr_data.var()
            corr_weighted_curr_data = strategy_data.multiply_weights(curr_data, corr_weight, multiply_power=0.5)
            curr_corr = corr_weighted_curr_data.corr()
            estimated_cov_mat = curr_corr.mul(np.sqrt(curr_var), axis=0). \
                mul(np.sqrt(curr_var), axis=1)
            return [estimated_cov_mat, curr_var]

        # 首先估计var
        # 使用将数据乘以对应的权重, 再计算的方式来进行估计. 在将权重乘在数据上时, 要做一些调整.
        var_weighted_curr_data = strategy_data.multiply_weights(curr_data, var_weight, multiply_power=0.5)
        # 使用算auto covariance的函数, 进行acov的计算
        # 注意, 这样计算时, 那些lag的项会损失一些数据
        curr_acov = var_weighted_curr_data.apply(sm.tsa.acovf, axis=0)
        # 取lag=0到lag=var_nw_lag这些项, 即第1行到第var_nw_lag+1行
        curr_acov = curr_acov.iloc[0:var_nw_lag + 1, :]
        # 生成对各个lag项的对应权重
        nw_lag_weight_var = 2 * (var_nw_lag + 1 - np.arange(0, var_nw_lag + 1)) / (var_nw_lag + 1)
        nw_lag_weight_var[0] = 1
        # 由于是先对数据乘以了权重, 再用新的数据进行acovf的计算, 虽然比每个lag都乘以一次权重加快了速度, 但是却有偏差
        # 具体的偏差来源是, 不lag的数据和lag了n项的数据, 原本应当将权重乘在lag(n)这项数据上, 但是由于事先乘了,
        # 导致这一项数据的每一项的权重变小了, 比例为指数权重中每差n项的权重比例差距. 将这个比例的根号乘在计算出的acovf上即可
        # corr的估计由于数量级无关, 因此常数的偏差是没有影响的, 不需要做调整
        magnitude_adj = np.sqrt((var_weight[-1]/var_weight[-2]) ** (np.arange(var_nw_lag + 1)))
        curr_var = curr_acov.mul(nw_lag_weight_var, axis=0).mul(magnitude_adj, axis=0).sum()

        # 估计corr
        # 同样的将数据乘以对应权重, 但事实上corr的计算与数量级无关,
        # 因此做不做multiply_weights内的常数调整是无所谓的, 只需要将权重乘到数据上去就行
        corr_weighted_curr_data = strategy_data.multiply_weights(curr_data, corr_weight, multiply_power=0.5)
        # 接下来通过算cross-correlation的函数, 计算cc
        # 暂时只想到通过循环来计算, 先建立储存数据的panel
        curr_cc = pd.Panel(np.nan, items=np.arange(corr_nw_lag+1), major_axis=curr_data.columns,
                           minor_axis=curr_data.columns)
        # 根据因子进行循环, 每次循环计算一个因子与其余因子的cross-correlation
        for name, factor in corr_weighted_curr_data.iteritems():
            full_ccf = corr_weighted_curr_data.apply(sm.tsa.stattools.ccf, axis=0, y=factor)
            # 取lag=0到lag=corr_nw_lag这一部分, 注意将循环的因子存在列中, 循环的因子factor是lag的那一项
            # 因此, 为了保证curr_cc中, major_axis是当前量, minor_axis是lag后的量, 要将循环的因子按列储存
            curr_cc.ix[:, :, name] = full_ccf.iloc[0:corr_nw_lag+1, :].values
        # 循环结束后, 生成对各个lag项的对应权重, 注意, 因为要加转置后的矩阵, 因此第一项权重是0.5
        nw_lag_weight_corr = (corr_nw_lag + 1 - np.arange(0, corr_nw_lag + 1)) / (corr_nw_lag + 1)
        nw_lag_weight_corr[0] = 0.5
        # curr_cc = curr_cc.apply(lambda x: x.mul(nw_lag_weight_corr, axis=1), axis=(0, 2))
        for item, df in curr_cc.iteritems():
            curr_cc.ix[item] = df * nw_lag_weight_corr[item]
        curr_corr = curr_cc.sum(0) + curr_cc.transpose(0, 2, 1).sum(0)

        estimated_cov_mat = curr_corr.mul(np.sqrt(curr_var), axis=0). \
            mul(np.sqrt(curr_var), axis=1).mul(forecast_steps)

        # 注意, 返回的daily factor var是不需要做nw调整的
        return [estimated_cov_mat, curr_acov.iloc[0, :]]

    # 对估计出的协方差矩阵做volatility regime adjustment
    def get_vra_cov_mat(self, *, sample_size=252, vra_half_life=42):
        # 初始化储存vra后的协方差矩阵的panel
        self.vra_cov_mat = self.initial_cov_mat * np.nan
        # 计算标准化的因子收益率, 即用每天实现的因子收益, 除以之前预测的当天的因子波动率
        standardized_fac_ret = self.base_factor_return.div(np.sqrt(self.daily_var_forecast.shift(1))).\
            replace([np.inf, -np.inf], np.nan)
        # 计算factor cross-sectional bias statistic
        factor_cs_bias = standardized_fac_ret.pow(2).mean(1).pow(0.5)

        # 初始化储存vra multiplier的seires
        self.vra_multiplier = pd.Series(np.nan, index=self.initial_cov_mat.items)

        # 开始循环计算vra后的协方差矩阵
        for cursor, time in enumerate(self.base_factor_return.index):
            # 根据sample size来决定从第几期开始计算
            # 注意, 因为用来进行调整的factor volatility multiplier(vra multiplier)的计算
            # 要用到预测的因子方差的时间序列, 所以, 最开始的几期的计算会只有很少的时间序列上的点
            if time < pd.Timestamp('2011-05-03'):
            # if cursor < sample_size - 1:
                continue

            vra_multiplier = factor_base.vra_multiplier_estimator(cursor, factor_cs_bias,
                                sample_size=sample_size, vra_half_life=vra_half_life)
            self.vra_multiplier.ix[time] = vra_multiplier

        # 循环结束后, 计算vra_cov_mat, 事实上只需要将vra multiplier的平方乘在每个时间对应的协方差矩阵上即可
        self.vra_cov_mat = self.initial_cov_mat.apply(lambda x: x*self.vra_multiplier.pow(2), axis=0)

        # self.vra_cov_mat.to_hdf('bb_factor_vracovmat_all', '123')

    # 进行volatility regime adjustment的函数, 会返回vra后的协方差矩阵
    @staticmethod
    def vra_multiplier_estimator(cursor, factor_cs_bias, *, sample_size=252, vra_half_life=42):
        # 首先创建指数权重序列
        vra_weight = strategy_data.construct_expo_weights(vra_half_life, sample_size)
        # 取当期的数据
        curr_data = factor_cs_bias.iloc[cursor + 1 - sample_size:cursor + 1]

        # 将数据乘以对应的权重, 注意, 参考vra multiplier的计算方法, bias先平方, 然后权重按照根号幂乘在数据上
        weighted_data = strategy_data.multiply_weights(curr_data**2, vra_weight, multiply_power=1)
        # 根据乘以权重后的数据, 计算vra multiplier
        vra_multiplier = np.sqrt(weighted_data.mean())

        return vra_multiplier

    # 对初步估计出的协方差矩阵, 进行eigenfactor adjustment的函数
    def get_eigen_adjusted_cov_mat(self, *, n_of_sims=1000, scaling_factor=1.4, simed_sample_size=504):
        # 储存特征值调整后的协方差矩阵
        self.eigen_adjusted_cov_mat = self.vra_cov_mat * np.nan

        # 对日期进行循环
        for cursor, time in enumerate(self.vra_cov_mat.dropna().items):
            # if time < pd.Timestamp('2013-07-31'):
            # if cursor <504:
            #     continue
            curr_cov_mat = self.vra_cov_mat.ix[time]
            if (curr_cov_mat==0).all().all():
                continue
            # 放入进行eigen adjustment的函数进行计算
            outcome = factor_base.eigen_factor_adjustment(curr_cov_mat, n_of_sims, scaling_factor,
                                                         simed_sample_size)
            self.eigen_adjusted_cov_mat.ix[time] = outcome[0]

    # 进行eigenfactor adjustment的函数的并行版本
    def get_eigen_adjusted_cov_mat_parallel(self, *, n_of_sims=1000, scaling_factor=1.4, simed_sample_size=504):
        vra_cov_mat = self.vra_cov_mat.dropna()
        # 定义进行单期调整的函数
        def one_time_adjust_func(cursor):
            curr_cov_mat = vra_cov_mat.iloc[cursor]
            # print(vra_cov_mat.items[cursor])
            # 如果cov_mat是nan, 则返回它自己
            if (curr_cov_mat==0).all().all():
                return [curr_cov_mat, np.nan]
            # 放入eigen adjustment的函数进行计算
            outcome = factor_base.eigen_factor_adjustment(curr_cov_mat, n_of_sims, scaling_factor,
                                                         simed_sample_size)
            # 取调整后的cov_mat
            adjusted_cov_mat = outcome[0]
            bias = outcome[1]

            return [adjusted_cov_mat, bias]
        ncpus=20
        p = mp.ProcessPool(ncpus)
        data_size = np.arange(vra_cov_mat.shape[0])
        chunksize = int(len(data_size)/ncpus)
        results = p.map(one_time_adjust_func, data_size, chunksize=chunksize)
        adjusted_cov_mat = pd.Panel({k:v[0] for k, v in zip(vra_cov_mat.items, results)},
            major_axis=self.vra_cov_mat.major_axis, minor_axis=self.vra_cov_mat.minor_axis)
        self.simed_eigen_bias = pd.DataFrame({i: v[1] for i, v in zip(vra_cov_mat.items, results)}).T
        self.eigen_adjusted_cov_mat = adjusted_cov_mat.reindex(items=self.vra_cov_mat.items)

        # # 储存
        # self.eigen_adjusted_cov_mat.to_hdf('bb_factor_eigencovmat_all_sf3', '123')


    # 进行单次的eigen factor adjustment的函数
    @staticmethod
    def eigen_factor_adjustment(cov_mat, no_of_sims, scaling_factor, simed_sample_size):
        # 如果矩阵是一个singular matrix, 即这一期没有某个因子的数据, 则要把这个因子从协方差矩阵中剔除
        # 检查方法是, 如果该协方差矩阵的对角线是0, 则去除这个对角线上的标签的行与列
        missing_loc = np.where(np.diag(cov_mat) == 0)
        labels = cov_mat.index[missing_loc]
        if labels.size > 0:
            cov_mat = cov_mat.drop(labels, axis=0).drop(labels, axis=1)
        # 首先对估计的协方差矩阵进行eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # 储存每次模拟结果的矩阵
        simed_d = np.empty((no_of_sims, eigenvalues.size))
        simed_d_hat = np.empty((no_of_sims, eigenvalues.size))

        # 对于每一次模拟
        for i in range(no_of_sims):
            # 生成由k个特征因子*T个时间长度(即sample size)组成的收益矩阵
            # 第k行, 即第k个特征因子的收益的方差应该是对应的第k个特征值
            eigenfactor_returns = np.random.normal(0, np.sqrt(eigenvalues), (simed_sample_size, eigenvalues.size))
            # 用eigenvectors与模拟得到的eigenfactor_return点乘, 得到模拟的因子收益
            simulated_factor_return = pd.DataFrame(np.dot(eigenfactor_returns, eigenvectors.T),
                                                   index=np.arange(simed_sample_size), columns=cov_mat.index)

            # # 将月度的收益变为日度的, 否则数量级不对
            # simulated_factor_return /= np.sqrt(21)

            # 将生成的因子收益放入协方差矩阵的估计函数中, 估计模拟的协方差矩阵
            # 注意, 这里的cursor, 要写成sample_size-1, 即要从数据的第一项用到最后一项
            # 另外再算nw lag项的时候, 由于只模拟了sample_size项, 因此lag项会有数据损失, 暂时不管这个问题
            # simulated_cov_mat = factor_base.initial_cov_estimator(simed_sample_size-1, simulated_factor_return,
            #     sample_size=simed_sample_size, var_half_life=1e16, corr_half_life=1e16, var_nw_lag=0,
            #     corr_nw_lag=0, forecast_steps=1)[0]
            simulated_cov_mat = factor_base.initial_cov_estimator(simed_sample_size-1, simulated_factor_return,
                sample_size=simed_sample_size, var_half_life=1e16, corr_half_life=1e16, var_nw_lag=5,
                corr_nw_lag=2, forecast_steps=1)[0]

            # 计算模拟出的协方差矩阵的特征向量
            simed_eigenvalues, simed_eigenvectors = np.linalg.eig(simulated_cov_mat)
            # 用特征向量和真实的协方差矩阵计算出一个对角线表示特征值的矩阵, 即D-hat
            # 只取它的对角线上的元素
            d_hat = np.diagonal(np.dot(np.dot(simed_eigenvectors.T, cov_mat), simed_eigenvectors))

            # 将得到的结果储存起来
            simed_d[i, :] = simed_eigenvalues
            simed_d_hat[i, :] = d_hat

        # 模拟结束后, 计算simulated volatility bias
        # 这里的计算参考USE4, 因为USE4比eigenfactor adjustment更新一些
        bias = np.sqrt(np.mean(simed_d_hat/simed_d, axis=0))
        # eigenfactor adjustment报告中的bias, 这个bias的调整幅度更大
        # bias = np.mean(np.sqrt(simed_d_hat/simed_d), axis=0)
        # 通过scaling factor来进行修正
        scaled_bias = scaling_factor * (bias - 1) + 1

        # 通过bias来对原始估计的协方差矩阵的特征值进行修正
        adjusted_eigenvalues = (scaled_bias ** 2) * eigenvalues
        # 通过修正后的eigenvalues来计算修正后的协方差矩阵
        adjusted_cov_mat = np.dot(np.dot(eigenvectors, np.diag(adjusted_eigenvalues)), eigenvectors.T)

        # 如果有被删除的因子, 则要在这里将它们的值都补成0
        if labels.size>0:
            # 由于curr_loc中的位置是升序排列的, 因此, 当处在前面位置的元素被插入后,
            # 后面的元素的位置就能反应其当时在原始矩阵中的位置. 如果不是升序排列, 就会出问题
            # 注意missing_loc是一个tuple, 它的第一个元素才是包含loc信息的ndarray, 因此循环要循环它的第一个元素
            for curr_loc in missing_loc[0]:
                adjusted_cov_mat = np.insert(adjusted_cov_mat, curr_loc, 0, axis=0)
                adjusted_cov_mat = np.insert(adjusted_cov_mat, curr_loc, 0, axis=1)
                # bias和scaled bias怎么插入还有待考虑, 因为这里的位置并不代表因子的位置
                bias = np.insert(bias, curr_loc, 1)
                scaled_bias = np.insert(scaled_bias, curr_loc, 1)

        return [adjusted_cov_mat, bias, scaled_bias]

    # 对原始的specific vol进行估计的函数
    def get_initial_spec_vol(self, *, sample_size=360, spec_var_half_life=84, nw_lag=5, forecast_steps=21):
        # # 首先将从来就没有数据的那些股票丢掉
        # valid_data = self.specific_return.dropna(axis=1, how='all')
        valid_data = self.specific_return
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        # 初始化储存时间序列预测的, 结构化预测的spec risk的dataframe
        ts_spec_vol = valid_data * np.nan
        ts_daily_spec_vol = valid_data * np.nan
        str_spec_vol = valid_data * np.nan
        str_daily_spec_vol = valid_data * np.nan
        Zu = valid_data * np.nan
        ts_weight = valid_data * np.nan

        # ts_spec_vol = pd.read_hdf('bb_factor_specvol_all', '123')

        # 开始循环计算
        for cursor, time in enumerate(valid_data.index):
            # if cursor < sample_size - 1:
            if time < pd.Timestamp('2013-06-13'):
                continue

            # 在每一次计算开始的时候, 只取那些当期可投资, 且有specific return的股票
            curr_inv_stks = np.logical_and(self.base_data.if_tradable['if_inv', time, :],
                                           self.specific_return.ix[time, :].notnull())
            valid_data = self.specific_return.ix[:, curr_inv_stks]

            # 取当期的数据, 由于acovf函数不能接受全是nan的股票, 因此将这里全是nan的股票再次剔除
            curr_data = valid_data.iloc[cursor + 1 - sample_size:cursor + 1, :].dropna(axis=1, how='all')

            # 首先计算时间序列的spec risk
            ts_outcome = factor_base.ts_spec_vol_estimator(curr_data, sample_size=sample_size,
                spec_var_half_life=spec_var_half_life, nw_lag=nw_lag, forecast_steps=forecast_steps)
            ts_spec_vol.ix[time, :] = ts_outcome[0]
            ts_daily_spec_vol.ix[time, :] = ts_outcome[1]

            # 计算sample std和robust std的那个比例, 即eue3中的Zu
            Zu.ix[time, :] = factor_base.get_ratio_Zu(curr_data)
            # 计算时间序列std的权重
            ts_weight.ix[time, :] = factor_base.get_ts_weight(curr_data, Zu.ix[time, :])

            # 通过时间序列的spec risk, 用回归计算结构化的spec risk
            reg_base = self.base_data.factor_expo.ix[:, time, valid_data.columns]
            # reg_base = reg_base.drop(['lncap', 'beta', 'nls', 'bp', 'ey', 'growth', 'leverage'], axis=1)
            str_outcome = factor_base.str_spec_vol_estimator(ts_spec_vol.ix[time, :],
                ts_daily_spec_vol.ix[time, :], ts_weight.ix[time, :], reg_base,
                n_style=3, n_indus=32, reg_weight=
                np.sqrt(self.base_data.stock_price.ix['FreeMarketValue', time, valid_data.columns]))
            str_spec_vol.ix[time, :] = str_outcome[0]
            str_daily_spec_vol.ix[time, :] = str_outcome[1]

            pass
        pass
        # ts_spec_vol.to_hdf('bb_factor_specvol_all', '123')
        # ts_daily_spec_vol.to_hdf('bb_factor_dailyspecvol_all', '123')

    # 估计原始spec vol的函数的并行版本
    def get_initial_spec_vol_parallel(self, *, sample_size=360, spec_var_half_life=84, nw_lag=5, forecast_steps=21):
        specific_return = self.specific_return.dropna(axis=0, how='all')
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        if_tradable = self.base_data.if_tradable
        global factor_expo
        factor_expo = self.base_data.factor_expo
        mv = self.base_data.stock_price.ix['FreeMarketValue']

        def one_time_estimator_func(cursor):
            time = specific_return.index[cursor]
            # 在每一次计算开始的时候, 只取那些当期可投资, 且有specific return的股票
            curr_inv_stks = np.logical_and(if_tradable['if_inv', time, :],
                                           specific_return.ix[time, :].notnull())
            valid_data = specific_return.ix[:, curr_inv_stks]


            # 取当期的数据, 由于acovf函数不能接受全是nan的股票, 因此将这里全是nan的股票再次剔除
            curr_data = valid_data.iloc[cursor + 1 - sample_size:cursor + 1, :].dropna(axis=1, how='all')
            if curr_data.empty:
                return [pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series()]
            # 首先计算时间序列的spec risk
            ts_outcome = factor_base.ts_spec_vol_estimator(curr_data, sample_size=sample_size,
                spec_var_half_life=spec_var_half_life, nw_lag=nw_lag, forecast_steps=forecast_steps)

            # 计算sample std和robust std的那个比例, 即eue3中的Zu
            Zu = factor_base.get_ratio_Zu(curr_data)
            # 计算时间序列std的权重
            ts_weight = factor_base.get_ts_weight(curr_data, Zu)

            # 通过时间序列的spec risk, 用回归计算结构化的spec risk
            reg_base = factor_expo.ix[:, time, valid_data.columns]
            # # reg_base = reg_base.drop(['lncap', 'beta', 'nls', 'bp', 'ey', 'growth', 'leverage'], axis=1)
            str_outcome = factor_base.str_spec_vol_estimator(ts_outcome[0], ts_outcome[1], ts_weight,
                reg_base, n_style=10, n_indus=28, reg_weight=np.sqrt(mv.ix[time, valid_data.columns]))

            return [ts_outcome[0], ts_outcome[1], str_outcome[0], str_outcome[1], ts_weight]

        ncpus = 20
        p = mp.ProcessPool(ncpus)
        data_size = np.arange(specific_return.shape[0])
        chunksize = int(len(data_size)/ncpus)
        results = p.map(one_time_estimator_func, data_size, chunksize=chunksize)
        ts_spec_vol = pd.DataFrame({i: v[0] for i, v in zip(specific_return.index, results)}).T
        ts_daily_spec_vol = pd.DataFrame({i: v[1] for i, v in zip(specific_return.index, results)}).T
        str_spec_vol = pd.DataFrame({i: v[2] for i, v in zip(specific_return.index, results)}).T
        str_daily_spec_vol = pd.DataFrame({i: v[3] for i, v in zip(specific_return.index, results)}).T
        ts_weight = pd.DataFrame({i: v[4] for i, v in zip(specific_return.index, results)}).T

        ts_spec_vol = ts_spec_vol.reindex(columns=self.base_data.stock_price.minor_axis)
        ts_daily_spec_vol = ts_daily_spec_vol.reindex(columns=self.base_data.stock_price.minor_axis)
        str_spec_vol = str_spec_vol.reindex(columns=self.base_data.stock_price.minor_axis)
        str_daily_spec_vol = str_daily_spec_vol.reindex(columns=self.base_data.stock_price.minor_axis)
        ts_weight = ts_weight.reindex(columns=self.base_data.stock_price.minor_axis)


        # ts_spec_vol.to_hdf('bb_factor_tsspecvol_all', '123')
        # ts_daily_spec_vol.to_hdf('bb_factor_tsdailyspecvol_all', '123')
        # str_spec_vol.to_hdf('bb_factor_strspecvol_all', '123')
        # str_daily_spec_vol.to_hdf('bb_factor_strdailyspecvol_all', '123')
        # ts_weight.to_hdf('ts_weight_all', '123')

        weighted_spec_vol = ts_spec_vol*ts_weight + str_spec_vol*(1-ts_weight)
        weighted_daily_spec_vol = ts_daily_spec_vol*ts_weight + str_daily_spec_vol*(1-ts_weight)
        # weighted_spec_vol.to_hdf('bb_factor_weightedspecvol_hs300', '123')
        # weighted_daily_spec_vol.to_hdf('bb_factor_weighteddailyspecvol_hs300', '123')

        self.initial_spec_vol = weighted_spec_vol
        self.initial_daily_spec_vol = weighted_daily_spec_vol

        pass


    # 计算时间序列预测法的spec vol
    @staticmethod
    def ts_spec_vol_estimator(curr_data, *, sample_size=360, spec_var_half_life=84,
                               nw_lag=5, forecast_steps=21):
        # 首先创建指数权重序列
        spec_var_weight = strategy_data.construct_expo_weights(spec_var_half_life, sample_size)

        # 使用将数据乘以对应的权重, 再计算的方式来进行估计. 在将权重乘在数据上时, 要做一些调整.
        weighted_curr_data = strategy_data.multiply_weights(curr_data, spec_var_weight, multiply_power=0.5)
        # 要根据nw lag进行循环计算, lag=0表示计算自己的方差
        # 不能使用sm.tsa.acovf函数的原因在于, 由于股票的数据有缺失, 使用acovf时, dropna后,
        # 间隔一个的数据, 时间间隔不一定是1, 因此必须要通过循环在dataframe上进行shift才能保证间隔是1
        curr_acov = pd.DataFrame(np.nan, index=np.arange(nw_lag + 1), columns=weighted_curr_data.columns)
        for curr_lag in range(nw_lag + 1):
            lagged_data = weighted_curr_data.shift(curr_lag)
            temp_corr = weighted_curr_data.corrwith(lagged_data)
            curr_acov.iloc[curr_lag, :] = lagged_data.std() * weighted_curr_data.std() * temp_corr
        # 将curr_acov中的nan填成0, 如果一直股票的数据<=nw lag, 那么这只股票的curr acov就有一些项是nan
        # 因为lag过后的数据不足2个, 无法计算, 这些数据的ts_weight会是0, 也就是说肯定会被str_spec_vol代替
        # 如果不把nan填成0, 那么ts_spec_vol就会是nan, 从而导致weigted_spec_vol是nan
        curr_acov = curr_acov.fillna(0.0)
        # 生成对各个lag项对应的权重, 注意, 这里使用的是eue3中的方法, 一来这和协方差估计的nw权重方法一样
        # 二来, 目前并没有理解use4中的nw half_life是是什么意思, 如何用来加权, 等之后理解了可以再改
        nw_lag_weight = 2 * (nw_lag + 1 - np.arange(0, nw_lag + 1)) / (nw_lag + 1)
        nw_lag_weight[0] = 1
        # 这里需要进行与协方差矩阵估计量中一样的方差调整, 因为这里同样是先乘以了权重, 再计算acovf
        # 虽然提高了速度, 但是造成了偏差, 要进行调整
        magnitude_adj = np.sqrt((spec_var_weight[-1] / spec_var_weight[-2]) ** (np.arange(nw_lag + 1)))
        curr_ts_spec_var = curr_acov.mul(nw_lag_weight, axis=0).mul(magnitude_adj, axis=0).sum().mul(forecast_steps)
        curr_ts_spec_vol = np.sqrt(curr_ts_spec_var)

        # 注意, 返回的daily specific risk forecast是不需要进行nw调整的
        return [curr_ts_spec_vol, np.sqrt(curr_acov.iloc[0, :])]

    # 计算Zu的函数
    @staticmethod
    def get_ratio_Zu(curr_data, *, robust_arg=1.35, truncate_multiplier=10):
        # 算每支股票的robust std
        robust_std = curr_data.apply(lambda x: (x.quantile(0.75)-x.quantile(0.25))/robust_arg, axis=0)
        # 将每支股票的spec return截取在trunc_multiplier*robust_std之间
        upper = robust_std * truncate_multiplier
        lower = robust_std * -truncate_multiplier
        truncated_data = np.where(curr_data<=upper, curr_data, upper)
        truncated_data = np.where(curr_data>=lower, truncated_data, lower)
        truncated_data = np.where(curr_data.isnull(), np.nan, truncated_data)
        truncated_data = pd.DataFrame(truncated_data, index=curr_data.index, columns=curr_data.columns)
        # 计算等权的std
        equal_std = truncated_data.std()
        # 计算Zu
        Zu = (equal_std/robust_std - 1).abs()

        return Zu

    # 计算时间序列std所用的权重的函数
    # 样本内的有效数据个数达到upper threshold, 且Zu小于等于Zu threshold, 则使用ts std, 即权重为1
    # 若不足lower threshold, 则使用str std, 即权重为0
    @staticmethod
    def get_ts_weight(curr_data, Zu, *, upper_thre=180, lower_thre=60, Zu_thre=1):
        valid_n = curr_data.notnull().sum()
        # 根据eue3的公式计算权重, 注意Zu的这一部分的np.max被去掉了, eue3中的这个地方应该是有错
        y = np.minimum(1, np.maximum(0, (valid_n-lower_thre)/(upper_thre-lower_thre))) * \
                np.minimum(1, np.exp(Zu_thre-Zu))

        return y

    # 计算structured specific vol的函数
    # 注意, daily的spec vol也需要计算str_vol
    @staticmethod
    def str_spec_vol_estimator(ts_spec_vol, ts_daily_spec_vol, ts_weight, base_expo, *, n_style,
                               n_indus=28, reg_weight=1):
        if isinstance(reg_weight, int):
            reg_weight = pd.Series(1, index=ts_spec_vol.index)
        # 取ts weight=1的股票, 只有这些股票才进入回归
        valid_stocks = ts_weight == 1
        valid_ts_spec_vol = ts_spec_vol.ix[valid_stocks]
        valid_ts_daily_spec_vol = ts_daily_spec_vol.ix[valid_stocks]
        valid_base_expo = base_expo.ix[valid_stocks, :]
        valid_reg_weight = reg_weight.ix[valid_stocks]
        if valid_base_expo.iloc[:, :10].isnull().all().all():
            return [pd.Series(), pd.Series()]

        # 先算str_spec_vol
        # 回归的被解释变量, 为ts vol取对数
        ln_ts_vol = np.log(valid_ts_spec_vol)
        # 进行回归
        params = strategy_data.constrained_gls_barra_base(ln_ts_vol, valid_base_expo, weights=valid_reg_weight,
                    indus_ret_weights=valid_reg_weight**2, n_style=n_style, n_indus=n_indus)[0]
        # 计算模型拟合部分, 这一部分的时候要将所有股票都算进去
        fitted_part = base_expo.dot(params)
        str_sepc_vol = np.exp(fitted_part)

        # 计算E0, E0是reg weighted average of ratio between ts vol and str vol
        # 暂时只计算valid stocks的均值
        ratio = valid_ts_spec_vol.div(str_sepc_vol.reindex(index=valid_ts_spec_vol.index)).dropna()
        E0 = np.average(ratio, weights=valid_reg_weight.reindex(index=ratio.index))
        str_sepc_vol = str_sepc_vol.mul(E0)

        # 还需要计算daily_str_spec_vol
        ln_ts_daily_vol = np.log(valid_ts_daily_spec_vol)
        # 进行回归
        params_daily = strategy_data.constrained_gls_barra_base(ln_ts_daily_vol, valid_base_expo,
                        weights=valid_reg_weight, indus_ret_weights=valid_reg_weight**2,
                        n_style=n_style, n_indus=n_indus)[0]
        fitted_part_daily = base_expo.dot(params_daily)
        str_daily_spec_vol = np.exp(fitted_part_daily)
        # 同样需要计算E0 daily
        ratio_daily = valid_ts_daily_spec_vol.div(str_daily_spec_vol.reindex(index=
                        valid_ts_daily_spec_vol.index)).dropna()
        E0_daily = np.average(ratio_daily, weights=valid_reg_weight.reindex(index=ratio_daily.index))
        str_daily_spec_vol = str_daily_spec_vol.mul(E0_daily)

        return [str_sepc_vol, str_daily_spec_vol]

    # 对spec vol做volatility regime adjustment
    def get_vra_spec_vol(self, *, sample_size=252, vra_half_life=42):
        # spec_vol = pd.read_hdf('bb_factor_weightedspecvol_hs300', '123')
        # daily_spec_vol = pd.read_hdf('bb_factor_weighteddailyspecvol_hs300', '123')

        # 标准化的spec return, 用每天的实现残余收益除以之前预测的当天的残余波动率
        standardized_spec_ret = self.specific_return.div(self.initial_daily_spec_vol.shift(1))
        # 由于个股的标准化收益容易出现极值, 因此, 在截面上对标准化收益进行去极值处理
        standardized_spec_ret = strategy_data.winsorization(standardized_spec_ret)

        # 计算specific cross-sectional bias statistic
        specific_cs_bias = pd.Series(np.nan, index=standardized_spec_ret.index)
        for time, curr_std_ret in standardized_spec_ret.dropna(axis=0, how='all').iterrows():
            valid_data = curr_std_ret.dropna().pow(2)
            cap_weight = self.base_data.stock_price.ix['FreeMarketValue', time, valid_data.index]
            specific_cs_bias.ix[time] = np.sqrt(np.average(valid_data, weights=cap_weight))
            # specific_cs_bias.ix[time] = np.sqrt(valid_data.mean())

        self.vra_multiplier_spec = pd.Series(np.nan, index=self.specific_return.index)
        # 循环计算vra multiplier
        for cursor, time in enumerate(self.specific_return.index):
            if time < pd.Timestamp('2011-05-03'):
                continue

            # 使用与算因子vra乘数时候一样的函数来进行计算, 两者的算法是一致的
            vra_multiplier = factor_base.vra_multiplier_estimator(cursor, specific_cs_bias,
                                sample_size=sample_size, vra_half_life=vra_half_life)
            self.vra_multiplier_spec.ix[time] = vra_multiplier

        # 循环结束后, 计算vra_spec_vol, 只需将原本的vol乘以vra multiplier即可
        self.vra_spec_vol = self.initial_spec_vol.mul(self.vra_multiplier_spec, axis=0)

        # self.vra_spec_vol.to_hdf('bb_factor_vraspecvol_hs300', '123')

        pass

    # 对spec vol做bayesian shrinkage
    def get_bayesian_shrinkage_spec_vol(self, *, shrinkage_parameter=0.1):
        # initial_spec_vol = pd.read_hdf('bb_factor_vraspecvol_all', '123')

        bs_spec_vol = self.vra_spec_vol * np.nan

        # 按照日期进行循环
        for time, curr_spec_vol in self.vra_spec_vol.iterrows():
            if curr_spec_vol.isnull().all():
                continue
            valid_spec_vol = curr_spec_vol.dropna()
            curr_mv = self.base_data.stock_price.ix['FreeMarketValue', time, valid_spec_vol.index]
            # 按照市值将股票分位10分位
            mv_decile = pd.qcut(curr_mv, 10, labels=False)
            # 按照市值加权, 计算每个市值分位数内的预测风险的均值
            grouped_forecast_mean = valid_spec_vol.groupby(mv_decile).transform(
                lambda x: np.average(x, weights=curr_mv.ix[x.index]))
            # 计算每个市值分位数内, 预测风险的等权标准差
            grouped_forecast_std = valid_spec_vol.groupby(mv_decile).transform(lambda x: x.std())

            # 按照use4的公式, 计算shrinkage intensity
            shrinkage_intensity = (shrinkage_parameter*(valid_spec_vol-grouped_forecast_mean).abs())/ \
                (grouped_forecast_std + shrinkage_parameter*(valid_spec_vol-grouped_forecast_mean).abs())
            # 计算bayesian shrinkage后的spec vol
            bs_spec_vol.ix[time, :] = shrinkage_intensity * grouped_forecast_mean + \
                (1 - shrinkage_intensity) * valid_spec_vol
            pass
        pass

        self.bs_spec_vol = bs_spec_vol

    # 构建风险预测的函数
    def construct_risk_forecast_parallel(self, *, freq='m', covmat_sample_size=504, var_half_life=84,
            corr_half_life=504, var_nw_lag=5, corr_nw_lag=2, vra_sample_size=252, vra_half_life=42,
            eigen_adj_sims=1000, scaling_factor=1.4, specvol_sample_size=360, specvol_half_life=84,
            specvol_nw_lag=5, shrinkage_parameter=0.1):
        # 将freq转到forecast_step
        freq_map = {'m': 21, 'w': 5}
        forecast_steps = freq_map[freq]

        # 估计原始的特征收益波动率
        self.get_initial_spec_vol_parallel(sample_size=specvol_sample_size, spec_var_half_life=specvol_half_life,
                                           nw_lag=specvol_nw_lag, forecast_steps=forecast_steps)

        # 对特征波动率进行vra
        self.get_vra_spec_vol(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 暂时不做bayesian shrinkage
        self.get_bayesian_shrinkage_spec_vol(shrinkage_parameter=shrinkage_parameter)

        # 将spec vol改成spec var
        self.spec_var = self.bs_spec_vol ** 2

        # 估计原始协方差矩阵
        self.get_initial_cov_mat_parallel(sample_size=covmat_sample_size, var_half_life=var_half_life,
                                          corr_half_life=corr_half_life, var_nw_lag=var_nw_lag,
                                          corr_nw_lag=corr_nw_lag, forecast_steps=forecast_steps)

        # 进行vra
        self.get_vra_cov_mat(sample_size=vra_sample_size, vra_half_life=vra_half_life)

        # 进行特征值调整
        self.get_eigen_adjusted_cov_mat_parallel(n_of_sims=eigen_adj_sims, scaling_factor=scaling_factor,
                                                 simed_sample_size=covmat_sample_size)

        # 储存数据
        self.eigen_adjusted_cov_mat.to_hdf('bb_riskmodel_covmat_'+self.base_data.stock_pool, '123')
        self.spec_var.to_hdf('bb_riskmodel_specvar_'+self.base_data.stock_pool, '123')
        # self.eigen_adjusted_cov_mat.to_hdf('barra_riskmodel_covmat_all_facret', '123')
        # self.spec_var.to_hdf('barra_riskmodel_specvar_all_facret', '123')

    # 根据barra的bias statistics来测试风险预测能力的函数
    def risk_forecast_performance(self, *, no_of_sims=10000, freq='m', test_type='random'):
        # 测试barra的估计量
        forecasted_cov_mat = pd.read_hdf('bb_factor_eigencovmat_hs300_sf3', '123')
        # # 测试最简单的估计量
        # forecasted_cov_mat = self.base_factor_return.rolling(504).cov() * 21


        # 首先将实现的因子收益求和
        realized_factor_return = self.base_factor_return.resample(freq, label='left').apply(
            lambda x: (1 + x).prod() - 1)
        # 取距离实现时间段最近的那个预测值
        risk_forecast = forecasted_cov_mat.resample(freq, label='left').last().shift(1, axis=0)
        # panel shift过后, 将他们的时间标签对齐
        realized_factor_return = realized_factor_return.iloc[1:, :]

        # 建立储存bias stats的df
        bias_stats = pd.DataFrame(np.nan, index=forecasted_cov_mat.resample('m', label='left').
                                  first().items, columns=np.arange(no_of_sims))

        # 对随机组合做测试
        if test_type == 'random':
            for i in range(no_of_sims):
                # 生成随机数, 来表示随机因子暴露组合, 然后通过考察对随机组合的风险预测情况来考察风险预测能力
                factor_expo = np.random.uniform(-1, 1, forecasted_cov_mat.shape[1])
                # country factor的暴露设置为1
                factor_expo[-1] = 0
                factor_expo = pd.Series(factor_expo, index=forecasted_cov_mat.major_axis)

                bias_stats.iloc[:, i] = factor_base.get_bias_stats(realized_factor_return, risk_forecast,
                                                                  factor_expo)

        # 对优化组合做测试
        if test_type == 'optimized':
            for i in range(no_of_sims):
                # 在因子层面生成随机数, 代表每个因子的alpha, 从标准正态分布中抽取
                factor_alphas = np.random.normal(size=forecasted_cov_mat.shape[1])
                # 根据这个因子收益建立最优化的因子组合
                def opt_func(cov_mat, alpha):
                    if cov_mat.isnull().any().any():
                        return pd.Series(np.nan, index=cov_mat.index)
                    else:
                        inv_mat = np.linalg.pinv(cov_mat)
                        holding = np.dot(inv_mat, alpha) / np.dot(np.dot(alpha, inv_mat), alpha)
                        return pd.Series(holding, index=cov_mat.index)
                factor_holding = risk_forecast.apply(opt_func, axis=(1, 2), alpha=factor_alphas).T

                bias_stats.iloc[:, i] = factor_base.get_bias_stats(realized_factor_return, risk_forecast,
                                                                  factor_holding)

        # 对eigen factor portfolio做测试
        if test_type == 'eigen':
            # 将预测的协方差矩阵做特征值分解, 得到特征矩阵
            # 每个时间点上的每个特征矩阵, 代表着每个因子在每个特征组合中的权重, 即每个特征组合上的因子暴露
            eigenvectors = risk_forecast.dropna().apply(lambda x: np.linalg.eig(x)[1], axis=(1,2))
            # 注意, panel.set_axis函数的功能还处于开发阶段, 参数不稳定, 暂时默认是inplace的
            eigenvectors.set_axis(labels=risk_forecast.major_axis, axis=1)
            # 依照特征向量的顺序做循环
            for cursor, k in enumerate(eigenvectors.minor_axis):
                bias_stats.iloc[:, cursor] = factor_base.get_bias_stats(realized_factor_return,
                                                risk_forecast, eigenvectors.iloc[:, :, cursor].T)

        pass

    # 测试风险预测能力的函数的并行版本
    # freq表示风险预测的频率, 即是预测未来一个月的风险, 或是预测未来一周的风险
    # bias_type等于1, 为barra版本的bias stats,
    # 等于2, 为实现波动率用根号scaling然后除以预测波动率, 即最简单的那种算法
    def risk_forecast_performance_parallel(self, *, no_of_sims=10000, freq='m', test_type='random',
                                           bias_type=1):
        # # 测试估计量
        # forecasted_cov_mat = pd.read_hdf('bb_riskmodel_vracovmat_hs300', '123')
        # 测试barra的预测数据
        self.base_factor_return = pd.read_hdf('barra_real_fac_ret', '123')
        forecasted_cov_mat = pd.read_hdf('barra_fore_cov_mat', '123')
        self.base_factor_return = self.base_factor_return.ix['2011-05-03':'2017-02-28', :]
        forecasted_cov_mat = forecasted_cov_mat.ix['2011-05-03':'2017-02-28', :, :]
        # # 测试最简单的估计量
        # forecasted_cov_mat = self.base_factor_return.rolling(504).cov() * 21

        # 如果是使用barra版本的bias stats, 则将实现的因子收益在周期内求和
        if bias_type == 1:
            realized_factor_return = self.base_factor_return.resample(freq, label='left').apply(
                lambda x: (1 + x).prod() - 1)
            # panel shift过后, 将实现收益和风险预测的时间标签对齐
            realized_factor_return = realized_factor_return.iloc[1:, :]
        # 如果是使用最原始版本的bias stats, 因为是要计算周期内的因子波动率, 因此并不进行求和
        else:
            realized_factor_return = self.base_factor_return

        # 取距离实现时间段最近的那个预测值
        risk_forecast = forecasted_cov_mat.resample(freq, label='left').last().shift(1, axis=0)


        # 对随机组合做测试
        if test_type == 'random':
            def test_random_func(i):
                # 生成随机数, 来表示随机因子暴露组合, 然后通过考察对随机组合的风险预测情况来考察风险预测能力
                # factor_expo = np.random.uniform(-1, 1, forecasted_cov_mat.shape[1])
                factor_expo = np.zeros(39)
                # country factor的暴露设置为1
                # factor_expo[:-1] = np.random.uniform(-1, 1, 38)
                # factor_expo[-1] = 0
                # country = 1, 行业加和等于1, 相当于预测整个组合
                # factor_expo[0:10] = np.random.uniform(-1, 1, 10)
                # factor_expo[-1] = 1
                # factor_expo[10:-1] = np.random.uniform(0, 1, 28)
                # factor_expo[10:-1] = factor_expo[10:-1]/np.sum(factor_expo[10:-1])
                # country = 0, 行业加和等于0, 相当于预测组合的超额部分
                # factor_expo[0:-1] = np.random.uniform(-1, 1, 38)
                # factor_expo[-1] = 0
                # indus = factor_expo[10:-1]
                # indus_plus = (indus>0)
                # indus_minus = (indus<0)
                # indus_plus_w = indus[indus_plus]/np.sum(indus[indus_plus])
                # indus_minus_w = - (indus[indus_minus]/np.sum(indus[indus_minus]))
                # indus[indus_plus] = indus_plus_w
                # indus[indus_minus] = indus_minus_w
                # factor_expo[10:-1] = indus

                # # 测试barra的数据的时候, country factor的loc=13
                # factor_expo[13] = 0
                factor_expo = pd.Series(factor_expo, index=forecasted_cov_mat.major_axis)

                return factor_base.get_bias_stats(realized_factor_return, risk_forecast, factor_expo,
                                                 bias_type=bias_type, freq=freq)
            ncpus = 20
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(no_of_sims)
            chunksize = int(len(data_size)/ncpus)
            results = p.map(test_random_func, data_size, chunksize=chunksize)
            bias_stats = pd.concat([i for i in results], axis=1)

        # 对优化组合做测试
        if test_type == 'optimized':
            def test_opt_func(i):
                # 在因子层面生成随机数, 代表每个因子的alpha, 从标准正态分布中抽取
                factor_alphas = np.random.normal(size=forecasted_cov_mat.shape[1])
                # 把country factor的alpha设为0
                factor_alphas[-1] = 0

                # 根据这个因子收益建立最优化的因子组合
                def opt_func(cov_mat, alpha):
                    if cov_mat.isnull().any().any():
                        return pd.Series(np.nan, index=cov_mat.index)
                    else:
                        inv_mat = np.linalg.pinv(cov_mat)
                        holding = np.dot(inv_mat, alpha) / np.dot(np.dot(alpha, inv_mat), alpha)
                        return pd.Series(holding, index=cov_mat.index)

                factor_holding = risk_forecast.apply(opt_func, axis=(1, 2), alpha=factor_alphas).T

                return factor_base.get_bias_stats(realized_factor_return, risk_forecast, factor_holding,
                                                 bias_type=bias_type, freq=freq)
            ncpus = 20
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(no_of_sims)
            chunksize = int(len(data_size)/ncpus)
            results = p.map(test_opt_func, data_size, chunksize=chunksize)
            bias_stats = pd.concat([i for i in results], axis=1)

        # 对eigen factor portfolio做测试, 这个测试很快, 不用做并行计算了
        if test_type == 'eigen':
            # 建立储存bias stats的df
            bias_stats = pd.DataFrame(np.nan, index=forecasted_cov_mat.resample(freq, label='left').
                                      first().items, columns=np.arange(risk_forecast.shape[1]))
            # 将预测的协方差矩阵做特征值分解, 得到特征矩阵
            # 每个时间点上的每个特征矩阵, 代表着每个因子在每个特征组合中的权重, 即每个特征组合上的因子暴露
            eigenvectors = risk_forecast.dropna().apply(lambda x: np.linalg.eig(x)[1], axis=(1, 2))
            # 注意, panel.set_axis函数的功能还处于开发阶段, 参数不稳定, 暂时默认是inplace的
            eigenvectors.set_axis(labels=risk_forecast.major_axis, axis=1)
            # 依照特征向量的顺序做循环
            for cursor, k in enumerate(eigenvectors.minor_axis):
                bias_stats.iloc[:, cursor] = factor_base.get_bias_stats(realized_factor_return,
                    risk_forecast, eigenvectors.iloc[:, :, cursor].T, bias_type=bias_type, freq=freq)

        # eigenfactor portfolio的测试需要的结果和其他的情况计算方法不一样
        if test_type == 'eigen':
            bias_mean = bias_stats.mean()
            # 储存
            # bias_mean.to_csv('bias_stats_eigenfactor_before.csv', na_rep='NaN')
        else:
            # 计算关于bias stats的统计量
            bias_simed_mean = bias_stats.mean(1)
            bias_mean = bias_simed_mean.mean()
            bias_std = bias_simed_mean.std()
            bias_outlier_ratio = (np.logical_or(bias_stats>1.34, bias_stats<0.66).\
                sum(1)/(no_of_sims)).mean()
            bias_quantile = (bias_simed_mean.quantile(0.05), bias_simed_mean.quantile(0.95))
            # 进行打印
            output_str = 'bias mean is: {0}\nbias std is: {1}\nbias outlier ratio is: {2}\n' \
                         'bias 5% and 95% quantile is: {3}, {4}\n'.format(bias_mean, bias_std,
                bias_outlier_ratio, bias_quantile[0], bias_quantile[1])
            print(output_str)

            # bias_simed_mean.to_hdf('bias_after_vra', '123')
            # f4 = plt.figure()
            # ax4 = f4.add_subplot(1, 1, 1)
            # plt.plot(bias_simed_mean.dropna(), 'b-')
            # ax4.set_xlabel('Time')
            # ax4.set_ylabel('Bias Statistics')
            # ax4.set_title('Time Series of Bias Stats (after VRA)')
            # plt.xticks(rotation=30)
            # plt.grid()
            # plt.savefig(str(os.path.abspath('.')) + '/bias_after_vra.png', dpi=1200)
            pass

    # 预测是风险预测能力的函数, 这次是测试specific risk的预测能力
    def risk_forecast_performance_parallel_spec(self, *, no_of_sims=10000, freq='m', test_type='stock',
                                                bias_type=1, cap_weighted_bias=False):
        # 测试的spec var估计量
        forecasted_spec_vol = pd.read_hdf('bb_factor_vraspecvol_all', '123')
        forecasted_spec_var = forecasted_spec_vol ** 2
        # forecasted_spec_var = pd.read_hdf('barra_fore_spec_var', '123')
        # self.specific_return = pd.read_hdf('barra_real_spec_ret', '123')

        # 先把从来就没有specific return的股票去掉, 这样可以节约一些时间
        valid_specific_return = self.specific_return.dropna(axis=1, how='all')
        # 如果是使用barra版本的bias stats, 则将实现的因子收益在周期内求和
        if bias_type == 1:
            # realized_spec_return = valid_specific_return.resample(freq, label='left').apply(
            #     lambda x: (1 + x).prod() - 1)
            realized_spec_return = valid_specific_return.resample(freq, label='left').sum()
            # panel shift过后, 将实现收益和风险预测的时间标签对齐
            realized_spec_return = realized_spec_return.iloc[1:, :]
        # 如果是使用最原始版本的bias stats, 因为是要计算周期内的因子波动率, 因此并不进行求和
        else:
            realized_spec_return = valid_specific_return

        # 取距离实现时间段最近的那个预测值
        risk_forecast = forecasted_spec_var.resample(freq, label='left').last().shift(1, axis=0)

        # 取月末市值
        monthly_mv = self.base_data.stock_price.ix['FreeMarketValue'].fillna(method='ffill'). \
            resample(freq, label='left').last()

        # 对每个个股做测试
        if test_type == 'stock' or test_type == 'bayesian':
            # bias_stats = pd.DataFrame(np.nan, index=risk_forecast.index, columns=risk_forecast.columns)
            # for cursor, stock in enumerate(realized_spec_return.columns):
            #     stock_holding = pd.Series(0, index=risk_forecast.columns)
            #     stock_holding.ix[stock] = 1
            #     bias_stats.ix[:, stock] = factor_base.get_bias_stats(realized_spec_return, risk_forecast,
            #         stock_holding, bias_type=bias_type, freq=freq)

            def single_stock_func(cursor):
                stock_holding = pd.Series(0, index=risk_forecast.columns)
                stock_holding.iloc[cursor] = 1
                curr_bias = factor_base.get_bias_stats(realized_spec_return, risk_forecast,
                                                       stock_holding, bias_type=bias_type, freq=freq)
                return curr_bias

            ncpus = 20
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(risk_forecast.shape[1])
            chunksize = int(len(data_size) / ncpus)
            results = p.map(single_stock_func, data_size, chunksize=chunksize)
            bias_stats = pd.DataFrame({i: v for i, v in zip(risk_forecast.columns, results)})

        if test_type == 'bayesian':
            grouped_bias_mean = pd.DataFrame(np.nan, index=bias_stats.index, columns=np.arange(10))
            for cursor, time in enumerate(bias_stats.index):
                curr_bias_stats = bias_stats.ix[time, :]
                if curr_bias_stats.isnull().all():
                    continue
                # 在每个时间点上, 按照预测的spec risk将股票分成10份
                curr_risk_forecast = risk_forecast.ix[time, :]
                grouped_stocks = pd.qcut(curr_risk_forecast, 10, labels=False)
                # 组内求bias stats的均值
                grouped_bias_mean.ix[time, :] = curr_bias_stats.groupby(grouped_stocks).mean()
                # grouped_bias_mean.ix[time, :] = curr_bias_stats.groupby(grouped_stocks).apply(
                #     lambda x: np.average(x.dropna(), weights=monthly_mv.ix[time, x.dropna().index]))

        bias_stats = bias_stats.ix['20120430':'20170228', :]
        # 计算关于bias stats的统计量
        if cap_weighted_bias:
            bias_simed_mean = pd.Series(np.nan, index=bias_stats.index)
            for cursor, time in enumerate(bias_stats.index):
                curr_bias_stats = bias_stats.ix[time, :]
                if curr_bias_stats.isnull().all():
                    continue
                bias_simed_mean.ix[time] = np.average(curr_bias_stats.dropna(), weights=
                monthly_mv.ix[time, curr_bias_stats.dropna().index])
        else:
            bias_simed_mean = bias_stats.mean(1)
        bias_mean = bias_simed_mean.mean()
        bias_std = bias_simed_mean.std()
        bias_quantile = (bias_simed_mean.quantile(0.05), bias_simed_mean.quantile(0.95))
        output_str = 'bias mean is: {0}\nbias std is: {1}\nbias 5% and 95% quantile is: {2}, {3}\n'. \
            format(bias_mean, bias_std, bias_quantile[0], bias_quantile[1])
        print(output_str)

        pass

    # 测试整个组合的风险预测能力的函数
    def risk_forecast_performance_total_parallel(self, *, no_of_sims=10000, freq='m', test_type='random',
                                                 bias_type=1):
        self.base_data.generate_if_tradable()
        self.base_data.handle_stock_pool()
        # # 测试自己的
        # forecasted_cov_mat = pd.read_hdf('bb_riskmodel_covmat_hs300', '123')
        # forecasted_spec_var = pd.read_hdf('bb_riskmodel_specvar_hs300', '123')
        # 测试barra的
        self.base_factor_return = pd.read_hdf('barra_real_fac_ret', '123')
        forecasted_cov_mat = pd.read_hdf('barra_fore_cov_mat', '123')
        forecasted_spec_var = pd.read_hdf('barra_fore_spec_var_new', '123') * 0
        self.specific_return = pd.read_hdf('barra_real_spec_ret_new', '123')
        self.base_data.factor_expo = pd.read_hdf('barra_factor_expo_new', '123')
        # # 测试barra的数据的时候, 如果一直股票没有country factor的暴露, 则认为这只股票是不可投资的
        # barra_if_inv = pd.DataFrame(True, index=self.base_data.factor_expo.major_axis,
        #                             columns=self.base_data.factor_expo.minor_axis)
        # barra_if_inv = barra_if_inv.where(self.base_data.factor_expo.ix['CNE5S_COUNTRY'].notnull(), False)
        # self.base_data.if_tradable = pd.Panel({'if_inv': barra_if_inv})

        # 如果是使用barra版本的bias stats, 则将实现的因子收益在周期内求和
        if bias_type == 1:
            realized_factor_return = self.base_factor_return.resample(freq, label='left').apply(
                lambda x: (1 + x).prod() - 1)
            # panel shift过后, 将实现收益和风险预测的时间标签对齐
            realized_factor_return = realized_factor_return.iloc[1:, :]
            realized_spec_return = self.specific_return.resample(freq, label='left').sum()
            # panel shift过后, 将实现收益和风险预测的时间标签对齐
            realized_spec_return = realized_spec_return.iloc[1:, :]
        else:
            realized_factor_return = self.base_factor_return
            realized_spec_return = self.specific_return
        realized_return = (realized_factor_return, realized_spec_return)

        # 取距离实现时间段最近的那个预测值
        risk_forecast_factor = forecasted_cov_mat.resample(freq, label='left').last().shift(1, axis=0)
        risk_forecast_spec = forecasted_spec_var.resample(freq, label='left').last().shift(1, axis=0).iloc[1:, :]
        risk_forecast = (risk_forecast_factor, risk_forecast_spec)
        # 指示股票是否可投资的if_inv矩阵, 也取距离实现时间段最近的那个可交易信息,
        # 即在做出下一个周期的投资决策时, 能得到的最新的可投资数据, 即这个周期的最后一天
        if_inv = self.base_data.if_tradable.ix['if_inv'].resample(freq, label='left').last().shift(1).iloc[1:, :]
        # 因子暴露数据也是使用上个周期最后一天的数据, 注意, 在计算bias stats=2的时候, 由于周期内暴露始终保持
        # 上周期最后一天的暴露的数据, 因此这里的计算是粗糙版的计算
        factor_expo = self.base_data.factor_expo.resample(freq, label='left', axis=1).last().shift(1, axis=1)

        # 对随机组合做测试
        if test_type == 'random':
            def test_random_func(i):
                # 生成随机数, 代表每支股票的权重
                stock_weight = pd.DataFrame(np.random.uniform(0, 1, risk_forecast_spec.shape),
                                            index=risk_forecast_spec.index, columns=risk_forecast_spec.columns)
                # 只取可投资的股票, 归一权重
                stock_weight = stock_weight.where(if_inv, np.nan)
                stock_weight = stock_weight.div(stock_weight.sum(1), axis=0)

                return factor_base.get_bias_stats(realized_return, risk_forecast, stock_weight,
                                                  bias_type=bias_type, freq=freq, factor_expo=factor_expo)

            ncpus = 20
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(no_of_sims)
            chunksize = int(len(data_size) / ncpus)
            results = p.map(test_random_func, data_size, chunksize=chunksize)
            bias_stats = pd.concat([i for i in results], axis=1)
            pass

        # 对优化组合做测试
        if test_type == 'optimized':
            # 股票层面的协方差矩阵的逆矩阵, 由因子暴露和因子的协方差矩阵得到
            global inv_mat
            inv_mat = pd.Panel(np.nan, items=risk_forecast_spec.index,
                               major_axis=risk_forecast_spec.columns, minor_axis=risk_forecast_spec.columns)
            # for time, curr_cov_mat in risk_forecast_factor.iteritems():
            #     # if curr_cov_mat.isnull().all().all():
            #     if time < pd.Timestamp('2011-04-30'):
            #         continue
            #     curr_factor_expo = factor_expo.ix[:, time, :]
            #     print(time)
            #     inv_mat.ix[time] = np.linalg.pinv(np.dot(curr_factor_expo.fillna(0.0).dot(curr_cov_mat.fillna(0.0)),
            #                                       curr_factor_expo.fillna(0.0).T))
            # inv_mat.to_hdf('inv_mat_hs300', '123')
            inv_mat = pd.read_hdf('inv_mat_hs300', '123')

            def test_opt_func(i):
                # 个股层面生成alpha
                stock_alphas = np.random.normal(size=risk_forecast_spec.shape[1])
                # 循环解优化组合
                stock_weight = pd.DataFrame(np.nan, index=risk_forecast_spec.index,
                                            columns=risk_forecast_spec.columns)
                for time, curr_inv_mat in inv_mat.iteritems():
                    if curr_inv_mat.isnull().any().any():
                        continue
                    holding = np.dot(curr_inv_mat, stock_alphas) / np.dot(np.dot(stock_alphas, curr_inv_mat),
                                                                          stock_alphas)
                    stock_weight.ix[time] = holding
                print(i)
                return factor_base.get_bias_stats(realized_return, risk_forecast, stock_weight,
                                                  bias_type=bias_type, freq=freq, factor_expo=factor_expo)

            ncpus = 25
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(no_of_sims)
            chunksize = int(len(data_size) / ncpus)
            results = p.map(test_opt_func, data_size, chunksize=chunksize)
            bias_stats = pd.concat([i for i in results], axis=1)

        # 对自己给的组合做测试
        if test_type == 'custom':
            stock_weight = pd.read_hdf('opt_holding_tar_hs300', '123')
            # stock_weight = pd.read_hdf('barra_opt_holding_tar_hs300', '123')
            # stock_weight = pd.read_hdf('opt_holding_barraopt_hs300', '123')
            # 由于输入持仓是调仓期之间是0持仓, 因此要做适当的格式转化
            stock_weight = stock_weight.mask((stock_weight == 0).all(1), np.nan). \
                fillna(method='ffill').fillna(0.0)
            stock_weight = stock_weight.reindex(index=risk_forecast_spec.index, method='ffill'). \
                shift(-1).fillna(0.0)

            bias_stats = factor_base.get_bias_stats(realized_return, risk_forecast, stock_weight,
                                                    bias_type=bias_type, freq=freq, factor_expo=factor_expo)

        # bias_stats.to_hdf('bias_stats_barra_opt', '123')
        bias_simed_mean = bias_stats.ix['20120430':'20170228', :].mean(1)
        bias_mean = bias_simed_mean.mean()
        bias_std = bias_simed_mean.std()
        bias_quantile = (bias_simed_mean.quantile(0.05), bias_simed_mean.quantile(0.95))
        output_str = 'bias mean is: {0}\nbias std is: {1}\nbias 5% and 95% quantile is: {2}, {3}\n'. \
            format(bias_mean, bias_std, bias_quantile[0], bias_quantile[1])
        print(output_str)

    # 根据所给的数据, 计算当前组合的bias statistics, 用来检测风险预测的能力
    # freq表示风险预测的频率, 即是预测未来一个月的风险, 或是预测未来一周的风险
    # bias_type == 1为计算barra版本的bias stats, 计算实现的标准收益率的波动率来进行计算
    # == 2 为计算最普通版本的bias stats, 计算周期内实现收益的波动率, 直接用根号进行scaling
    @staticmethod
    def get_bias_stats(realized_return, risk_forecast, port_weight, *, bias_type=1, freq='m',
                       factor_expo=None):
        # 根据组合的因子暴露, 乘以因子收益, 算出组合的实现因子收益, 并且计算预测的组合风险
        if isinstance(port_weight, pd.Series):
            # 判断放入的是协方差矩阵还是specific risk, 针对不同的风险有不同的算法
            # 如果是panel, 则是计算协方差矩阵的bias stats
            if isinstance(risk_forecast, pd.Panel):
                realized_port_return = realized_return.mul(port_weight, axis=1).sum(1)
                forecast_port_vol = risk_forecast.apply(lambda x: np.sqrt(np.dot(port_weight.dot(x), port_weight)),
                                                        axis=(1, 2))
            # 如果是dataframe, 则是计算spec risk的bias stats
            else:
                realized_port_return = realized_return.mul(port_weight, axis=1).sum(1)
                forecast_port_vol = np.sqrt(risk_forecast.mul(port_weight.pow(2), axis=1).sum(1))

        elif isinstance(port_weight, pd.DataFrame):
            # 先计算预测的组合风险, 否则如果bias_type=2, port_weight将会被reindex
            # 如果是panel, 则是计算协方差矩阵的bias stats
            if isinstance(risk_forecast, pd.Panel):
                forecast_port_vol = pd.Series(np.nan, index=risk_forecast.items)
                for time, cov_mat in risk_forecast.iteritems():
                    if cov_mat.isnull().any().any():
                        forecast_port_vol.ix[time] = np.nan
                    else:
                        forecast_port_vol.ix[time] = np.sqrt(np.dot(port_weight.ix[time, :].dot(cov_mat),
                                                                    port_weight.ix[time, :]))
            # 如果是dataframe, 则是计算spec risk的bias stats
            elif isinstance(risk_forecast, pd.DataFrame):
                forecast_port_vol = np.sqrt(risk_forecast.mul(port_weight.pow(2)).sum(1))
            # 如果是一个tuple, 则是计算整体组合的估计情况, 协方差矩阵是第一个, spec risk是第二个
            # 注意, 在计算整体组合的风险预测的bias stats时, port_weight一定是每天都有一个值, 因此一定是DataFrame
            else:
                cov_mat = risk_forecast[0]
                spec_var = risk_forecast[1]
                # 首先根据股票权重, 股票的因子暴露, 计算组合的因子权重
                # 这里采取粗糙的算法, 即并不调用strategy data中的get_port_expo函数
                port_factor_expo = np.einsum('ijk,jk->ji', factor_expo.fillna(0.0), port_weight.fillna(0.0))
                port_factor_expo = pd.DataFrame(port_factor_expo, index=cov_mat.items, columns=cov_mat.major_axis)
                # 同样的需要先计算组合的预测风险
                forecast_port_vol = pd.Series(np.nan, index=cov_mat.items)
                for time, curr_cov_mat in cov_mat.iteritems():
                    if curr_cov_mat.isnull().any().any():
                        forecast_port_vol.ix[time] = np.nan
                    else:
                        forecast_port_vol.ix[time] = np.dot(port_factor_expo.ix[time, :].dot(curr_cov_mat),
                                                                    port_factor_expo.ix[time, :])
                forecast_port_vol += spec_var.mul(port_weight.pow(2)).sum(1)
                forecast_port_vol = np.sqrt(forecast_port_vol)
                # 计算组合的realized return, 此时realized_return参数中包含因子收益与残余收益

            if isinstance(realized_return, tuple):
                factor_ret = realized_return[0]
                spec_ret = realized_return[1]
                # 如果是算普通版本的bias stats, 需要将port_weight变成每天的版本
                if bias_type == 2:
                    port_weight = port_weight.reindex(index=factor_ret.index, method='ffill')
                    # 组合的暴露, 也直接采取ffill的方式, 这样的方式使得计算再一次变粗糙
                    port_factor_expo = port_factor_expo.reindex(index=factor_ret.index, method='ffill')
                realized_port_return = factor_ret.mul(port_factor_expo).sum(1)
                realized_port_return += spec_ret.mul(port_weight).sum(1)
            else:
                if bias_type == 2:
                    port_weight = port_weight.reindex(index=realized_return.index, method='ffill')
                realized_port_return = realized_return.mul(port_weight).sum(1)
        else:
            realized_port_return = realized_return * np.nan
            forecast_port_vol = realized_return * np.nan

        # 计算barra版本的bias stats
        if bias_type == 1:
            # 组合的standardized return
            standardized_return = realized_port_return/forecast_port_vol
            # 计算rolling window bias statistics
            rolling_bias_stats = standardized_return.rolling(12).std()
        # 计算最原始版本的bias stats
        elif bias_type == 2:
            # 计算组合在周期内的实现波动率, 且直接用根号来对波动率进行scaling
            # realized_port_vol = realized_port_return.resample(freq, label='left').apply(
            #     lambda x: x.std() * np.sqrt(x.dropna().size))
            realized_port_vol = realized_port_return.resample(freq, label='left').apply(
                lambda x: x.std() * np.sqrt(5))
            # 用组合的实现波动率除以预测波动率, 得到一个比例
            # 实现的组合波动率的时间标签需要和预测的波动率标签对齐, bias_type=1时的对齐工作在外面的函数中已经完成了
            vol_ratio = realized_port_vol.iloc[1:]/forecast_port_vol
            # 计算滚动12个月的波动率比例的均值, 这个统计量显示过去12个月的预测能力
            rolling_bias_stats = vol_ratio.rolling(12).apply(lambda x: np.nanmean(x))

        return vol_ratio

    # 这个函数为处理barra方面的原始数据, 把它做成自己的数据格式, 处理barra方面的数据是因为想测试barra的预测效果
    # 对比自己的预测效果, 可以有一个参考
    def handle_barra_data(self):
        # 取实现的因子收益, 其全部在一个文件夹里, 因此不需要循环
        fac_ret = pd.read_csv('CNE5S_100_DlyFacRet.20170309', sep='|', header=0, parse_dates=[2])
        factor_return = fac_ret.pivot_table(index='DataDate', columns='Factor', values='DlyReturn')
        realized_factor_ret = factor_return.reindex(index=self.base_factor_return.index)
        # 初始化要取的数据
        forecasted_cov_mat = pd.Panel(np.nan, items=self.base_factor_return.index,
                    major_axis=realized_factor_ret.columns, minor_axis=realized_factor_ret.columns)
        forecasted_spec_var = pd.DataFrame()
        realized_spec_ret = pd.DataFrame()
        factor_expo = pd.Panel()

        # 根据交易日进行循环
        for cursor, time in enumerate(self.base_factor_return.index):
            # 将时间转化成barra文件后缀的形式
            datestr = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2)
            # 读取预测协方差数据
            # 首先判断该天是否在文件中
            if not os.path.isfile('barra_data/CNE5S_100_Covariance.'+datestr):
                continue
            covmat = pd.read_csv('barra_data/CNE5S_100_Covariance.'+datestr, sep='|', header=2)[:-1]
            factor_cov1 = covmat.pivot_table(index='!Factor1', columns='Factor2', values='VarCovar')
            factor_cov2 = covmat.pivot_table(index='Factor2', columns='!Factor1', values='VarCovar')
            factor_cov = factor_cov1.where(factor_cov1.notnull(), factor_cov2).div(10000)
            # 读取股票的预测残余风险
            asset_data = pd.read_csv('barra_data/CNE5S_100_Asset_Data.'+datestr, sep='|', header=2)[:-1]
            spec_risk = asset_data.pivot_table(index='!Barrid', values='SpecRisk%')
            spec_var = (spec_risk/100)**2
            # 读取股票的实现残余收益
            asset_return = pd.read_csv('barra_data/CNE5_100_Asset_DlySpecRet.'+datestr, sep='|', header=2)[:-1]
            spec_return = asset_return.pivot_table(index='!Barrid', values='SpecificReturn')
            spec_return /= 100
            # 读取股票的因子暴露
            asset_expo = pd.read_csv('barra_data/CNE5S_100_Asset_Exposure.'+datestr, sep='|', header=2)[:-1]
            curr_factor_expo = asset_expo.pivot_table(index='!Barrid', columns='Factor', values='Exposure')

            forecasted_cov_mat.ix[time] = factor_cov
            spec_var.name = time
            spec_return.name = time
            forecasted_spec_var = forecasted_spec_var.join(spec_var, how='outer')
            realized_spec_ret = realized_spec_ret.join(spec_return, how='outer')
            curr_factor_expo = pd.Panel({time: curr_factor_expo})
            factor_expo = factor_expo.join(curr_factor_expo, how='outer')

            print(time)
            pass

        forecasted_spec_var = forecasted_spec_var.T.reindex(index=self.base_factor_return.index)
        realized_spec_ret = realized_spec_ret.T.reindex(index=self.base_factor_return.index)
        factor_expo = factor_expo.reindex(items=self.base_factor_return.index)
        # 将年化的单位转为日度(收益), 月度(风险)
        forecasted_cov_mat /= 12
        forecasted_spec_var /= 12

        # 储存结果
        realized_factor_ret.to_hdf('barra_real_fac_ret', '123')
        forecasted_cov_mat.to_hdf('barra_fore_cov_mat', '123')
        realized_spec_ret.to_hdf('barra_real_spec_ret', '123')
        forecasted_spec_var.to_hdf('barra_fore_spec_var', '123')
        factor_expo.transpose(2, 0, 1).to_hdf('barra_factor_expo', '123')
        pass









    # # 原始协方差矩阵的估计量函数
    # @staticmethod
    # def initial_cov_estimator_naive(cursor, complete_factor_return_data, *, sample_size=504,
    #                           var_half_life=84, corr_half_life=504, var_nw_lag=5, corr_nw_lag=2,
    #                           forecast_steps=21):
    #     # 首先创建指数权重序列
    #     var_weight = strategy_data.construct_expo_weights(var_half_life, sample_size)
    #     corr_weight = strategy_data.construct_expo_weights(corr_half_life, sample_size)
    #     # 取当期的数据
    #     curr_data = complete_factor_return_data.iloc[cursor + 1 - sample_size:cursor + 1, :]
    #
    #     # 先计算var
    #     curr_var = None
    #     # 根据var的nw lag进行循环, 从lag=0开始, lag=0则表示计算自己的方差
    #     for curr_var_lag in range(var_nw_lag + 1):
    #         # 根据当前的lag数, 取对应的lag data
    #         lagged_data = complete_factor_return_data.shift(curr_var_lag). \
    #                           iloc[cursor + 1 - sample_size:cursor + 1, :]
    #         # 将数据乘以对应的权重
    #         weighted_curr_data = curr_data.mul(np.sqrt(var_weight), axis=0)
    #         weighted_lagged_data = lagged_data.mul(np.sqrt(var_weight), axis=0)
    #         # 将其放入估计方差的函数中, 计算方差
    #         var = factor_base.var_func(weighted_curr_data, weighted_lagged_data)
    #         # 根据对应的权重, 将估计出的var加到原始的var上去
    #         # 当lag是0的时候, 计算的就是自身的方差
    #         if curr_var_lag == 0:
    #             curr_var = var * 1
    #         else:
    #             curr_var = curr_var + 2 * (var_nw_lag + 1 - curr_var_lag) / (var_nw_lag + 1) * var
    #
    #     # 再计算corr
    #     curr_corr = None
    #     # 根据corr的nw lag进行循环, 从lag=0开始, lag=0则表示计算自己的相关系数矩阵
    #     for curr_corr_lag in range(corr_nw_lag + 1):
    #         # 根据当前的lag数, 取对应的lag data
    #         lagged_data = complete_factor_return_data.shift(curr_corr_lag). \
    #                           iloc[cursor + 1 - sample_size:cursor + 1, :]
    #         # 将数据乘以对应的权重
    #         weighted_curr_data = curr_data.mul(np.sqrt(corr_weight), axis=0)
    #         weighted_lagged_data = lagged_data.mul(np.sqrt(corr_weight), axis=0)
    #         # 将其放入估计相关系数矩阵的函数中, 计算相关系数矩阵
    #         corr = factor_base.corr_func(weighted_curr_data, weighted_lagged_data)
    #         # 根据对应的权重, 将估计出的corr加到原始的corr上去
    #         # 当lag是0的时候, 计算的就是自身的相关系数矩阵
    #         if curr_corr_lag == 0:
    #             curr_corr = corr * 1
    #         else:
    #             curr_corr = curr_corr + (corr_nw_lag + 1 - curr_corr_lag) / (corr_nw_lag + 1) * \
    #                                     (corr + corr.T)
    #
    #     # 将估计得到的var和corr结合起来, 形成当期估计的协方差矩阵
    #     # 用日数据转移到周或月数据, 需要乘以天数差
    #     estimated_cov_mat = curr_corr.mul(np.sqrt(curr_var), axis=0). \
    #         mul(np.sqrt(curr_var), axis=1).mul(forecast_steps)
    #     if not np.all(np.linalg.eigvals(estimated_cov_mat) > 0):
    #         print(': is not positive definite matrix!\n')
    #     pass
    #
    #     return [estimated_cov_mat, curr_var]
    #
    # # 计算var的函数
    # @staticmethod
    # def var_func(df, lagged_df):
    #     # 因为乘以权重时已经除以了样本个数n, 而在估计corr以及std的时候再次除以了样本个数n-1
    #     # 因此, 要在数据上乘以样本个数n-1的根号, 才能使得数量级正确
    #     valid_n = np.logical_and(df.notnull(), lagged_df.notnull()).sum(0) - 1
    #     df = df.mul(np.sqrt(valid_n), axis=1)
    #     lagged_df = lagged_df.mul(np.sqrt(valid_n), axis=1)
    #     # 没有直接计算对应列的方差的函数, 因此, 先计算对应列的corr
    #     pairwise_corr = df.corrwith(lagged_df)
    #     # 再通过计算std, 最终得到对应列之间的var
    #     df_std = df.std()
    #     lagged_df_std = lagged_df.std()
    #     var = pairwise_corr * df_std * lagged_df_std
    #
    #     return var
    #
    # # 计算corr的函数
    # @staticmethod
    # def corr_func(df, lagged_df):
    #     # 返回的corr矩阵的格式为, 行是df的因子, 列是lagged_df的因子
    #     corr = pd.DataFrame(np.nan, index=df.columns, columns=lagged_df.columns)
    #     # 只能循环计算, 注意, 由于计算的是corr, 不受数量级影响, 因此不用进行样本个数n的调整
    #     for factor, s in df.iteritems():
    #         for l_factor, l_s in lagged_df.iteritems():
    #             corr.ix[factor, l_factor] = s.corr(l_s)
    #
    #     return corr
