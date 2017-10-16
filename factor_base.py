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
            if self.base_data.stock_pool == 'all':
                self.filename_appendix = ''
            else:
                self.filename_appendix = '_' + self.base_data.stock_pool
        # 有可能会使用到与当前股票池不同的股票池下计算出的因子值
        else:
            # 如果给的文件后缀是all, 则不加任何后缀
            if filename_appendix == 'all':
                self.filename_appendix = ''
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
            if time < pd.Timestamp('2011-03-03'):
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

        self.initial_cov_mat.to_hdf('bb_factor_covmat_hs300_raw', '123')
        self.daily_var_forecast.to_hdf('bb_factor_var_hs300_raw', '123')

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
        curr_acov = curr_acov.iloc[0:var_nw_lag+1, :]
        # 生成对各个lag项的对应权重
        nw_lag_weight_var = 2 * (var_nw_lag + 1 - np.arange(0, var_nw_lag + 1)) / (var_nw_lag + 1)
        nw_lag_weight_var[0] = 1
        curr_var = curr_acov.mul(nw_lag_weight_var, axis=0).sum()

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

        return [estimated_cov_mat, curr_var]

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

        self.vra_cov_mat.to_hdf('bb_factor_vracovmat_hs300', '123')

    # 进行volatility regime adjustment的函数, 会返回vra后的协方差矩阵
    @staticmethod
    def vra_multiplier_estimator(cursor, factor_cs_bias, *, sample_size=252, vra_half_life=42):
        # 首先创建指数权重序列
        vra_weight = strategy_data.construct_expo_weights(vra_half_life, sample_size)
        # 取当期的数据
        curr_data = factor_cs_bias.iloc[cursor + 1 - sample_size:cursor + 1]

        # 将数据乘以对应的权重, 注意, 参考vra multiplier的计算方法, bias先平方, 然后权重按照根号幂乘在数据上
        weighted_data = strategy_data.multiply_weights(curr_data**2, vra_weight, multiply_power=1)
        # 根据乘以权重后的数据, 计算vra multiplier, 注意use4中的公式可能有误, 因为文字上写明了是算均值, 而不是求和
        vra_multiplier = np.sqrt(weighted_data.mean())

        return vra_multiplier

    # 对初步估计出的协方差矩阵, 进行eigenfactor adjustment的函数
    def get_eigen_adjusted_cov_mat(self, *, n_of_sims=1000, scaling_factor=1.4, simed_sample_size=504):
        # 储存特征值调整后的协方差矩阵
        self.eigen_adjusted_cov_mat = self.initial_cov_mat * np.nan

        # 对日期进行循环
        for cursor, time in enumerate(self.initial_cov_mat.dropna().items):
            # if time < pd.Timestamp('2013-07-31'):
            # if cursor <504:
            #     continue
            curr_cov_mat = self.initial_cov_mat.ix[time]
            if (curr_cov_mat==0).all().all():
                continue
            # 放入进行eigen adjustment的函数进行计算
            outcome = factor_base.eigen_factor_adjustment(curr_cov_mat, n_of_sims, scaling_factor,
                                                         simed_sample_size)
            self.eigen_adjusted_cov_mat.ix[time] = outcome[0]

    # 进行eigenfactor adjustment的函数的并行版本
    def get_eigen_adjusted_cov_mat_parallel(self, *, n_of_sims=1000, scaling_factor=1.4, simed_sample_size=504):
        initial_cov_mat = self.initial_cov_mat.dropna()
        # 定义进行单期调整的函数
        def one_time_adjust_func(cursor):
            curr_cov_mat = initial_cov_mat.iloc[cursor]
            # print(initial_cov_mat.items[cursor])
            # 如果cov_mat是nan, 则返回它自己
            if (curr_cov_mat==0).all().all():
                return curr_cov_mat
            # 放入eigen adjustment的函数进行计算
            outcome = factor_base.eigen_factor_adjustment(curr_cov_mat, n_of_sims, scaling_factor,
                                                         simed_sample_size)
            # 取调整后的cov_mat
            adjusted_cov_mat = outcome[0]
            bias = outcome[1]

            return [adjusted_cov_mat, bias]
        ncpus=20
        p = mp.ProcessPool(ncpus)
        data_size = np.arange(initial_cov_mat.shape[0])
        chunksize = int(len(data_size)/ncpus)
        results = p.map(one_time_adjust_func, data_size, chunksize=chunksize)
        adjusted_cov_mat = pd.Panel({k:v[0] for k, v in zip(initial_cov_mat.items, results)},
            major_axis=self.initial_cov_mat.major_axis, minor_axis=self.initial_cov_mat.minor_axis)
        simed_eigen_bias = pd.DataFrame({i: v[1] for i, v in zip(initial_cov_mat.items, results)}).T
        self.eigen_adjusted_cov_mat = adjusted_cov_mat.reindex(items=self.initial_cov_mat.items)

        # 储存
        self.eigen_adjusted_cov_mat.to_hdf('bb_factor_eigencovmat_hs300_sf3', '123')


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
        vol_forecast = forecasted_cov_mat.resample(freq, label='left').last().shift(1, axis=0)
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

                bias_stats.iloc[:, i] = factor_base.get_bias_stats(realized_factor_return, vol_forecast,
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
                factor_holding = vol_forecast.apply(opt_func, axis=(1, 2), alpha=factor_alphas).T

                bias_stats.iloc[:, i] = factor_base.get_bias_stats(realized_factor_return, vol_forecast,
                                                                  factor_holding)

        # 对eigen factor portfolio做测试
        if test_type == 'eigen':
            # 将预测的协方差矩阵做特征值分解, 得到特征矩阵
            # 每个时间点上的每个特征矩阵, 代表着每个因子在每个特征组合中的权重, 即每个特征组合上的因子暴露
            eigenvectors = vol_forecast.dropna().apply(lambda x: np.linalg.eig(x)[1], axis=(1,2))
            # 注意, panel.set_axis函数的功能还处于开发阶段, 参数不稳定, 暂时默认是inplace的
            eigenvectors.set_axis(labels=vol_forecast.major_axis, axis=1)
            # 依照特征向量的顺序做循环
            for cursor, k in enumerate(eigenvectors.minor_axis):
                bias_stats.iloc[:, cursor] = factor_base.get_bias_stats(realized_factor_return,
                                                vol_forecast, eigenvectors.iloc[:, :, cursor].T)

        pass

    # 测试风险预测能力的函数的并行版本
    # freq表示风险预测的频率, 即是预测未来一个月的风险, 或是预测未来一周的风险
    # bias_type等于1, 为barra版本的bias stats,
    # 等于2, 为实现波动率用根号scaling然后除以预测波动率, 即最简单的那种算法
    def risk_forecast_performance_parallel(self, *, no_of_sims=10000, freq='m', test_type='random',
                                           bias_type=1):
        # 测试barra的估计量
        forecasted_cov_mat = pd.read_hdf('bb_factor_vracovmat_hs300', '123')
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
                factor_expo = np.random.uniform(-1, 1, forecasted_cov_mat.shape[1])
                # country factor的暴露设置为1
                factor_expo[-1] = 0
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
            bias_stats = pd.DataFrame(np.nan, index=forecasted_cov_mat.resample('m', label='left').
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
            bias_mean.to_csv('bias_stats_eigenfactor_before.csv', na_rep='NaN')
        else:
            # 计算关于bias stats的统计量
            bias_simed_mean = bias_stats.mean(1)
            bias_mean = bias_simed_mean.mean()
            bias_std = bias_simed_mean.std()
            bias_outlier_ratio = (np.logical_or(bias_stats>1.34, bias_stats<0.66).\
                sum(1)/(no_of_sims)).mean()
            bias_quantile = (bias_simed_mean.quantile(0.05), bias_simed_mean.quantile(0.95))
            # 进行打印
            output_str = 'bias mean is: {0}\n bias std is: {1}\n bias outlier ratio is: {2}\n' \
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

    # 根据所给的数据, 计算当前组合的bias statistics, 用来检测风险预测的能力
    # freq表示风险预测的频率, 即是预测未来一个月的风险, 或是预测未来一周的风险
    # bias_type == 1为计算barra版本的bias stats, 计算实现的标准收益率的波动率来进行计算
    # == 2 为计算最普通版本的bias stats, 计算周期内实现收益的波动率, 直接用根号进行scaling
    @staticmethod
    def get_bias_stats(realized_factor_return, risk_forecast, port_weight, *, bias_type=1, freq='m'):
        # 根据组合的因子暴露, 乘以因子收益, 算出组合的实现因子收益, 并且计算预测的组合风险
        if isinstance(port_weight, pd.Series):
            realized_port_return = realized_factor_return.dot(port_weight)

            forecast_port_vol = risk_forecast.apply(lambda x: np.sqrt(np.dot(port_weight.dot(x), port_weight)),
                                                    axis=(1, 2))
        elif isinstance(port_weight, pd.DataFrame):
            # 先计算预测的组合风险, 否则如果bias_type=2, port_weight将会被reindex
            forecast_port_vol = pd.Series(np.nan, index=risk_forecast.items)
            for time, cov_mat in risk_forecast.iteritems():
                if cov_mat.isnull().any().any():
                    forecast_port_vol.ix[time] = np.nan
                else:
                    forecast_port_vol.ix[time] = np.sqrt(np.dot(port_weight.ix[time, :].dot(cov_mat),
                                                                port_weight.ix[time, :]))
            # 如果是算普通版本的bias stats, 需要将port_weight变成每天的版本
            if bias_type == 2:
                port_weight = port_weight.reindex(index=realized_factor_return.index, method='ffill')
            realized_port_return = realized_factor_return.mul(port_weight).sum(1)
        else:
            realized_port_return = realized_factor_return * np.nan
            forecast_port_vol = realized_factor_return * np.nan

        # 计算barra版本的bias stats
        if bias_type == 1:
            # 组合的standardized return
            standardized_return = realized_port_return/forecast_port_vol
            # 计算rolling window bias statistics
            rolling_bias_stats = standardized_return.rolling(12).std()
        # 计算最原始版本的bias stats
        elif bias_type == 2:
            # 计算组合在周期内的实现波动率, 且直接用根号来对波动率进行scaling
            realized_port_vol = realized_port_return.resample(freq, label='left').apply(
                lambda x: x.std() * np.sqrt(x.dropna().size))
            # 用组合的实现波动率除以预测波动率, 得到一个比例
            # 实现的组合波动率的时间标签需要和预测的波动率标签对齐, bias_type=1时的对齐工作在外面的函数中已经完成了
            vol_ratio = realized_port_vol.iloc[1:]/forecast_port_vol
            # 计算滚动12个月的波动率比例的均值, 这个统计量显示过去12个月的预测能力
            rolling_bias_stats = vol_ratio.rolling(12).mean()

        return rolling_bias_stats

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
    #     return estimated_cov_mat
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
