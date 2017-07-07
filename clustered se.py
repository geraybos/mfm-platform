import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from data import data

# This is the file for testing clustered standard error of panel data regression\

# # 读取股票收益,市值, bp数据作为回归测试项
# cp = data.read_data(['ClosePrice_adj'])
# cp = cp['ClosePrice_adj']
# daily_return = np.log(cp/cp.shift(1))
# xdata = data.read_data(['lncap', 'bp'])
#
# y = daily_return.stack(dropna=False)
# x = xdata.to_frame(filter_observations=False)
#
# valid = pd.concat([y, x], axis=1).notnull().all(1)
#
# y = y[valid]
# x = x[valid]
#
# groups_time = y.index.get_level_values(0).values
# groups_stock = y.index.get_level_values(1).values
#
# model = sm.OLS(y, x)
# results1 = model.fit()
# # results2 = model.fit(cov_type='cluster', cov_kwds={'groups':groups_time})
# # results3 = model.fit(cov_type='cluster', cov_kwds={'groups':groups_stock})
# results4 = results1.get_robustcov_results(cov_type='cluster', groups=groups_time)
# results5 = results1.get_robustcov_results(cov_type='cluster', groups=groups_stock)

# 验证balance panel的fm回归和ols回归系数是否一致
y = pd.DataFrame(np.random.randn(10, 3), index=pd.date_range(start='2017-01-01', periods=10), columns=['a','b','c'])
x = pd.DataFrame(np.random.randn(10, 3), index=pd.date_range(start='2017-01-01', periods=10), columns=['a','b','c'])

# ols
stacked_y = y.stack()
stacked_x = x.stack()
model = sm.OLS(stacked_y, sm.add_constant(stacked_x))
results_ols = model.fit()

# fm
from pandas.stats.fama_macbeth import fama_macbeth
results_fm = fama_macbeth(y=stacked_y, x=stacked_x.to_frame())
results_fm2 = fama_macbeth(y=stacked_y, x=stacked_x.to_frame(), nw_lags_beta=3)

reg_re = pd.DataFrame(np.nan, index=y.index, columns=['coef', 't'])
for time, data in y.iterrows():
    model = sm.OLS(y.ix[time], sm.add_constant(x.ix[time]))
    re = model.fit()
    reg_re.ix[time, 'coef'] = re.params[1]
    reg_re.ix[time, 't'] = re.tvalues[1]


pass