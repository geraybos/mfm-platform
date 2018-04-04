import numpy as np
import pandas as pd
import os

# 批量读取performance txt文件, 提取信息进行对比
outter_folder = 'existing_factors_test/'

# 建立储存最后信息的dataframe
perf_info = pd.DataFrame(columns=['RunnerID', 'active_return', 'active_vol', 'info_ratio',
                                  'factor_return_mean', 'factor_return_std', 'ic_mean'])

for inner_folder in os.listdir(outter_folder):
    if inner_folder[0:2] != 'rv':
        continue
    if int(inner_folder[2:]) >= 37 and int(inner_folder[2:]) <= 54:
        continue
    for file in os.listdir(outter_folder + inner_folder + '/zz500'):
        if file == 'performance.txt':
            curr_perf = pd.read_table(outter_folder + inner_folder + '/zz500/' + file, sep=':',
                                      header=None)
            curr_info = pd.Series(index=perf_info.columns)
            # 依次储存数据
            curr_info['RunnerID'] = inner_folder
            curr_info['active_return'] = curr_perf.iloc[11, 1]
            curr_info['active_vol'] = curr_perf.iloc[12, 1]
            curr_info['info_ratio'] = curr_perf.iloc[13, 1]
            # curr_info['average_stock_num'] = curr_perf.iloc[19, 1]
            # curr_info['risk_ratio_mean'] = curr_perf.iloc[26, 1][:8]
            # curr_info['risk_ratio_std'] = curr_perf.iloc[26, 2][:8]
            # curr_info['alpha_mean'] = curr_perf.iloc[27, 1][:8]
            # curr_info['alpha_std'] = curr_perf.iloc[27, 2][:8]
            # curr_info['spec_risk_aversion'] = curr_perf.iloc[28, 2]
            factor_return_info = curr_perf.iloc[28, 1]
            a, b = factor_return_info.split(',')
            curr_info['factor_return_mean'] = a[6:]
            curr_info['factor_return_std'] = a[5:]
            curr_info['ic_mean'] = curr_perf.iloc[32, 1]
            perf_info = perf_info.append(curr_info, ignore_index=True)
            pass

# perf_info['spec_risk_aversion'] = perf_info['spec_risk_aversion'].map(float)
# perf_info = perf_info.sort_values('spec_risk_aversion')

# 储存信息
perf_info.to_csv(outter_folder + 'performance_summary_zz500.csv')
pass