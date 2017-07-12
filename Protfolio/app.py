#coding:utf8

import tushare as ts
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
'''
第一部分 正态性检验
'''
stock = ['000001','000063','000002','600570']

data1 = ts.get_hist_data(stock[0],start='2015-05-01',end='2016-08-15')
data2 = ts.get_hist_data(stock[1],start='2015-05-01',end='2016-08-15')
data3 = ts.get_hist_data(stock[2],start='2015-05-01',end='2016-08-15')
data4 = ts.get_hist_data(stock[3],start='2015-05-01',end='2016-08-15')

#print data1['close']
df=pd.concat([data1['close'],data2['close'],data3['close'],data4['close']],axis=1)
df.columns=stock
(df/df.ix[0]*100).plot(figsize = (8,5))

#print df

log_returns = np.log(df/df.shift(1))
#print log_returns.head()

#输出每只股票的统计数据
#定义print_statistics函数，为了更加易于理解的方式
#输出给定(历史或者模拟)数据集均值、偏斜度或者峰度等统计数字
def print_statistics(array):
    sta = scs.describe(array)
    print '%14s %15s' %('statistic','value')
    print 30*'-'
    print '%14s %15d' %('size', sta[0])
    print '%14s %15.5f' %('min', sta[1][0])
    print '%14s %15.5f' %('max', sta[1][1])
    print '%14s %15.5f' %('mean', sta[2])
    print '%14s %15.5f' %('std', np.sqrt(sta[3]))
    print '%14s %15.5f' %('skew', sta[4])
    print '%14s %15.5f' %('kurtosis', sta[5])

for st in stock:
    print '\nResults for stock %s' %st
    print 30*'-'
    log_data = np.array(log_returns[st].dropna())
    print_statistics(log_data)

#画qq图观察数据
sm.qqplot(log_returns[stock[0]].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()

#进行正态性检验
def normality_test(array):
    '''
    对给定的数据集进行正态性检验
    组合了3中统计学测试
    偏度测试（Skewtest）——足够接近0
    峰度测试（Kurtosistest)——足够接近0
    正态性测试
    '''
    print 'Skew of data set %15.3f' % scs.skew(array)
    print 'Skew test p-value %14.3f' % scs.skewtest(array)[1]
    print 'Kurt of data set %15.3f' % scs.kurtosis(array)
    print 'Kurt test p-value %14.3f' % scs.kurtosistest(array)[1]
    print 'Norm test p-value %14.3f' % scs.normaltest(array)[1]

for st in stock:
    print '\nResults for st %s' %st
    print 32*'-'
    log_data = np.array(log_returns[st].dropna())
    normality_test(log_data)

'''
从上述测试的p值来看，否定了数据集呈正态分布的测试假设。 这说明，股票市场收益率的正态假设不成立。
'''