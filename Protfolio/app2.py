#coding:utf8
import tushare as ts
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy.optimize as sco

'''
PART TWO：均值-方差投资组合理论
'''

#选取几只感兴趣的股票
#000001中国平安  000063中兴通讯  000002万科A   600570恒生电子
stock = ['000001','000063','000002','600570']

data1 = ts.get_hist_data(stock[0],start='2015-05-01',end='2016-05-01')
data2 = ts.get_hist_data(stock[1],start='2015-05-01',end='2016-05-01')
data3 = ts.get_hist_data(stock[2],start='2015-05-01',end='2016-05-01')
data4 = ts.get_hist_data(stock[3],start='2015-05-01',end='2016-05-01')
#将所有股票信息的收盘价整合成一个dataframe
df=pd.concat([data1['close'],data2['close'],data3['close'],data4['close']],axis=1)
#修改列名
df.columns=stock
#显示股价走势，比较一下几种股票的情况。规范起点为100
(df/df.ix[0]*100).plot(figsize = (8,4))

#每年252个交易日，用每日收益得到年化收益。
#计算投资资产的协方差是构建资产组合过程的核心部分。运用pandas内置方法生产协方差矩阵。
returns = np.log(df / df.shift(1))

print "年化收益"
print returns.mean()*252
print 32*"-"
print "协方差"
print returns.cov()*252
print 32*"-"
returns.hist(bins = 50, figsize = (9,6))


#给不同资产随机分配初始权重
#由于A股不允许建立空头头寸，所有的权重系数均在0-1之间
weights = np.random.random(4)
weights /= np.sum(weights)
print '初始化权重',weights
#计算预期组合年化收益、组合方差和组合标准差
print '初始权重收益',np.sum(returns.mean()*weights)*252
print '初始权重组合方差',np.dot(weights.T, np.dot(returns.cov()*252,weights))
print '初始权重组合标准差',np.sqrt(np.dot(weights.T, np.dot(returns.cov()* 252,weights)))
#用蒙特卡洛模拟产生大量随机组合
#进行到此，我们最想知道的是给定的一个股票池（证券组合）如何找到风险和收益平衡的位置。
#下面通过一次蒙特卡洛模拟，产生大量随机的权重向量，并记录随机组合的预期收益和方差。
port_returns = []
port_variance = []
for p in range(4000):
    weights = np.random.random(4)
    weights /=np.sum(weights)
    port_returns.append(np.sum(returns.mean()*252*weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)



#投资组合优化1——sharpe最大
#建立statistics函数来记录重要的投资组合统计数据（收益，方差和夏普比）
#通过对约束最优问题的求解，得到最优解。其中约束是权重总和为1。
print 32*"-"
print "投资组合优化方法1--sharpe值最大"
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(returns.mean()*weights)*252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights)))
    return np.array([port_returns, port_variance, port_returns/port_variance])

#最优化投资组合的推导是一个约束最优化问题

#最小化夏普指数的负值
def min_sharpe(weights):
    return -statistics(weights)[2]
#约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
#我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
bnds = tuple((0,1) for x in range(4))
#优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
opts = sco.minimize(min_sharpe, 4*[1./4,], method = 'SLSQP', bounds = bnds, constraints = cons)
print opts
print "权重",opts['x'].round(3)
#预期收益率、预期波动率、最优夏普指数
print '最大sharpe指数预期收益' , statistics(opts['x']).round(3)


print 32*"-"
print "投资组合优化方法2--方差最小"
#投资组合优化2——方差最小
#接下来，我们通过方差最小来选出最优投资组合。
#我们定义一个函数对 方差进行最小化
def min_variance(weights):
    return statistics(weights)[1]
optv = sco.minimize(min_variance, 4*[1./4,],method = 'SLSQP', bounds = bnds, constraints = cons)
print optv
print "权重",optv['x'].round(3)
#得到的预期收益率、波动率和夏普指数
print '最小方差预期收益' , statistics(optv['x']).round(3)

#组合的有效前沿
#有效前沿有既定的目标收益率下方差最小的投资组合构成。
#在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
target_returns = np.linspace(0.0,0.5,50)
target_variance = []
for tar in target_returns:
    cons = ({'type':'eq','fun':lambda x:statistics(x)[0]-tar},{'type':'eq','fun':lambda x:np.sum(x)-1})
    res = sco.minimize(min_variance, 4*[1./4,],method = 'SLSQP', bounds = bnds, constraints = cons)
    target_variance.append(res['fun'])

target_variance = np.array(target_variance)






plt.figure(figsize=(10,6))
#圆圈：蒙特卡洛随机产生的组合分布
plt.scatter(port_variance, port_returns, c = port_returns/port_variance,marker = 'o',s=5 )
#叉号：有效前沿（目标收益率下最优的投资组合）
plt.scatter(target_variance,target_returns, c = target_returns/target_variance, marker = 'X',s=20)
#红星：标记最高sharpe组合
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize = 15.0)
#黄星：标记最小方差组合
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize = 15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
plt.show()
