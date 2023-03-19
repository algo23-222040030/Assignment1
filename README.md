# 目录
- [代码运行](#代码运行)
  - [数据准备](#数据准备)
  - [安装所需依赖包](#安装所需依赖包)
  - [各模块说明](#各模块说明)
- [结果分析](#结果分析)
- [不足与改进](#不足与改进)

# 论文简介
波动率和换手率是常见的市场监测指标，其与市场长期走势明显负相关的指标，而且指标趋势性较好，因此可以用来判断股票市场状况。本文分享的工作借助波动率与换手率构造出的牛熊指标因子，借助牛熊指标因子开发的择时策略普遍优于直接对指数本身的择时策略。
# 改进
为了进一步拓展波动率和换手率的使用，在原研报的基础上进行了扩展，尝试使用OpenFE进行因子的自动生成，结果显示可以进一步提升策略的收益。
# 结果说明
- 为了处理方便，交易没有考虑手续费，由于交易频率较低，手续费影响有限；
- 原始研报中的牛熊因子和使用OpenFE进行自动因子挖掘都是在数据上直接进行的，存在过拟合失效的风险。
- 只展示了双均线策略和上证指数的回测结果，可以通过修改指数名称和自定义布林带策略进行回测。
# 数据准备
- 数据选取：因子构建部分采用指数数据，原文采用指数数据进行交易。  
具体而言，选取如下指数：上证综指（000001.SH）、中证500（000905.SH）、创业板指数（399006.SZ）、中小板指数（399005.SZ）、沪深300指数（000300.SH）
- 数据标签：日度成交额，成交量，换手率，指数收盘价
- 回测区间：各指数上市日至2023/3/10  
# 各模块说明
- getData.py: 通过akshare API得到所需数据；   
- getFactors.py: 构造gain因子，loss因子和vnsp因子；  
- testFactors.py：进行因子检验；
- getOrders.py: 实现择时策略，即当t − 1交易日的对应因子值比t − 2日的因子数值小时，说明卖出意愿变强，则在t交易日看空，反之则在t交易日看多；
- getResults.py: 进行策略回测，得到回测曲线；
- run.py: 在已有数据的基础上依次运行：getFactors.py，getOrders.py，getResults.py。
# 不足
-  
