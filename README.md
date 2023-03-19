# 目录
- [论文简介](#论文简介)
- [改进](#改进)
- [数据准备](#数据准备)
- [结果说明](#结果说明)
- [使用说明](#使用说明)
- [不足](#不足)

# 论文简介
波动率和换手率是常见的市场监测指标，其与市场长期走势明显负相关的指标，而且指标趋势性较好，因此可以用来判断股票市场状况。本文分享的工作借助波动率与换手率构造出的牛熊指标因子，借助牛熊指标因子开发的择时策略普遍优于直接对指数本身的择时策略。
# 改进
为了进一步拓展波动率和换手率的使用，在原研报的基础上进行了扩展，尝试使用OpenFE进行因子的自动生成，结果显示可以进一步提升策略的收益。
# 数据准备
- 数据选取：上证综指（000001.SH）、沪深300指数（000300.SH）
- 数据标签：日度成交额，成交量，换手率，指数收盘价
- 回测区间：2004年/指数上市日至2023.2.13  
# 结果说明
1. 回测时段为2004年至2023年初
2. 为了处理方便，交易没有考虑手续费，由于交易频率较低，手续费影响有限；
3. 原始研报中的牛熊因子和使用OpenFE进行自动因子挖掘都是在数据上直接进行的，存在过拟合失效的风险。
4. 只展示了双均线策略和上证指数的回测结果，可以通过修改指数名称和自定义布林带策略进行回测。
# 使用说明
- 运行main.py
# 不足
-  
