# 目录
- [目录](#目录)
- [背景](#背景)
- [好的思路](#好的思路)
- [细节](#细节)
- [代码运行](#代码运行)
  - [数据准备](#数据准备)
  - [安装所需依赖包](#安装所需依赖包)
  - [各模块说明](#各模块说明)
- [结果分析](#结果分析)
- [不足与改进](#不足与改进)
# 背景
出于对行为金融学的兴趣，选取行为金融学中的前景理论作为理论基础，复现广发证券研报《交易性择时策略研究之十四——基于V型处置效应的择时研究》，利用V型处置效应实现指数的择时策略。
# 好的思路
- 将处置效应的因子分解成gain和loss
- LLT方法降低因子频率:  
>本篇专题报告对因子走势进行了平滑性处理，具体的方法是采用广发证券金融工程在2013年7月26日《低延迟趋势线与交易性择时》报告中采用的LLT方法。

# 细节
- LLT的参数选取
- 因子的检验：择时因子的检验和选股因子的检验存在差异
- 仓位管理：对于看多看空的仓位处理没有提及，对于纯多和多空的差异没有提及；  
- 对仓位管理的思考：  
趋势跟踪（一把子梭哈）：看空，则开空仓，如有多仓平多仓；看多，则开多仓，如有空仓平空仓；  
均值回归（逐步加仓）：看空做多，逐步加仓，看多做空。
# 代码运行
根目录下的文件夹：/data, /factors, /orders, /figures  
运行顺序：getData.py -> run.py
## 数据准备
- 数据选取：因子构建部分采用指数数据，原文采用指数数据进行交易。  
具体而言，选取如下指数：上证综指（000001.SH）、中证500（000905.SH）、创业板指数（399006.SZ）、中小板指数（399005.SZ）、沪深300指数（000300.SH）
- 数据标签：日度成交额，成交量，换手率，指数收盘价
- 回测区间：各指数上市日至2023/3/10  

## 各模块说明
- getData.py: 通过akshare API得到所需数据；   
- getFactors.py: 构造gain因子，loss因子和vnsp因子；  
- testFactors.py：进行因子检验；
- getOrders.py: 实现择时策略，即当t − 1交易日的对应因子值比t − 2日的因子数值小时，说明卖出意愿变强，则在t交易日看空，反之则在t交易日看多；
- getResults.py: 进行策略回测，得到回测曲线；
- run.py: 在已有数据的基础上依次运行：getFactors.py，getOrders.py，getResults.py。
# 结果分析
- 因子与收益率的相关性不强；
- 多空的收益不如纯多；
- llt方法使得因子变为负值, 特别是接近初始状态的时候，中证500和创业板指的回撤很大，两者都是初始状态为熊市，说明策略在空头市场的收益率不是很好，可能需要修改开平仓规则。
# 不足与改进
1. 完善因子的检验；
2. 进行参数敏感性分析；
3. 对处置效应和V型处置效应进行区分；
4. 为了方便交易，可以采用指数ETF的数据进行交易，做空方面：引入股指期货；
5. 扩展到分钟数据。
