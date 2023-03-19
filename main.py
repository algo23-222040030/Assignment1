# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time 
import tushare
import pandas as pd
import numpy as np
import quantstats as qs
qs.extend_pandas()
import matplotlib.pyplot as plt
from openfe import openfe

class bull_bear_indicator:
    
    def __init__(self, ts_code, token = "tuhsare账户的token"):
        # 使用tushare的数据加载函数须注册账户并配置token，没有账户或积分不足可使用离线数据运行
        token = "tuhsare账户的token"
        tushare.set_token(token)
        self.ts = tushare.pro_api()
        self.ts_code = ts_code

    def fetch_index_data(self, **kwargs):
        start_date = kwargs.get("start",None)
        end_date = kwargs.get("end",None)
        datas = []
        while True:
            index_dailybasic_data = self.ts.index_dailybasic(ts_code = self.ts_code,end_date=end_date).iloc[::-1]
            index_daily_data = self.ts.index_daily(ts_code = self.ts_code,end_date=end_date).iloc[::-1]
            data = index_dailybasic_data.merge(index_daily_data, on=['trade_date','ts_code'])
            current_start_date = data.iloc[0]["trade_date"]
            if current_start_date == start_date:
                break
            else:
                start_date = current_start_date
                end_date = current_start_date
            datas.append(data)
            time.sleep(0.01)
        datas = datas[::-1]
        data = pd.concat(datas)
        data.drop_duplicates(inplace=True)
        data = data.reset_index(drop=True)
    
        data = data.rename(columns={"ts_code" : "code", "trade_date": "date"})
        data = data.set_index("date")
        data.index = pd.to_datetime(data.index)
    
        return data
    
    def factor_test(self, thread_num):
        try:
            data = self.fetch_index_data(self.ts_code)[["close", "turnover_rate_f", "pct_chg"]]
            data.to_csv("data"+"/"+self.ts_code+".csv")
        except:
            data = pd.read_csv("data"+"/"+self.ts_code+".csv")
            data = data.set_index("date")
        data["turnover_rate_f"] *= 0.01
        data["day_return"] = data.close.rolling(2).apply(lambda x:(x[-1]-x[0])/x[0])
        # 计算250天的波动率方差和换手率均值
        data["day_return"] = data.close.rolling(2).apply(lambda x:(x[-1]-x[0])/x[0])
        data["rolling_volatility"] = data.get("close").pct_change().rolling(250,min_periods=100).std()
        data["rolling_turnover_rate"] = data.get("turnover_rate_f").rolling(250,min_periods=100).mean()
        # 计算牛熊指标，并使用双均线策略进行回测
        data["market_status_indicator"] = data["rolling_volatility"]/data["rolling_turnover_rate"]
        data["market_status_indicator_20day_average"] = data.get("market_status_indicator").rolling(20).mean()
        data["market_status_indicator_60day_average"] = data.get("market_status_indicator").rolling(60).mean()
        
        data["market_status_indicator_average_flag"] = (data["market_status_indicator_20day_average"] <=
                                                        data["market_status_indicator_60day_average"]).astype(int)
        # 指标后移一个时间单位，和下一个时间单位对齐（上一时刻计算出的指标用于指示当前时刻的操作）
        data["market_status_indicator_average_flag"].iloc[1:] = data["market_status_indicator_average_flag"].iloc[:-1] 
        data["market_status_indicator_average_return"] = data.market_status_indicator_average_flag * data.day_return
        data["market_status_indicator_average_cumprod_return"] = np.cumprod(
            data["market_status_indicator_average_return"].fillna(0).values + 1) *100
        # 画图
        _ = plt.figure(figsize=(20, 5))
        plt.xlabel('Time') 
        plt.ylabel('Return')
        _ = plt.plot(data["market_status_indicator_average_cumprod_return"])
        _ = plt.legend(['market_status_indicator_average_cumprod_return',]) 
        # 使用rolling_volatility和rolling_turnover_rate作为特征，使用OpenFE挖掘高级特征
        ofe = openfe()
        train_x = data[["rolling_volatility","rolling_turnover_rate"]]
        train_y = data[['close']]
        features = ofe.fit(train_x, train_y, n_jobs = thread_num)  # generate new features
        print(f"最优特征算子{features[0].name} ; 最优特征输入数据：{[children.name for children in features[0].children]}")
        # 按照挖掘结果生成挖掘出来的第一个特征数据
        data["market_status_indicator_openfe"] = data["rolling_volatility"] - data["rolling_turnover_rate"]
        # 使用双均线策略对挖掘出的特征进行回测，并和原始研报中的牛熊指标进行对比
        data["market_status_indicator_openfe_20day_average"] = data.get("market_status_indicator_openfe").rolling(20).mean()
        data["market_status_indicator_openfe_60day_average"] = data.get("market_status_indicator_openfe").rolling(60).mean()
        data["market_status_indicator_openfe_average_flag"] = (data["market_status_indicator_openfe_20day_average"] <= 
                                                               data["market_status_indicator_openfe_60day_average"]).astype(int)
        # 指标后移一个时间单位，和下一个时间单位对齐（上一时刻计算出的指标用于指示当前时刻的操作）
        data["market_status_indicator_openfe_average_flag"].iloc[1:] = data["market_status_indicator_openfe_average_flag"].iloc[:-1] 
        data["market_status_indicator_openfe_average_return"] = data.market_status_indicator_openfe_average_flag * data.day_return
        data["market_status_indicator_openfe_average_cumprod_return"] = np.cumprod(
            data["market_status_indicator_openfe_average_return"].fillna(0).values + 1) *100
        # 画图
        _ = plt.figure(figsize=(20, 5))
        plt.xlabel('Time') 
        plt.ylabel('Return')
        _ = plt.plot(data["market_status_indicator_average_cumprod_return"])
        _ = plt.plot(data["market_status_indicator_openfe_average_cumprod_return"])
        _ = plt.legend(['market_status_indicator_average_cumprod_return', 'market_status_indicator_openfe_average_cumprod_return'])
        
        return data
    
if __name__ == '__main__':
    a = bull_bear_indicator("000300.SH")# 选择000001.SH或000300.SH
    b = a.factor_test(8)# 设置线程数
    b.sharpe()# 计算各因子夏普比率
