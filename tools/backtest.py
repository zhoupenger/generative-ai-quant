import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
from collections import deque
from more_operators import *
from more_operators import add, neg, sub, mul

class BacktestEngine:
    def __init__(self, data, signal_str, initial_cash=100000, face_value=1, slippage=1/1000, c_rate=5/10000, leverage_rate=1, min_margin_ratio=1/100):
        self.df = data
        self.initial_cash = initial_cash
        self.face_value = face_value # 每次至少买多少份
        self.slippage = slippage
        self.c_rate = c_rate # 手续费
        self.leverage_rate = leverage_rate
        self.min_margin_ratio = min_margin_ratio # 最小保证金比例
        self.signal_str = signal_str

        self.past_signals = deque(maxlen=10000)

        self.long_or_short = 0
        self.long_open_pos_price = 0
        self.short_open_pos_price = 0

        self.open_pos_condition = None
        self.close_pos_condition = None
        self.result = None

        #self.setup_data()

    def setup_data(self):
        self.df['long signal'] = np.nan
        self.df['short signal'] = np.nan
        self.df['next_open'] = self.df['open'].shift(-1)
        self.df['next_open'].fillna(value=self.df['close'], inplace=True)
        self.df = self.df.reset_index()

    def generate_signals(self):
        '''
        long_or_short = 0   
        long_open_pos_price = 0
        short_open_pos_price = 0
        '''

        for i in range(len(self.df)):
            #signals = add(neg(self.df.loc[i, 'close']), add(neg(math.cos(self.df.loc[i, 'SMA'])), self.df.loc[i, 'ROC']))
            #exec(self.signal_str)
            local_vars = {'i': i, 'self': self, 'add': add, 'sub': sub, 'neg': neg, 'mul': mul, 'math': math,
                          'protected_div': protected_div, 'delay': delay, 'delta': delta, 'ts_min': ts_min,
                          'ts_max': ts_max, 'ts_argmin': ts_argmin, 'ts_argmax': ts_argmax, 'ts_rank': ts_rank,
                          'ts_sum': ts_sum, 'ts_stddev': ts_stddev, 'ts_corr': ts_corr, 'ts_mean_return': ts_mean_return,
                          'DEMA': DEMA, 'KAMA': KAMA, 'MA': MA, 'MIDPOINT': MIDPOINT, 'MIDPRICE': MIDPRICE
                          }
            
            exec(self.signal_str, {}, local_vars)
            signals = local_vars['signals']

            self.past_signals.append(signals)

            upper_breakout = signals > np.percentile(self.past_signals, 80)
            lower_breakout = signals < np.percentile(self.past_signals, 20)

            if upper_breakout and self.long_or_short == 0:
                self.df.loc[i, 'long signal'] = 1
                self.long_open_pos_price = self.df.loc[i, 'next_open']
                self.long_or_short = 1

            elif (self.df.loc[i, 'close']-self.long_open_pos_price)/self.long_open_pos_price > 0.1 or (self.df.loc[i, 'close']-self.long_open_pos_price)/self.long_open_pos_price < -0.05 and self.long_or_short == 1:
                self.df.loc[i, 'long signal'] = 0
                self.long_or_short = 0
        
            elif lower_breakout and self.long_or_short == 0:
                self.df.loc[i, 'short signal'] = -1
                self.short_open_pos_price = self.df.loc[i, 'next_open']
                self.long_or_short = -1

            elif (self.df.loc[i, 'close']-self.short_open_pos_price)/self.short_open_pos_price < -0.1 or (self.df.loc[i, 'close']-self.short_open_pos_price)/self.short_open_pos_price > 0.05 and self.long_or_short == -1:
                self.df.loc[i,'short signal'] = 0
                self.long_or_short = 0

            #self.check_signals(i, upper_breakout, lower_breakout)

        self.df['signal'] = self.df[['long signal', 'short signal']].sum(axis=1, min_count=1, skipna=True)

    def consolidate_signals(self):
        # 实际的信号处理逻辑
        # 保留有数据的行数
        temp = self.df[self.df['signal'].notnull()][['signal']]
        # 把重复的信号删掉
        temp = temp[temp['signal'] != temp['signal'].shift(1)]
        # 保留有信号的行数
        self.df['signal'] = temp['signal']

        # 最终保留ohlcv与signal就行
        self.df.reset_index(inplace=True)
        self.df = self.df[['date', 'open', 'high', 'low', 'close', 'volume', 'next_open', 'signal']]

        # 把空值补全
        self.df['signal'].fillna(method='ffill', inplace=True)
        self.df['signal'].fillna(value=0, inplace=True)
        # 注意在k线走完后第二天才会进行开仓，实际持仓与signal差一行
        self.df['pos'] = self.df['signal'].shift()
        self.df['pos'].fillna(value=0, inplace=True)
    

    def post_process_signals(self):
        # 实际的信号后处理逻辑
        # 找出开仓平仓的k线
        condition1 = self.df['pos'] != 0 # 不空仓
        condition2 = self.df['pos'] != self.df['pos'].shift(1) # 当前周期与上一周期状态不一致
        self.open_pos_condition = condition1 & condition2 # 开仓条件

        condition1 = self.df['pos'] != 0 # 不空仓
        condition2 = self.df['pos'] != self.df['pos'].shift(-1) # 当前周期与下一周期状态不一致
        self.close_pos_condition = condition1 & condition2 # 平仓条件

        # 用start_time来标记开仓时间，并将其写入到df中
        self.df['start_time'] = np.nan
        self.df.loc[self.open_pos_condition.values, 'start_time'] = self.df['date']
        self.df['start_time'].fillna(method='ffill', inplace=True)
        self.df.loc[self.df['pos'] == 0, 'start_time'] = pd.NaT
        
    def calculate_finance(self):
        # 实际的金融计算逻辑
        # 开始计算资金曲线
        initial_cash = self.initial_cash
        face_value = self.face_value # 每次至少买多少份
        slippage = self.slippage
        c_rate = self.c_rate # 手续费
        leverage_rate = self.leverage_rate
        min_margin_ratio = self.min_margin_ratio # 最小保证金比例

        # 在开仓时
        # 开仓时contract_num为多少，注意开仓始终是以initial_cash为基准
        self.df.loc[self.open_pos_condition.values, 'contract_num'] = initial_cash * leverage_rate / (face_value * self.df['open'])
        self.df['contract_num'] = np.floor(self.df['contract_num'])
        # 考虑滑点的实际开仓价格
        self.df.loc[self.open_pos_condition.values, 'open_pos_price'] = self.df['open'] * (1 + slippage * self.df['pos'])
        # cash为开仓后扣除手续费后的金额
        self.df['cash'] = initial_cash - self.df['open_pos_price'] * face_value * self.df['contract_num'] * c_rate

        # 在持仓时
        # 买入之后cash, contract_num, open_pos_price不变
        for _ in ['contract_num', 'open_pos_price', 'cash']:
            self.df[_].fillna(method='ffill', inplace=True)
        self.df.loc[self.df['pos'] == 0, ['contract_num', 'open_pos_price', 'cash']] = None

        # 在平仓时
        # 平仓价格
        self.df.loc[self.close_pos_condition.values, 'close_pos_price'] = self.df['next_open'] * (1 - slippage * self.df['pos'])
        # 平仓时手续费
        self.df.loc[self.close_pos_condition.values, 'close_pos_fee'] = self.df['close_pos_price'] * face_value * self.df['contract_num'] * c_rate

        # 计算利润
        # profit是开仓至今的持仓盈亏
        self.df['profit'] = face_value * self.df['contract_num'] * (self.df['close'] - self.df['open_pos_price']) * self.df['pos']
        self.df.loc[self.close_pos_condition.values, 'profit'] = face_value * self.df['contract_num'] * (self.df['close_pos_price'] - self.df['open_pos_price']) * self.df['pos']

        # 账户净值
        self.df['net_value'] = self.df['cash'] + self.df['profit']

        # 计算爆仓
        self.df.loc[self.df['pos'] == 1, 'price_min'] = self.df['low']
        self.df.loc[self.df['pos'] == -1, 'price_min'] = self.df['high']
        self.df['profit_min'] = face_value * self.df['contract_num'] * (self.df['price_min'] - self.df['open_pos_price']) * self.df['pos']
        # 账户净值最小值
        self.df['net_value_min'] = self.df['cash'] + self.df['profit_min']
        # 保证金率，账户最低净值/合约价值
        self.df['margin_ratio'] = self.df['net_value_min'] / (face_value * self.df['contract_num'] * self.df['price_min'])  
        # 计算是否爆仓
        self.df.loc[self.df['margin_ratio'] <= (min_margin_ratio + c_rate), 'is_liquidate'] = 1

        # 平仓时扣除手续费
        self.df.loc[self.close_pos_condition.values, 'net_value'] -= self.df['close_pos_fee']

        # 对爆仓进行进一步处理
        self.df['is_liquidate'] = self.df.groupby('start_time')['is_liquidate'].fillna(method='ffill')
        self.df.loc[self.df['is_liquidate'] == 1, 'net_value'] = 0

        # 计算资金曲线
        # 检查复利的影响

        self.df['equity_change'] = self.df['net_value'].pct_change()
        self.df.loc[self.open_pos_condition.values, 'equity_change'] = self.df.loc[self.open_pos_condition.values, 'net_value'] / initial_cash - 1
        self.df['equity_change'].fillna(value=0, inplace=True)
        self.df['equity_curve'] = (1 + self.df['equity_change']).cumprod()

        self.result = self.df[['date', 'profit', 'net_value', 'equity_curve']]


    def run_backtest(self):
        self.setup_data()
        self.generate_signals()
        self.consolidate_signals()
        self.post_process_signals()
        self.calculate_finance()

        return self.result

'''
if __name__ == '__main__':
    engine = BacktestEngine('clean_data_with_indicators.feather')
    engine.run_backtest()
'''