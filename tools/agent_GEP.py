import geppy as gep
from deap import creator, base, tools
from more_operators import *
import math
from collections import deque
import numpy as np
import pandas as pd
import random
import operator 
import warnings
warnings.filterwarnings("ignore")

class GeneticProgram:
    def __init__(self, data, head=10, genes=2,
                 tournsize=3, mut_invert=0.1, mut_is_transpose=0.1, mut_ris_transpose=0.1, mut_gene_transpose=0.1,
                 cx_1p=0.1, cx_2p=0.1, cx_gene=0.1):
        self.data = data # Dataframe格式的数据集
        self.toolbox = gep.Toolbox()
        self.pset = None
        self.head = head # 基因头部
        self.genes = genes # 一个染色体由几个基因组成

        self.tournsize = tournsize
        self.mut_invert = mut_invert
        self.mut_is_transpose = mut_is_transpose
        self.mut_ris_transpose = mut_ris_transpose
        self.mut_gene_transpose = mut_gene_transpose
        self.cx_1p = cx_1p
        self.cx_2p = cx_2p
        self.cx_gene = cx_gene

        self.setup()

    def setup(self):
        # Define genetic programming elements
        self.pset = gep.PrimitiveSet('Main', input_names=['_open','_high','_low','_close','_volume'
                                             , 'SMA', 'EMA', 'RSI', 'DX', 'ATR', 'BIAS'
                                             ,'ROC'])
        self.pset.add_function(operator.add, 2)
        self.pset.add_function(operator.sub, 2)
        self.pset.add_function(operator.mul, 2)
        self.pset.add_function(protected_div, 2)
        self.pset.add_function(math.sin, 1)       
        self.pset.add_function(math.cos, 1)
        self.pset.add_function(math.tan, 1)

        # 自定义functions
        self.pset.add_function(neg, 1)
        self.pset.add_function(delay, 2)
        self.pset.add_function(delta, 2)
        self.pset.add_function(ts_min, 2)
        self.pset.add_function(ts_max, 2)
        self.pset.add_function(ts_argmin, 2)
        self.pset.add_function(ts_argmax, 2)
        self.pset.add_function(ts_rank, 2)
        self.pset.add_function(ts_sum, 2)
        self.pset.add_function(ts_stddev, 2)
        self.pset.add_function(ts_corr, 3)
        self.pset.add_function(ts_mean_return, 2)

        # 注册到基本功能集合中（如果已有pset对象）
        self.pset.add_function(DEMA, 2)
        self.pset.add_function(KAMA, 2)
        self.pset.add_function(MA, 2)
        self.pset.add_function(MIDPOINT, 2)
        self.pset.add_function(MIDPRICE, 3)

        self.pset.add_ephemeral_terminal(name='enc', gen=lambda: random.randint(-10, 10))

        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

        #h = 10
        #n_genes = 2
        self.toolbox.register('gene_gen', gep.Gene, pset=self.pset, head_length=self.head)
        self.toolbox.register('individual', creator.Individual, gene_gen=self.toolbox.gene_gen, n_genes=self.genes, linker=operator.add)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('compile', gep.compile_, pset=self.pset)

    '''    
    def evaluate(self, individual, X, Y):
        func = self.toolbox.compile(individual)
        Yp = np.array(list(map(func, X)))
        return np.mean(np.abs(Y - Yp)),

    def init_toolbox_with_evaluate(self, X, Y):
        self.toolbox.register('evaluate', self.evaluate, X=X, Y=Y)
    '''

    def evaluate(self,individual, _open, _high, _low, _close, _volume, SMA, EMA, RSI, DX, ATR, BIAS, ROC, data):
        """Evaluate the fitness of an individual: accumulated profits"""

        past_signals = deque(maxlen=10000)
        func = self.toolbox.compile(individual)

        signals = np.array(list(map(func, _open, _high, _low, _close, _volume, SMA, EMA, RSI, DX, ATR, BIAS, ROC)))
        # 原本的dataframe即为Y的来源
        df = data
        #df = df.dropna()

        df['long signal'] = np.nan
        df['short signal'] = np.nan
    
        df['next_open'] = df['open'].shift(-1)
        df['next_open'].fillna(value=df['close'], inplace=True)

        long_or_short = 0
        long_open_pos_price = 0
        short_open_pos_price = 0

        # 只有reset_index()才能将index设置为从0开始
        df = df.reset_index()

        for i in range(len(df)):
        # 开多：当 Si 向上突破 Si-10000~Si 序列的 80 分位数时，则在信号产生的一分钟后开多单
        # 开多平仓：如果策略净值回撤超过 5%时则止盈/止损。
        # 开空：当 Si 向下突破 Si-10000~Si 序列的 20 分位数时，则在信号产生的一分钟后开空单
        # 开空平仓：如果策略净值回撤超过 5%时则止盈/止损。
            past_signals.append(signals[i])

        # 突破检查
            upper_breakout = signals[i] > np.percentile(past_signals, 80)
            lower_breakout = signals[i] < np.percentile(past_signals, 20)

            if upper_breakout and long_or_short == 0:
                df.loc[i, 'long signal'] = 1
                long_open_pos_price = df.loc[i, 'next_open']
                long_or_short = 1

            # 2:1盈亏比
            elif (df.loc[i, 'close']-long_open_pos_price)/long_open_pos_price > 0.1 or (df.loc[i, 'close']-long_open_pos_price)/long_open_pos_price < -0.05 and long_or_short == 1:
                df.loc[i, 'long signal'] = 0
                long_or_short = 0

            elif lower_breakout and long_or_short == 0:
                df.loc[i, 'short signal'] = -1
                short_open_pos_price = df.loc[i, 'next_open']
                long_or_short = -1

            elif (df.loc[i, 'close']-short_open_pos_price)/short_open_pos_price < -0.1 or (df.loc[i, 'close']-short_open_pos_price)/short_open_pos_price > 0.05 and long_or_short == -1:
                df.loc[i,'short signal'] = 0
                long_or_short = 0

        # 把信号序列给清空
        past_signals.clear()

        df['signal'] = df[['long signal', 'short signal']].sum(axis=1, min_count=1, skipna=True)
        # 保留有数据的行数

        temp = df[df['signal'].notnull()][['signal']]
        # 把重复的信号删掉
        temp = temp[temp['signal'] != temp['signal'].shift(1)]
        # 保留有信号的行数
        df['signal'] = temp['signal']

        # 最终保留ohlcv与signal就行
        result = df[['date', 'open', 'high', 'low', 'close', 'volume', 'signal']]
        df = result

        # 把空值补全
        df['signal'].fillna(method='ffill', inplace=True)
        df['signal'].fillna(value=0, inplace=True)
        # 注意在k线走完后第二天才会进行开仓，实际持仓与signal差一行
        df['pos'] = df['signal'].shift()
        df['pos'].fillna(value=0, inplace=True)

        df['next_open'] = df['open'].shift(-1)
        df['next_open'].fillna(value=df['close'], inplace=True)

        # 找出开仓平仓的k线
        condition1 = df['pos'] != 0 # 不空仓
        condition2 = df['pos'] != df['pos'].shift(1) # 当前周期与上一周期状态不一致
        open_pos_condition = condition1 & condition2 # 开仓条件

        condition1 = df['pos'] != 0 # 不空仓
        condition2 = df['pos'] != df['pos'].shift(-1) # 当前周期与下一周期状态不一致
        close_pos_condition = condition1 & condition2 # 平仓条件

        #df = df.reset_index()
    
        df['start_time'] = np.nan
        df.loc[open_pos_condition.values, 'start_time'] = df['date']
        df['start_time'].fillna(method='ffill', inplace=True)
        df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

        # 开始计算资金曲线
        initial_cash = 100000
        face_value = 1 # 每次至少买多少份
        slippage = 1 / 1000
        c_rate = 5 / 10000 # 手续费
        leverage_rate = 1
        min_margin_ratio = 1 / 100 # 最小保证金比例

        # 在开仓时
        # 开仓时contract_num为多少，注意开仓始终是以initial_cash为基准
        df.loc[open_pos_condition.values, 'contract_num'] = initial_cash * leverage_rate / (face_value * df['open'])
        df['contract_num'] = np.floor(df['contract_num'])
        # 考虑滑点的实际开仓价格
        df.loc[open_pos_condition.values, 'open_pos_price'] = df['open'] * (1 + slippage * df['pos'])
        # cash为开仓后扣除手续费后的金额
        df['cash'] = initial_cash - df['open_pos_price'] * face_value * df['contract_num'] * c_rate

        # 在持仓时
        # 买入之后cash, contract_num, open_pos_price不变
        for _ in ['contract_num', 'open_pos_price', 'cash']:
            df[_].fillna(method='ffill', inplace=True)
        df.loc[df['pos'] == 0, ['contract_num', 'open_pos_price', 'cash']] = None
    
        # 在平仓时
        # 平仓价格
        df.loc[close_pos_condition.values, 'close_pos_price'] = df['next_open'] * (1 - slippage * df['pos'])
        # 平仓时手续费
        df.loc[close_pos_condition.values, 'close_pos_fee'] = df['close_pos_price'] * face_value * df['contract_num'] * c_rate

        # 计算利润
        # profit是开仓至今的持仓盈亏
        df['profit'] = face_value * df['contract_num'] * (df['close'] - df['open_pos_price']) * df['pos']
        df.loc[close_pos_condition.values, 'profit'] = face_value * df['contract_num'] * (df['close_pos_price'] - df['open_pos_price']) * df['pos']

        # 账户净值
        #df['net_value'] = df['cash'] + df['profit']

        pf = df.dropna(subset=['profit'])

        # 没有交易记录
        if len(pf) == 0:
            return 0,
        # 存在交易记录
        else:
            # last_profit = pf['profit'].iloc[-1]
            # 计算收益
            #last_return = last_profit / initial_cash

            df['net_value'] = df['cash'] + df['profit']

            df['equity_change'] = df['net_value'].pct_change()
            df.loc[open_pos_condition.values, 'equity_change'] = df.loc[open_pos_condition.values, 'net_value'] / initial_cash - 1
            df['equity_change'].fillna(value=0, inplace=True)
            df['equity_curve'] = (1 + df['equity_change']).cumprod()

            last_profit = df['equity_curve'].iloc[-1]
            last_return = (last_profit - 1) / 1

            # 计算每日的最大累积值（到目前为止的最高点）
            df['cum_max'] = df['equity_curve'].cummax()
            # 计算回撤：从峰值下跌的百分比
            df['drawdown'] = (df['equity_curve'] - df['cum_max']) / df['cum_max']
            # 找到最大回撤
            max_drawdown = df['drawdown'].min()  # 因为回撤是负数，最大回撤是最小值

            return_to_max_drawdown = last_return / abs(max_drawdown)
            return return_to_max_drawdown,

    def init_toolbox_with_evaluate(self, _open, _high, _low, _close, _volume, SMA, EMA, RSI, DX, ATR, BIAS, ROC, data):
        self.toolbox.register('evaluate', self.evaluate, _open=_open, _high=_high, _low=_low, _close=_close, _volume=_volume, SMA=SMA, EMA=EMA, RSI=RSI, DX=DX, ATR=ATR, BIAS=BIAS, ROC=ROC, data=data)
                              
    def run(self, n_pop=100, n_gen=100):
        # 定义适应度函数，用于精英的筛选
        _open = self.data.open.values
        _high = self.data.high.values
        _low = self.data.low.values
        _close = self.data.close.values
        _volume = self.data.volume.values
        SMA = self.data.SMA.values
        EMA = self.data.EMA.values
        RSI = self.data.RSI.values
        DX = self.data.DX.values
        ATR = self.data.ATR.values
        BIAS = self.data.BIAS.values
        ROC = self.data.ROC.values
        
        # 然后初始化针对具体数据评价函数的工具箱设置
        self.init_toolbox_with_evaluate(_open, _high, _low, _close, _volume, SMA, EMA, RSI, DX, ATR, BIAS, ROC, self.data)
        
        # 注册遗传算法的其它操作
        self.toolbox.register('select', tools.selTournament, tournsize=self.tournsize)
        self.toolbox.register('mut_uniform', gep.mutate_uniform, pset=self.pset, ind_pb=0.05, pb=1)
        self.toolbox.register('mut_invert', gep.invert, pb=self.mut_invert)
        self.toolbox.register('mut_is_transpose', gep.is_transpose, pb=self.mut_is_transpose)
        self.toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=self.mut_ris_transpose)
        self.toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=self.mut_gene_transpose)
        self.toolbox.register('cx_1p', gep.crossover_one_point, pb=self.cx_1p)
        self.toolbox.register('cx_2p', gep.crossover_two_point, pb=self.cx_2p)
        self.toolbox.register('cx_gene', gep.crossover_gene, pb=self.cx_gene)

        self.toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: expected one point mutation in an individual
        self.toolbox.pbs['mut_ephemeral'] = 1  # 也可以这样来设置概率
        
        # 统计
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 初始化种群
        pop = self.toolbox.population(n=n_pop)
        
        # 初始化精英策略的记录器
        hof = tools.HallOfFame(3)
        
        # 运行遗传算法
        pop, log = gep.gep_simple(pop, self.toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)
        
        return pop, log, hof

'''
# 生成实例并运行
gp = GeneticProgram(seed=42)
pop, log, hof = gp.run(n_pop=100, n_gen=30)  # 为了测试，可以先设定较少的世代数
'''