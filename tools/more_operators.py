# 自定义算子
import math
import talib
import numpy as np
from scipy.stats import rankdata

def add(a, b):
    "Same as a + b."
    return a + b

def sub(a, b):
    "Same as a - b."
    return a - b

def mul(a, b):
    "Same as a * b."
    return a * b

# 保护性除法
def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

# 返回相反数
def neg(X):
    return -X

# 返回d天以前的数据
def delay(data, d):
    # 有可能常数会误传给data
    # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        return 0
    d = int(d)
    data_len = len(data)
    if d >= data_len:
        return None
    else:
        return data[data_len - d - 1]

# 返回data-d天之前的差值
def delta(data, d):
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
        return 0
    d = int(d)
    data_len = len(data)
    if d >= data_len:
        return None
    else:
        return data[data_len - 1] - data[data_len - d - 1]

# 返回过去d天里时序最小的值
def ts_min(data, d):
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return min(data)  # 如果历史天数越界，可以考虑返回全部历史天数的最小值
    else:
        return min(data[data_len - d:])

# 返回过去d天里时序最大的值
def ts_max(data, d):
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return max(data)
    else:
        return max(data[data_len - d:])

# 返回过去d天X值构成的时序数列中最小值出现的位置。
def ts_argmin(data, d):
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return data.index(min(data))  # 如果历史天数越界，可以考虑返回全部历史天数的最小值的位置
    else:
        return data.index(min(data[data_len - d:]))

# 返回过去d天X值构成的时序数列中最大值出现的位置。
def ts_argmax(data, d):
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        # 如果 data 不是 list 或 ndarray 类型，则返回一个默认值
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return data.index(max(data))
    else:
        return data.index(max(data[data_len - d:]))
    
# 过去 d 天 X 值构成的时序数列中本截面日𝑋𝑖值所处分位数。
def ts_rank(data, d):
    if not isinstance(data, (list, np.ndarray)):
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        ranks = rankdata(data)
        return ranks[-1] / len(ranks)
    else:
        ranks = rankdata(data[-d:])
        return ranks[-1] / len(ranks)

# 过去 d 天 X 值构成的时序数列之和
def ts_sum(data, d):
    if not isinstance(data, (list, np.ndarray)):
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return sum(data)
    else:
        return sum(data[-d:])

# 过去 d 天 X 值构成的时序数列的标准差
def ts_stddev(data, d):
    if not isinstance(data, (list, np.ndarray)):
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        return np.std(data)
    else:
        return np.std(data[-d:])

# 过去 d 天 X 值构成的时序数列与Y 构成的时序数列的相关系数
def ts_corr(data_x, data_y, d):
    if not (isinstance(data_x, (list, np.ndarray)) and isinstance(data_y, (list, np.ndarray))):
        return 0
    d = int(d)
    data_len = len(data_x)
    if len(data_y) != data_len:
        return 0
    if d > data_len:
        return np.corrcoef(data_x, data_y)[0, 1]
    else:
        return np.corrcoef(data_x[-d:], data_y[-d:])[0, 1]

# 过去 d 天 X 值构成的时序数列的变化率的平均值
def ts_mean_return(data, d):
    if not isinstance(data, (list, np.ndarray)):
        return 0
    d = int(d)
    data_len = len(data)
    if d > data_len:
        returns = (np.array(data[1:]) - np.array(data[:-1])) / np.array(data[:-1])
        return np.mean(returns)
    else:
        returns = (np.array(data[-d + 1:]) - np.array(data[-d:-1])) / np.array(data[-d:-1])
        return np.mean(returns)
    
# 以下是利用talib生成的一些指标函数

# 过去 d 天 X 值的双移动平均线，属于趋势信号
def DEMA(X, d):
    if not isinstance(X, (list, np.ndarray)):
        return 0
    d = int(d)
    try:
        result = talib.DEMA(np.array(X, dtype=float), timeperiod=d)
        return result[-1]  # 返回最新的 DEMA 值
    except:
        return 0

# 过去 d 天 X 值的考夫曼自适应移动平均线，属于趋势信号
def KAMA(X, d):
    if not isinstance(X, (list, np.ndarray)):
        return 0
    d = int(d)
    try:
        result = talib.KAMA(np.array(X, dtype=float), timeperiod=d)
        return result[-1]  # 返回最新的 KAMA 值
    except:
        return 0

# 过去 d 天 X 构成的时序数列的平均值，属于趋势信号
def MA(X, d):
    if not isinstance(X, (list, np.ndarray)):
        return 0
    d = int(d)
    try:
        result = talib.MA(np.array(X, dtype=float), timeperiod=d)
        return result[-1]  # 返回最新的 MA 值
    except:
        return 0

# 过去 d 天 X 值构成的时序数列的最大值与最小值的平均值
def MIDPOINT(X, d):
    if not isinstance(X, (list, np.ndarray)):
        return 0
    d = int(d)
    try:
        result = talib.MIDPOINT(np.array(X, dtype=float), timeperiod=d)
        return result[-1]  # 返回最新的 MIDPOINT 值
    except:
        return 0

# 过去 d 天 high 序列的最大值与 low 序列的最小值的平均值
def MIDPRICE(high, low, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.MIDPRICE(np.array(high, dtype=float), np.array(low, dtype=float), timeperiod=d)
        return result[-1]  # 返回最新的 MIDPRICE 值
    except:
        return 0

# 过去 d 天的阿隆震荡指标，属于动量信号
def AROONOSC(high, low, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.AROONOSC(high, low, timeperiod=d)
        return result[-1]  # 返回最新的 AROONOSC 值
    except:
        return 0

# 过去 d 天的威廉指标，表示的是市场属于超买还是超卖状态，属于动量信号
def WILLR(high, low, close, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray)) and isinstance(close, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.WILLR(high, low, close, timeperiod=d)
        return result[-1]  # 返回最新的 WILLR 值
    except:
        return 0

# 过去 d 天的顺势指标，测量股价是否已超出正常分布范围，属于动量信号
def CCI(high, low, close, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray)) and isinstance(close, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.CCI(high, low, close, timeperiod=d)
        return result[-1]  # 返回最新的 CCI 值
    except:
        return 0

# 过去 d 天的平均趋向指数，指标判断盘整、震荡和单边趋势，属于动量信号
def ADX(high, low, close, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray)) and isinstance(close, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.ADX(high, low, close, timeperiod=d)
        return result[-1]  # 返回最新的 ADX 值
    except:
        return 0

# 过去 d 天的资金流量指标，反映市场的运行趋势，属于动量信号
def MFI(high, low, close, volume, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray)) and isinstance(close, (list, np.ndarray)) and isinstance(volume, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.MFI(high, low, close, volume, timeperiod=d)
        return result[-1]  # 返回最新的 MFI 值
    except:
        return 0

# 过去 d 天的归一化波动幅度均值，属于波动性信号。
def NATR(high, low, close, d):
    if not (isinstance(high, (list, np.ndarray)) and isinstance(low, (list, np.ndarray)) and isinstance(close, (list, np.ndarray))):
        return 0
    d = int(d)
    try:
        result = talib.NATR(high, low, close, timeperiod=d)
        return result[-1]  # 返回最新的 NATR 值
    except:
        return 0