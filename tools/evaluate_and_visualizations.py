import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_equity_curve_with_split(backtest_test_return, bt_return):
    # 得到分割时间，并转换为字符串格式，用于图例
    split_time = backtest_test_return['date'][0]
    split_time_str = str(split_time)
    
    # 设置图形尺寸
    plt.figure(figsize=(10, 6))
    
    # 绘制equity_curve
    plt.plot(bt_return['date'], bt_return['equity_curve'], label='Equity Curve')
    
    # 添加红色虚线标出分割时间点
    plt.axvline(split_time, color='red', linestyle='--', label=f'Split Time: {split_time_str}')
    plt.legend()
    
    # 设置标题和坐标轴标签
    plt.title('Equity Curve Before and After the GEP')
    plt.xlabel('Date')
    plt.ylabel('Equity Curve')
    
    plt.show()


def get_backtest_ratios(df, rf=0.02):
    # 计算指标
    final_value = df['equity_curve'].iloc[-1]  # 最终的净值
    return_rate = ( final_value - 1) * 100  # 盈利率

    # 最大回撤
    peak = df['net_value'].expanding().max()
    drawdown = (df['net_value'] - peak) / peak
    max_drawdown = drawdown.min() * 100  # 最大回撤

    # 夏普比率
    # 假定无风险利率为0，实际操作中应使用实际的无风险回报率
    risk_free_rate = rf
    df['daily_return'] = df['equity_curve'].pct_change()

    # 计算夏普比率，年化因子取决于数据样本频率
    annualized_return = np.mean(df['daily_return']) * 252
    annualized_volatility = np.std(df['daily_return']) * np.sqrt(252)

    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    return return_rate, max_drawdown, sharpe_ratio


# train_or_test = 'backtest_train_return' or 'backtest_test_return'
def plot_train_or_test_return(df, title, rf=0.02):

    # 计算指标
    final_value = df['equity_curve'].iloc[-1]  # 最终的净值
    return_rate = ( final_value - 1) * 100  # 盈利率

    # 最大回撤
    peak = df['net_value'].expanding().max()
    drawdown = (df['net_value'] - peak) / peak
    max_drawdown = drawdown.min() * 100  # 最大回撤

    # 夏普比率
    # 假定无风险利率为0，实际操作中应使用实际的无风险回报率
    risk_free_rate = rf
    df['daily_return'] = df['equity_curve'].pct_change()

    # 计算夏普比率，年化因子取决于数据样本频率
    annualized_return = np.mean(df['daily_return']) * 252
    annualized_volatility = np.std(df['daily_return']) * np.sqrt(252)

    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    # 绘制 Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['equity_curve'], label='Equity Curve', color='blue')
    plt.fill_between(df['date'], df['equity_curve'], 1, where=df['equity_curve'] < 1, color='red', alpha=0.3)
    plt.fill_between(df['date'], df['equity_curve'], 1, where=df['equity_curve'] >= 1, color='green', alpha=0.3)

    # 标注
    plt.text(0.05, 0.40, f'Return: {return_rate:.2f}%\nMax Drawdown: {max_drawdown:.2f}%\nSharte Ratio: {sharpe_ratio:.2f}',
         verticalalignment='top', horizontalalignment='left', transform=plt.gca().transAxes)


    if title == 'backtest_train_return':
        plt.title('Equity Curve for Training Set')
    else:
        plt.title('Equity Curve for Testing Set')

    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()

def plot_return(data, train_or_test, rf=0.02):
    df = data

    # 计算指标
    final_value = df['equity_curve'].iloc[-1]  # 最终的净值
    return_rate = ( final_value - 1) * 100  # 盈利率

    # 最大回撤
    peak = df['net_value'].expanding().max()
    drawdown = (df['net_value'] - peak) / peak
    max_drawdown = drawdown.min() * 100  # 最大回撤

    # 夏普比率
    # 假定无风险利率为0，实际操作中应使用实际的无风险回报率
    risk_free_rate = rf
    df['daily_return'] = df['equity_curve'].pct_change()

    # 计算夏普比率，年化因子取决于数据样本频率
    annualized_return = np.mean(df['daily_return']) * 252
    annualized_volatility = np.std(df['daily_return']) * np.sqrt(252)

    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    # 绘制 Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['equity_curve'], label='Equity Curve', color='blue')
    plt.fill_between(df['date'], df['equity_curve'], 1, where=df['equity_curve'] < 1, color='red', alpha=0.3)
    plt.fill_between(df['date'], df['equity_curve'], 1, where=df['equity_curve'] >= 1, color='green', alpha=0.3)

    # 标注
    plt.text(0.05, 0.40, f'Return: {return_rate:.2f}%\nMax Drawdown: {max_drawdown:.2f}%\nSharte Ratio: {sharpe_ratio:.2f}',
         verticalalignment='top', horizontalalignment='left', transform=plt.gca().transAxes)


    if train_or_test == 'backtest_train_return':
        plt.title('Equity Curve for Training Set')
    else:
        plt.title('Equity Curve for Testing Set')

    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()
    plt.clf()

#生成数据库测试文件
def get_clean_databse(path):
    df = {
    "signal": ['Test'],
    "盈利率": [0],
    "最大回撤": [0],
    "夏普比率": [0],
    }
    df = pd.DataFrame(df)

    df.to_feather(path)

def calculate_return_rate(df):
    # 计算指标
    final_value = df['equity_curve'].iloc[-1]  # 最终的净值
    return_rate = ( final_value - 1)  # 盈利率
    return return_rate

def calculate_max_drawdown(df):
    # 最大回撤
    peak = df['net_value'].expanding().max()
    drawdown = (df['net_value'] - peak) / peak
    max_drawdown = drawdown.min()  # 最大回撤
    return max_drawdown

def calculate_sharpe_ratio(df):
    # 夏普比率
    # 假定无风险利率为0，实际操作中应使用实际的无风险回报率
    risk_free_rate = 0.02
    df['daily_return'] = df['equity_curve'].pct_change()

    # 计算夏普比率，年化因子取决于数据样本频率
    annualized_return = np.mean(df['daily_return']) * 252
    annualized_volatility = np.std(df['daily_return']) * np.sqrt(252)

    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    return sharpe_ratio


def add_result_to_database(return_df, signal_express, path):
    signals = signal_express
    return_rate = calculate_return_rate(return_df)
    drawdown = calculate_max_drawdown(return_df)
    sharp_rate = calculate_sharpe_ratio(return_df)

    original_result = pd.read_feather(path)

    new_data = {
    "signal": [signals],
    "盈利率": [return_rate],
    "最大回撤": [drawdown],
    "夏普比率": [sharp_rate],
    }

    df_new = pd.DataFrame(new_data)

    # 将新的 DataFrame 添加(横向拼接)到原来的 DataFrame
    result = pd.concat([original_result, df_new], axis=0)

    # 将新的 DataFrame 保存为 feather 文件
    result.to_feather(path)