import mplfinance as mpf
import pandas as pd
from datetime import timedelta
import base64

from datetime import datetime
import pandas as pd

import json
import http.client, urllib.parse
import pandas as pd


def draw_candlestick(df, savepath):
    # 转换为mplfinance可用的Dataframe
    df['date'] = pd.to_datetime(df['date'])
    df.index = pd.DatetimeIndex(df['date'])
    df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]

    # 创建一个样式
    mc = mpf.make_marketcolors(up='r', down='g', volume='black')
    s  = mpf.make_mpf_style(marketcolors=mc)

    apdict = mpf.make_addplot(df['volume'], panel=1, color='red', secondary_y=False, width=0.5)
    

    # 画图
    mpf.plot(df, type='candle', style=s, addplot=apdict, volume=True, savefig=savepath)


def plot_stock_graph(df, current_time, base_path):
    
    df['date'] = pd.to_datetime(df['date'])
    # 筛选出此时间点之前的所有数据
    df_before_current_time = df[df['date'] <= current_time]

    # 过去10天
    ten_days_ago = pd.to_datetime(current_time) - timedelta(days=10)
    df_past_10_days = df_before_current_time[df_before_current_time['date'] > ten_days_ago]

    # 过去一个月
    one_month_ago = pd.to_datetime(current_time) - pd.DateOffset(months=1)
    df_past_one_month = df_before_current_time[df_before_current_time['date'] > one_month_ago]

    # 过去10天的图生成出来
    save_path = base_path + current_time + '_10days.png'
    draw_candlestick(df_past_10_days, save_path)
    save_path = base_path + current_time + '_1month.png'
    draw_candlestick(df_past_one_month, save_path)

    #return df_past_10_days, df_past_one_month

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def fetch_news(api_key, symbols, limit, current_date, base_path):
    conn = http.client.HTTPSConnection('api.marketaux.com')
    params = urllib.parse.urlencode({
        'api_token': api_key,
        'symbols': symbols,
        'limit': limit,
        'published_on': current_date
        })

    conn.request('GET', '/v1/news/all?{}'.format(params))
    res = conn.getresponse()
    data = res.read()

    # 将字节转化为字符串
    str_data = data.decode("UTF-8")
    json_data = json.loads(str_data)

    news_list = []
    for i in range(min(limit, len(json_data['data']))):
        news_list.append(json_data['data'][i]['description'])
    
    # 将列表转为dataframe
    df = pd.DataFrame(news_list, columns=['News'])
    
    save_path = base_path + current_date + '.feather'

    df.to_feather(save_path)


def generate_ohlcv_changes(df, current_date, base_path):
    df['date'] = pd.to_datetime(df['date']).dt.date  # 将日期转化为date
    dt_current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
    condition = (df['date'] <= dt_current_date)  # 创建布尔条件
    past_df = df[condition]  # 使用布尔索引筛选df

    # 计算过去10天的信息
    past_10_days_df = past_df.tail(10)
    past_10_days = past_10_days_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    past_10_days['change'] = past_10_days['close'].pct_change()  # 计算收盘价涨跌幅

    save_path = base_path + current_date + '_short_term.feather'
    past_10_days.to_feather(save_path)

    # 计算过去30天的信息
    past_month_df = past_df.tail(30)
    past_month = past_month_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    past_month['change'] = past_month['close'].pct_change()  # 计算收盘价涨跌幅

    save_path = base_path + current_date + '_long_term.feather'
    past_month.to_feather(save_path)