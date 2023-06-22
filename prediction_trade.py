# モジュールのインポート
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import r2_score
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

yf.pdr_override()
plt.style.use("fivethirtyeight")

# Googleの情報の取得
s_target = 'GOOG'
df = pdr.get_data_yahoo(s_target, start = '2014-01-01', end = datetime.now())
df.head()

# 株価をグラフで表示
plt.figure(figsize=(16, 6))
plt.title(s_target + 'Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Close Price USD($)', fontsize = 14)
plt.show()

# 移動平均線の表示
ma_day = [10, 20, 50]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    df[column_name] = df['Adj Close'].rolling(ma).mean()

plt.figure(figsize=(16, 6))
plt.title(s_target + 'Close Price MA History')
plt.plot(df['Close'][-300:])
plt.plot(df['MA for 10 days'][-300:])
plt.plot(df['MA for 20 days'][-300:])
plt.plot(df['MA for 50 days'][-300:])
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Close Price USD($)', fontsize = 14)
plt.legend(['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], loc='upper right')
plt.show()
