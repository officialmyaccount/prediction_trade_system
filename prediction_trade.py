# モジュールのインポート
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
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

# 移動平均線の表示
ma_day = [10, 20, 50]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    df[column_name] = df['Adj Close'].rolling(ma).mean()

# ローソク足の表示
mpf.plot(df[-365:], type='candle', volume=True, mav=(10, 20, 50), mavcolors = ('red', 'blue', 'yellow'),  figratio=(16, 6), title=s_target + ' Close Price History')
