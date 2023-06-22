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
