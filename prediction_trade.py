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

# Closeのデータ
data = df.filter(['Close'])
dataset = data.values

# データを0から1で正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

# 使用するトレーニングデータの割合
training_data_len = int(np.ceil(len(dataset) * .8))

# 予測する野に必要な期間の指定
window_size = 60
train_data = scaled_data[0: int(training_data_len), :]

# train_dataをxとyに分ける
x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size: i, 0])
    y_train.append(train_data[i, 0])

# numpy arrayに変換
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTMモデルの作成
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, batch_size=32, epochs=5)

# テストデータを作成
test_data = scaled_data[training_data_len - window_size: , :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])

# numpy arrayに変換
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# 予測を実行
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 二乗平均平方根誤差
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

# 決定係数
r2s = r2_score(y_test, predictions)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# 移動平均線の表示
ma_day = [10, 20, 50]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    df[column_name] = df['Adj Close'].rolling(ma).mean()

# ローソク足の表示
mpf.plot(df[-365:], type='candle', volume=True, mav=(10, 20, 50), mavcolors = ('red', 'blue', 'yellow'),  figratio=(16, 6), title=s_target + ' Close Price History')
