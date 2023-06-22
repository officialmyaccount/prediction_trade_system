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
