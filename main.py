
import yfinance as yf
import stocks
from moving_average import SMA, EMA
from framework import myticker
from Buy_n_hold import Buy_n_Hold
from RSI import RSI_
from Bbands import Bollinger
from ML_strats import Lin_Reg
from lstm import LSTM_

ticker = yf.Ticker("NVDA")
ms = myticker(ticker)
print("ms", ms.train_test_data)
print(len(ms.ma_data))

strat = SMA(ms.ma_data, stocks.ma_slow, stocks.ma_fast, stocks.test_period, stocks.max_p)
outp = strat.implement_strat()
strat.get_stats()
strat.plotter()

strat1 = EMA(ms.ma_data, stocks.ma_slow, stocks.ma_fast, stocks.test_period, stocks.max_p)
outp1 = strat1.implement_strat()
strat1.get_stats()
strat1.plotter()

strat2 = Buy_n_Hold(ms.test_data, stocks.test_period, stocks.max_p)
outp2 = strat2.implement_strat()
strat2.get_stats()
strat2.plotter()

strat3 = RSI_(ms.ma_data, stocks.test_period, stocks.rsi_period, stocks.rsi_high_threshold, stocks.rsi_low_threshold, stocks.max_p)
outp4 = strat3.implement_strat()
strat3.get_stats()
strat3.plotter()

strat4 = Bollinger(ms.ma_data, stocks.bol_window, stocks.k, stocks.test_period, stocks.k)
outp4 = strat4.implement_strat()
strat4.get_stats()
strat4.plotter()

strat5 = Lin_Reg(ms.train_test_data, stocks.ml_lookback, stocks.ml_threshold, stocks.test_period, stocks.max_p)
outp5 = strat5.implement_strat()
strat5.get_stats()
strat5.plotter()

strat6 = LSTM_()
outp6 = strat6.implement_strat()
strat6.get_stats()
strat6.plotter()


    

    
    

