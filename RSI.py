import numpy as np
from framework import Strategy
import matplotlib.pyplot as plt
import stocks

class RSI_(Strategy):
    def __init__(self, data, test_period = 90, rsi_period = 14, rsi_high_threshold = 70, rsi_low_threshold  = 30, max_p = 1):
        self.data = data
        self.rsi_period = rsi_period
        self.rsi_low_threshold = rsi_low_threshold
        self.rsi_high_threshold = rsi_high_threshold
        self.test_period = test_period
        self.max_p = max_p
    
    def implement_strat(self):

        # data is the cleaned data containing ohlc data datewise for a ticker (or multiple tickers)
        # The testing period of the algo is the last test_period days, default 3 months
        close_prices = self.data.iloc[:, 3]
        returns = close_prices.iloc[1:].to_numpy() - close_prices.iloc[:-1].to_numpy()
        # print("we got close prices? : ", close_prices)
        # print("we got returns ? : ", returns)
        
        test_data = returns[-self.test_period :]
        close_prices = close_prices.iloc[-self.test_period :].to_numpy()

        # For RSI, we need the testing period data and rsi_period more days for calculating the initial RSI
        returns = returns[-(self.test_period + self.rsi_period) :]
        # print(self.test_period, self.rsi_period)
        # print("returns length is : ", len(returns))

        # print("returns : ", returns[-5:])
        # print("close : ", close_prices[-5:])

        # Initial avg calculations
        avg_gain = 0
        avg_loss = 0
        for i in range(self.rsi_period):
            if returns[i] > 0:
                avg_gain = avg_gain + returns[i]
            else:
                avg_loss = avg_loss + returns[i]

        avg_gain = avg_gain / self.rsi_period
        avg_loss = avg_loss / self.rsi_period

        RS = avg_gain / np.abs(avg_loss)
        RSI = 100 - (100 / (1 + RS))

        # From now onwards we can just calculate RSI for each day
        signal = [0]*(self.rsi_period + self.test_period)
        signal[self.rsi_period - 1] = RSI

        for j in range(self.rsi_period, self.rsi_period + self.test_period):
            curr_ret = returns[j]
            if curr_ret > 0:
                avg_gain = (avg_gain*(self.rsi_period - 1) + curr_ret) / self.rsi_period
            else:
                avg_loss = (avg_loss*(self.rsi_period - 1) + curr_ret) / self.rsi_period
            try:
                RS = avg_gain / np.abs(avg_loss)
            except:
                RS = 100
            RSI = 100 - (100 / (1 + RS))
            signal[j] = RSI
        
        signal = signal[self.rsi_period - 1 : -1]
        # print("max rsi is  : ", max(signal))
        # print("min rsi is  : ", min(signal))
        # print("what're the thresholds ? : ", self.rsi_low_threshold, self.rsi_high_threshold)
    

        # If the signal, i.e, RSI is below the rsi_threshold, then we buy and if it's above (100 - rsi_threshold) then we sell
        curr_pos = [0] * (self.test_period+1)
        balance = [0] * (self.test_period+1)
        pnl = [0] * (self.test_period+1)
        num_trades = 0
        for i in range(len(signal)):
            if signal[i] > self.rsi_high_threshold:
                # We need to neutralize position and buy if we were already short
                if curr_pos[i] <= 0:
                    qty_bought = self.max_p + (-1*curr_pos[i])
                    num_trades = num_trades + qty_bought
                    amt_spent = -1*(qty_bought*close_prices[i])
                    curr_pos[i+1] = curr_pos[i] + qty_bought
                    balance[i+1] = balance[i] + amt_spent
                    pnl[i+1] = pnl[i] + (balance[i] - (-1*curr_pos[i])*close_prices[i])
                else:
                    curr_pos[i+1] = curr_pos[i]
                    balance[i+1] = balance[i]
                    pnl[i+1] = pnl[i]
            
            elif (signal[i] < self.rsi_low_threshold):
                # We need to neutralize position and sell if we were already long
                if curr_pos[i] >= 0:
                    qty_sold = self.max_p + curr_pos[i]
                    num_trades = num_trades + qty_sold
                    amt_received = (qty_sold*close_prices[i])
                    curr_pos[i+1] = curr_pos[i] - qty_sold
                    balance[i+1] = balance[i] + amt_received
                    pnl[i+1] = pnl[i] + (balance[i] + (curr_pos[i])*close_prices[i])
                else:
                    curr_pos[i+1] = curr_pos[i]
                    balance[i+1] = balance[i]
                    pnl[i+1] = pnl[i]

            else:
                # We do nothing, just wait
                curr_pos[i+1] = curr_pos[i]
                balance[i+1] = balance[i]
                pnl[i+1] = pnl[i]

        # We'll neutralize the position on the last day and check
        if (curr_pos[-1] != 0):
            qty_ = (-1)*curr_pos[-1]
            num_trades = num_trades + qty_
            amt_ = close_prices[-1]*(-1)*qty_
            curr_pos[-1] = curr_pos[-1] + qty_
            balance[-1] = balance[-1] + amt_
            pnl[-1] = pnl[-1] + amt_

        balance = balance[1:]
        curr_pos = curr_pos[1:]
        pnl = pnl[1:]
        portfolio_value = curr_pos*close_prices

        self.balance = balance
        self.positions = curr_pos
        self.rsi = signal
        self.close_prices = close_prices
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return_dict = {"positions" : curr_pos ,
                       "balance" : balance, 
                       "rsi" : signal,
                       "close_prices" : test_data,
                       "pnl": pnl, 
                       "num_trades" : num_trades,
                       "portfolio_value" : portfolio_value}
        return return_dict