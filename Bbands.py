import numpy as np
from framework import Strategy
import matplotlib.pyplot as plt
import stocks

class Bollinger(Strategy):
    def __init__(self, data, bol_window = 20, k = 2, test_period = 90, max_p = 1):
        self.data = data
        self.bol_window = bol_window
        self.k = k
        self.test_period = test_period
        self.max_p = max_p
    
    def implement_strat(self):

        # data is the cleaned data containing ohlc data datewise for a ticker (or multiple tickers)
        # The testing period of the algo is the last test_period days, default 3 months
        close_prices = self.data.iloc[:, 3]
        test_data = close_prices.iloc[-self.test_period:].to_numpy()
        close_prices_sig = close_prices.iloc[-(self.test_period+1):-1].to_numpy()

        # Use the close prices to calculate the moving average and standard deviation
        moving_avg = close_prices.rolling(window=self.bol_window).mean()
        moving_std = close_prices.rolling(window=self.bol_window).std()
        
        moving_avg = moving_avg.iloc[-(self.test_period+1):-1].to_numpy()
        moving_std = moving_std.iloc[-(self.test_period+1):-1].to_numpy()

        # Now using these to define the upper and lower Bollinger bands
        upper_band = moving_avg + self.k * moving_std
        middle_band = moving_avg
        lower_band = moving_avg - self.k * moving_std

        # Below is the Bollinger Bounce strategy, where we buy if the price goes below the lower band
        # and sell if it goes above the upper band
        buy_signal = lower_band - close_prices_sig
        sell_signal = close_prices_sig - upper_band


        curr_pos = [0] * (self.test_period+1)
        balance = [0] * (self.test_period+1)
        pnl = [0] * (self.test_period+1)
        num_trades = 0
        for i in range(len(buy_signal)):
            if buy_signal[i] >= 0:
                # We need to neutralize position and buy if we were already short
                if curr_pos[i] <= 0:
                    qty_bought = self.max_p + (-1*curr_pos[i])
                    num_trades = num_trades + qty_bought
                    amt_spent = -1*(qty_bought*test_data[i])
                    curr_pos[i+1] = curr_pos[i] + qty_bought
                    balance[i+1] = balance[i] + amt_spent
                    pnl[i+1] = pnl[i] + (balance[i] - (-1*curr_pos[i])*test_data[i])
                else:
                    curr_pos[i+1] = curr_pos[i]
                    balance[i+1] = balance[i]
                    pnl[i+1] = pnl[i]
            
            elif (sell_signal[i] >= 0):
                # We need to neutralize position and sell if we were already long
                if curr_pos[i] >= 0:
                    qty_sold = self.max_p + curr_pos[i]
                    num_trades = num_trades + qty_sold
                    amt_received = (qty_sold*test_data[i])
                    curr_pos[i+1] = curr_pos[i] - qty_sold
                    balance[i+1] = balance[i] + amt_received
                    pnl[i+1] = pnl[i] + (balance[i] + (curr_pos[i])*test_data[i])
                else:
                    curr_pos[i+1] = curr_pos[i]
                    balance[i+1] = balance[i]
                    pnl[i+1] = pnl[i]
            
            else:
                curr_pos[i+1] = curr_pos[i]
                balance[i+1] = balance[i]
                pnl[i+1] = pnl[i]

        # We'll neutralize the position on the last day and check
        if (curr_pos[-1] != 0):
            qty_ = (-1)*curr_pos[-1]
            num_trades = num_trades + qty_
            amt_ = test_data[-1]*(-1)*qty_
            curr_pos[-1] = curr_pos[-1] + qty_
            balance[-1] = balance[-1] + amt_
            pnl[-1] = pnl[-1] + amt_

        balance = balance[1:]
        curr_pos = curr_pos[1:]
        pnl = pnl[1:]
        portfolio_value = np.array(curr_pos)*test_data

        self.balance = balance
        self.positions = curr_pos
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.close_prices = test_data
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return_dict = {"positions" : curr_pos ,
                       "balance" : balance, 
                       "upper_band" : upper_band,
                       "lower_band" : lower_band,
                       "close_prices" : test_data,
                       "pnl": pnl, 
                       "num_trades" : num_trades,
                       "portfolio_value" : portfolio_value}
        return return_dict
    
    def plotter(self):
        plt.style.use('seaborn-v0_8-darkgrid')  

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # First subplot: Price and SMAs
        axs[0].plot(self.close_prices, label="Close Price", color="blue", linewidth=1)
        axs[0].plot(self.upper_band, label=f"Upper Bollinger Band", color="red", linestyle="--", linewidth=1)
        axs[0].plot(self.lower_band, label=f"Lower Bollinger Band", color="green", linestyle="--", linewidth=1)
        axs[0].set_title("Price and Bollinger Bands")
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="upper left")
        axs[0].grid(True, alpha=0.3)

        # Second subplot: PnL
        axs[1].plot(self.pnl, label="PnL", color="purple", linewidth=1)
        axs[1].set_title("Strategy PnL")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("PnL")
        axs[1].legend(loc="upper left")
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()