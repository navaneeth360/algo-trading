import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from framework import Strategy

class Buy_n_Hold(Strategy):
    def __init__(self, data, test_period = 90, max_p = 1):
        self.data = data
        self.test_period = test_period
        self.max_p = max_p
    
    def implement_strat(self):

        # data is the cleaned data containing ohlc data datewise for a ticker (or multiple tickers)
        # The testing period of the algo is the last test_period days, default 3 months
        close_prices = self.data.iloc[:, 3]
        test_data = close_prices.iloc[-self.test_period:].to_numpy()

        signal = np.array([1] * self.test_period)

        # The signal in this case is just to buy on the first day and hold till the last day when you sell
        signal[-1] = -1

        # Buy on the first day
        curr_pos = [self.max_p] * (self.test_period)
        buy_price = test_data[0]
        balance = [-self.max_p*buy_price] * (self.test_period)
        pnl = [0] * (self.test_period)
        num_trades = self.max_p
        
        # We'll neutralize the position on the last day and check
        if (curr_pos[-1] != 0):
            qty_ = (-1)*curr_pos[-1]
            num_trades = num_trades + np.abs(qty_)
            amt_ = test_data[-1]*(-1)*qty_
            curr_pos[-1] = curr_pos[-1] + qty_
            balance[-1] = balance[-1] + amt_
            pnl[-1] = pnl[-1] + amt_

        balance = balance
        curr_pos = curr_pos
        pnl = pnl
        portfolio_value = curr_pos*test_data

        self.balance = balance
        self.positions = curr_pos
        self.close_prices = test_data
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return_dict = {"positions" : curr_pos ,
                       "balance" : balance, 
                       "close_prices" : test_data,
                       "pnl": pnl, 
                       "num_trades" : num_trades,
                       "portfolio_value" : portfolio_value}
        return return_dict
    
    def plotter(self):
        plt.style.use('seaborn-v0_8-darkgrid')  # Use a modern, clean style

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        
        # Second subplot: PnL
        axs.plot(self.pnl, label="PnL", color="purple", linewidth=1)
        axs.set_title("Strategy PnL")
        axs.set_xlabel("Time")
        axs.set_ylabel("PnL")
        axs.legend(loc="upper left")
        axs.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()