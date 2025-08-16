import numpy as np
from framework import Strategy
import matplotlib.pyplot as plt
import stocks

class SMA(Strategy):
    def __init__(self, data, ma_slow = 200, ma_fast = 50, test_period = 90, max_p = 1):
        self.data = data
        self.ma_slow = ma_slow
        self.ma_fast = ma_fast
        self.test_period = test_period
        self.max_p = max_p
    
    def implement_strat(self):

        # data is the cleaned data containing ohlc data datewise for a ticker (or multiple tickers)
        # The testing period of the algo is the last test_period days, default 3 months
        close_prices = self.data.iloc[:, 3]
        test_data = close_prices.iloc[-self.test_period:].to_numpy()

        # Use the close prices to calculate the 50 and 200 day SMA (which I call fast and slow SMA)
        sma_slow_full = close_prices.rolling(window=self.ma_slow).mean()
        sma_fast_full = close_prices.rolling(window=self.ma_fast).mean()
        
        sma_slow = sma_slow_full.iloc[-(self.test_period+1):-1].to_numpy()
        sma_fast = sma_fast_full.iloc[-(self.test_period+1):-1].to_numpy()
        signal = np.array(sma_fast) - np.array(sma_slow)


        
        # sma_slow = [0] * self.test_period
        # sma_fast = [0] * self.test_period
        # for i in range(self.test_period):
        #     sma_slow[i] = np.mean(close_prices[-1*(self.test_period - i + self.ma_slow) : -1*(self.test_period - i)])
        #     sma_fast[i] = np.mean(close_prices[-1*(self.test_period - i + self.ma_fast) : -1*(self.test_period - i)])
        #     print(i, sma_slow[i], "end index", -1*(self.test_period - i))
        # signal = np.array(sma_fast) - np.array(sma_slow)

        

        # When the fast sma crosses above the slow sma, it is known as the Golden Cross 
        # This is an indicator to buy
        # The opposite scenario is the Death Cross, when we would sell

        # Position is +ve for long, -ve for short 
        # The below part of the algo is the strategy we use on the signal, which could vary in many ways
        # We could take varying position based on signal magnitude, keep buying until signal sign changes, gradually change position etc.
        # Below is the "Classical SMA" where we change position only when the signal changes sign, else we don't buy or sell
        curr_pos = [0] * (self.test_period+1)
        balance = [0] * (self.test_period+1)
        pnl = [0] * (self.test_period+1)
        num_trades = 0
        for i in range(len(signal)):
            if signal[i] > 0:
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
            
            else:
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
        portfolio_value = curr_pos*test_data

        self.balance = balance
        self.positions = curr_pos
        self.sma_slow = sma_slow
        self.sma_fast = sma_fast
        self.close_prices = test_data
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return_dict = {"positions" : curr_pos ,
                       "balance" : balance, 
                       "sma_slow" : sma_slow,
                       "sma_fast" : sma_fast,
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
        axs[0].plot(self.sma_slow, label=f"SMA Slow ({stocks.ma_slow})", color="red", linestyle="--", linewidth=1)
        axs[0].plot(self.sma_fast, label=f"SMA Fast ({stocks.ma_fast})", color="green", linestyle="--", linewidth=1)
        axs[0].set_title("Price and Moving Averages")
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


class EMA(Strategy):
    def __init__(self, data, ma_slow = 200, ma_fast = 50, test_period = 90, max_p = 1):
        self.data = data
        self.ma_slow = ma_slow
        self.ma_fast = ma_fast
        self.test_period = test_period
        self.max_p = max_p
    
    def implement_strat(self):

        # data is the cleaned data containing ohlc data datewise for a ticker (or multiple tickers)
        # The testing period of the algo is the last test_period days, default 3 months
        close_prices = self.data.iloc[:, 3]
        test_data = close_prices.iloc[-self.test_period:].to_numpy()

        # Use the close prices to calculate the 50 and 200 day SMA (which I call fast and slow SMA)
        ema_slow_full = close_prices.ewm(span=self.ma_slow, adjust = False).mean()
        ema_fast_full = close_prices.ewm(span=self.ma_fast, adjust = False).mean()
        
        ema_slow = ema_slow_full.iloc[-(self.test_period+1):-1].to_numpy()
        ema_fast = ema_fast_full.iloc[-(self.test_period+1):-1].to_numpy()
        signal = np.array(ema_fast) - np.array(ema_slow)


        
        # sma_slow = [0] * self.test_period
        # sma_fast = [0] * self.test_period
        # for i in range(self.test_period):
        #     sma_slow[i] = np.mean(close_prices[-1*(self.test_period - i + self.ma_slow) : -1*(self.test_period - i)])
        #     sma_fast[i] = np.mean(close_prices[-1*(self.test_period - i + self.ma_fast) : -1*(self.test_period - i)])
        #     print(i, sma_slow[i], "end index", -1*(self.test_period - i))
        # signal = np.array(sma_fast) - np.array(sma_slow)

        

        # When the fast sma crosses above the slow sma, it is known as the Golden Cross 
        # This is an indicator to buy
        # The opposite scenario is the Death Cross, when we would sell

        # Position is +ve for long, -ve for short 
        # The below part of the algo is the strategy we use on the signal, which could vary in many ways
        # We could take varying position based on signal magnitude, keep buying until signal sign changes, gradually change position etc.
        # Below is the "Classical SMA" where we change position only when the signal changes sign, else we don't buy or sell
        curr_pos = [0] * (self.test_period+1)
        balance = [0] * (self.test_period+1)
        pnl = [0] * (self.test_period+1)
        num_trades = 0
        for i in range(len(signal)):
            if signal[i] > 0:
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
            
            else:
                # We need to neutralize position and sell if we were already short
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
        portfolio_value = curr_pos*test_data

        self.balance = balance
        self.positions = curr_pos
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        self.close_prices = test_data
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return_dict = {"positions" : curr_pos ,
                       "balance" : balance, 
                       "ema_slow" : ema_slow,
                       "ema_fast" : ema_fast,
                       "close_prices" : test_data,
                       "pnl": pnl, 
                       "num_trades" : num_trades,
                       "portfolio_value" : portfolio_value}
        return return_dict
    
    def plotter(self):
        plt.style.use('seaborn-v0_8-darkgrid')  # Use a modern, clean style

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # First subplot: Price and SMAs
        axs[0].plot(self.close_prices, label="Close Price", color="blue", linewidth=1)
        axs[0].plot(self.ema_slow, label=f"EMA Slow ({stocks.ma_slow})", color="red", linestyle="--", linewidth=1)
        axs[0].plot(self.ema_fast, label=f"EMA Fast ({stocks.ma_fast})", color="green", linestyle="--", linewidth=1)
        axs[0].set_title("Price and Moving Averages")
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






                





        



