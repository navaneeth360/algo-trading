from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import datetime
import stocks
import pandas as pd
import yfinance as yf

class Strategy(ABC):
    @classmethod
    def implement_strat(self):
        # Function to implement or run the strategy
        # Varies depending on the strategy
        pass

    @classmethod
    def plotter(self):
        # Function to plot the cumulative profits, position 
        plt.plot(self.pnl)
        plt.show()
        pass

    def get_stats(self):
        # Function to determine how good the strategy is by finding various metrics
        mean_profit = sum(self.pnl)/len(self.pnl)
        std_profit = np.std(self.pnl)
        sharpe = mean_profit / std_profit
        print("Sharpe is : ", sharpe)
        final_pnl = self.pnl[-1]
        print("Cumulative pnl is : ", final_pnl)
        print("Profit per trade is : ", final_pnl / self.num_trades)
        print("Number of trades executed : ", self.num_trades)
        max_drawdown = 0
        peak = self.portfolio_value[0]
        for i in self.portfolio_value:
            if i > peak:
                peak = i
            max_drawdown = max((peak - i) / peak, max_drawdown)
        print("Maximum drawdown is : ", max_drawdown*100)
        pass

class myticker:
    # This class is a wrapper around yfinance Ticker to fetch and clean stock data, just give the ticker symbol as input
    def __init__(self, ticker):
        self.ticker = ticker
        # For ML models, we will use 1 year of data for training (365 days) and 3 months for testing (90 days)
        train_start = datetime.datetime.now() - datetime.timedelta(days=(stocks.ml_train_period + stocks.test_period)*2)
        test_end = datetime.datetime.now()

        self.train_test_data = self.get_past_data(ticker, start = train_start, end = test_end)

        self.train_test_data = self.clean_data(self.train_test_data)

        # For MA based strategies, we will use 3 months of data for testing and we will need to fetch data for the first testing day
        # Extracting twice the required amount of data to compensate for weekends
        start_date = datetime.datetime.now() - datetime.timedelta(days=(stocks.test_period + stocks.ma_slow + 1)*2)
        
        self.ma_data = self.get_past_data(ticker, start=start_date, end=test_end)
        self.ma_data = self.clean_data(self.ma_data)

    @staticmethod
    def get_past_data(ticker, start=None, end=None, interval=365):
        stock = ticker
        if start is None or end is None:
            start = datetime.datetime.now() - datetime.timedelta(days=interval)
            end = datetime.datetime.now()
            data = stock.history(start=start, end=end)
        else:
            data = stock.history(start=start, end=end)

        if data.empty:
            print(f"No data returned for ticker {ticker}")
            raise ValueError(f"No data returned for ticker {ticker}")
        
        return data

    @staticmethod
    def clean_data(data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        data_reqd = data[stocks.features]

        if data_reqd.isnull().values.any():
            # Null values are dealt using the fill-forward method where the last valid observation is carried forward
            data_reqd.fillna(method='ffill', inplace=True)
        
        return data_reqd
    