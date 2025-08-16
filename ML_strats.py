import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import stocks
from framework import Strategy

class Lin_Reg(Strategy):

    def __init__(self, train_test_data, lookback=30, threshold=0.5, test_period=90, max_p=1):
        self.train_test_data = train_test_data
        self.lookback = lookback
        self.threshold = threshold
        self.test_period = test_period
        self.max_p = max_p

    def implement_strat(self):

        close_prices = self.train_test_data.iloc[:, 3].to_numpy()
        volumes = self.train_test_data.iloc[:, 4].to_numpy()
        y = close_prices[self.lookback:]

        X = []
        for i in range(self.lookback, len(close_prices)):
            close_p = close_prices[i-self.lookback:i]
            vols = volumes[i-self.lookback:i]
            features = np.concatenate([close_p, vols])
            X.append(features)

        X = np.array(X)
        print(X[-5:])
        print(X.shape)
        print(len(y))
        print(y[-5:])

        ind = len(y) - self.test_period
        X_train, X_test = X[:ind], X[ind:]
        y_train, y_test = y[:ind], y[ind:]


        # The initial features are the close price and the volume 
        # close_prices = self.train_test_data.iloc[:, 3].to_numpy()
        # close_prices_train = close_prices[:-(self.test_period+1)]
        # close_prices_test = close_prices[-(self.test_period+1):-1]

        # volumes = self.train_test_data.iloc[:, 4].to_numpy()
        # volumes_train = volumes[:-(self.test_period+1)]
        # volumes_test = volumes[-(self.test_period+1):-1]

        # y_train = close_prices[1:-self.test_period]
        # y_test = close_prices[-self.test_period:]
        # # close_prices_test = close_prices.iloc[-self.test_period:].to_numpy()

        # X_train = np.column_stack((close_prices_train, volumes_train))
        # print(X_train)
        # X_test = np.column_stack((close_prices_test, volumes_test))

        # print(X_train.shape, X_test.shape, len(y_train), len(y_test))

        model = LinearRegression()
        model.fit(X_train, y_train)

        print(model.coef_, model.score(X_train,y_train))
        
        predictions = model.predict(X_test)
        # print(predictions)
        # print(y_test)

        predictions = np.array(predictions)
        signal = predictions - y_test
        
        curr_pos = [0] * (self.test_period + 1)
        balance = [0] * (self.test_period + 1)
        pnl = [0] * (self.test_period + 1)
        num_trades = 0

        for i in range(len(signal)):
            if signal[i] > self.threshold:
                # Go long or switch from short to long
                if curr_pos[i] <= 0:
                    qty_bought = self.max_p + (-1 * curr_pos[i])
                    num_trades += qty_bought
                    amt_spent = -1 * (qty_bought * y_test[i])
                    curr_pos[i + 1] = curr_pos[i] + qty_bought
                    balance[i + 1] = balance[i] + amt_spent
                    pnl[i + 1] = pnl[i] + (balance[i] - (-1 * curr_pos[i]) * y_test[i])
                else:
                    curr_pos[i + 1] = curr_pos[i]
                    balance[i + 1] = balance[i]
                    pnl[i + 1] = pnl[i]

            elif signal[i] < -self.threshold:
                # Go short or switch from long to short
                if curr_pos[i] >= 0:
                    qty_sold = self.max_p + curr_pos[i]
                    num_trades += qty_sold
                    amt_received = qty_sold * y_test[i]
                    curr_pos[i + 1] = curr_pos[i] - qty_sold
                    balance[i + 1] = balance[i] + amt_received
                    pnl[i + 1] = pnl[i] + (balance[i] + curr_pos[i] * y_test[i])
                else:
                    curr_pos[i + 1] = curr_pos[i]
                    balance[i + 1] = balance[i]
                    pnl[i + 1] = pnl[i]
            else:
                # Hold position
                curr_pos[i + 1] = curr_pos[i]
                balance[i + 1] = balance[i]
                pnl[i + 1] = pnl[i]

        # Neutralize position on last day
        if curr_pos[-1] != 0:
            qty_ = (-1) * curr_pos[-1]
            num_trades += qty_
            amt_ = y_test[-1] * (-1) * qty_
            curr_pos[-1] += qty_
            balance[-1] += amt_
            pnl[-1] += amt_

        balance = balance[1:]
        curr_pos = curr_pos[1:]
        pnl = pnl[1:]


        portfolio_value = np.array(curr_pos) * y_test

        self.balance = balance
        self.positions = curr_pos
        self.predictions = predictions
        self.close_prices = y_test
        self.signal = signal
        self.pnl = pnl
        self.num_trades = num_trades
        self.portfolio_value = portfolio_value

        return {
            "positions": curr_pos,
            "balance": balance,
            "predictions": predictions,
            "signal": signal,
            "close_prices": y_test,
            "pnl": pnl,
            "num_trades": num_trades,
            "portfolio_value": portfolio_value
        }

    def plotter(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Price + Predictions
        axs[0].plot(self.close_prices, label="Close Price", color="blue", linewidth=1)
        axs[0].plot(self.predictions, label="Predicted Price", color="orange", linestyle="--", linewidth=1)
        axs[0].set_title("Price and Linear Regression Predictions")
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="upper left")
        axs[0].grid(True, alpha=0.3)

        # PnL
        axs[1].plot(self.pnl, label="PnL", color="purple", linewidth=1)
        axs[1].set_title("Strategy PnL")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("PnL")
        axs[1].legend(loc="upper left")
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
