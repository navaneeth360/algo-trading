# Algorithmic Trading Strategies in Python

This repository implements and backtests multiple algorithmic trading strategies on historical stock data using the **[yfinance](https://pypi.org/project/yfinance/)** API.  
The framework is modular, making it easy to add new strategies, evaluate performance, and visualize results.  

## Features

- **Implemented Strategies**  
  - Simple Moving Average (SMA)  
  - Exponential Moving Average (EMA)  
  - Bollinger Bands  
  - Relative Strength Index (RSI)  
  - Linear Regression  
  - Long Short-Term Memory (LSTM) neural network  
  - Buy & Hold (baseline)  

- **Pipeline**  
  1. Data retrieval from Yahoo Finance using `yfinance`  
  2. Preprocessing module for cleaning and preparing financial time series  
  3. Strategy implementation and signal generation  
  4. Performance evaluation with key metrics  
  5. Visualization of cumulative PnL, stock price, and indicators  

- **Performance Metrics**  
  - Sharpe Ratio  
  - Number of trades  
  - Profit per trade  
  - Strategy return vs. Buy & Hold baseline  

- **Visualizations**  
  - Cumulative profit and loss (PnL)  
  - Stock price with strategy signals  
  - Technical indicators (moving averages, RSI, Bollinger bands, etc.)  

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/algotrading-strategies.git
cd algotrading-strategies
pip install -r requirements.txt
```

N V Navaneeth Rajesh
(22b1215)

