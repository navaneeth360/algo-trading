# List of stocks in our universe

tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "BRK.B", "TSLA", "V",
    "JNJ", "JPM", "WMT", "UNH", "LLY", "MA", "HD", "PG", "DIS", "BAC",
    "NVDA", "XOM", "KO", "ADBE", "VZ", "PFE", "CSCO", "PEP", "T", "CMCSA",
    "NFLX", "ABT", "CRM", "ABBV", "ORCL", "COST", "ACN", "NKE", "MRK", "INTC",
    "MCD", "TMO", "WFC", "MDT", "TXN", "DHR", "NEE", "LIN", "AMGN", "LOW",
    "BMY", "HON", "QCOM", "SBUX", "UNP", "CVX", "RTX", "PM", "IBM", "MS",
    "GS", "CAT", "MMM", "INTU", "AVY", "GILD", "GE", "BLK", "UPS", "CVS",
    "AXP", "MDLZ", "AMT", "BKNG", "ANTM", "PLD", "ISRG", "SPGI", "DUK", "ADP",
    "BP", "SYK", "DUK", "C", "SO", "ZTS", "CCI", "TGT", "CL", "PNC",
    "CI", "RTX", "GS", "SHW", "RTX", "PNC", "BDX", "GM", "BSX", "MO",
    "FDX", "EL", "DE", "CI", "GPN", "USB", "TFC", "F", "NSC", "VRTX",
    "SYK", "ICE", "AIG", "CLX", "MMC", "LMT", "ELV", "EW", "CAT", "ADSK",
    "EMR", "GE", "ETN", "HAL", "DG", "HCA", "OXY", "BIIB", "ROST", "MU",
    "CSX", "ICE", "PAYX", "MSI", "ZBH", "KMB", "MSFT", "CCL", "ICE", "REGN",
    "KMI", "AEP", "EQIX", "CI", "LRCX", "ICE", "PNW", "ICE", "HUM", "SPG",
    "VRTX", "HSY", "GM", "HPE", "MCO", "PSA", "STZ", "GIS", "ICE", "KHC",
    "ADM", "CCI", "ICE", "CTSH", "ROK", "ALL", "ROP", "AMD", "MSFT", "ICE",
    "DLR", "EXC", "LHX", "ICE", "FLT", "ICE", "AFL", "ICE", "WELL", "ICE",
    "PH", "ICE", "ICE"
]

# Desired columns for stock data
features = ["Open", "High", "Low", "Close", "Volume"]

# Defining the period for testing our strategies 
test_period = 40

# Defining MA startegy periods
ma_slow = 10
ma_fast = 2 

# Defining ML periods and thresholds
ml_lookback = 10
ml_threshold = 0.5
ml_train_period = 400

# Defining the RSI period and threshold
rsi_period = 4
rsi_high_threshold = 70
rsi_low_threshold = 30

# Defining the Bollinger band moving window and k
bol_window = 20
k = 2

# Define the maximum position we're willing to take in a single trade (can be adjusted to be dynamic based on risk)
max_p = 1
