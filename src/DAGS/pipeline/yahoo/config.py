import os

DATA_BASE = "/opt/airflow/data/yahoo/"
DATA_RAW_PATH = os.path.join(DATA_BASE, "raw")
DATA_PROCESSED_PATH = os.path.join(DATA_BASE, "processed")
DATA_CLEANED_PATH = os.path.join(DATA_BASE, "cleaned")
CONSTITUENTS_PATH = os.path.join(DATA_BASE, "sp500_constituents_with_history.csv")
MASTER_PANEL_PATH = os.path.join(DATA_CLEANED_PATH, "feature_panel_with_sp500_membership.csv")
SP500_ONLY_PATH = os.path.join(DATA_CLEANED_PATH, "sp500_only_feature_panel.csv")

YEARS = 10
INTERVAL = "1d"
BATCH_SIZE = 50
MIN_REQUIRED_ROWS = 300

REVISED_STATIC_COLS = [
    # Valuation
    'targetMeanPrice', 'targetHighPrice', 'targetLowPrice', 'targetMedianPrice',
    'forwardEps', 'trailingEps', 'forwardPE', 'trailingPE',
    'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToRevenue', 'enterpriseToEbitda',

    # Growth
    'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth', 'revenuePerShare',

    # Profitability
    'grossMargins', 'ebitdaMargins', 'operatingMargins', 'profitMargins',
    'returnOnEquity', 'returnOnAssets',

    # Financial Health
    'debtToEquity', 'currentRatio', 'quickRatio',
    'totalCash', 'totalDebt', 'freeCashflow', 'operatingCashflow',

    # Dividends
    'dividendRate', 'dividendYield', 'payoutRatio', 'fiveYearAvgDividendYield',

    # Analyst
    'recommendationKey', 'numberOfAnalystOpinions',

    # Other
    'sharesOutstanding', 'marketCap', 'sector', 'industry', 'beta',
    'fiftyDayAverage', 'twoHundredDayAverage', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'
]
