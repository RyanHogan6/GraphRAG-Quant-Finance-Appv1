import os, pandas as pd, numpy as np
# import ta
from sklearn.preprocessing import StandardScaler
from .config import *

def load_all_data(path=DATA_RAW_PATH):
    data = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            
            ticker = file.replace(".csv", "")
            print(f'processing ticker: {ticker}')
            df = pd.read_csv(os.path.join(path, file))
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df.sort_values('date', inplace=True)
            data[ticker] = df
    return data

def add_ml_features(df, lookbacks=[5,10,20,50,100], horizons=[1,5,10]):
    # -- paste your existing TA code here --
    ## ... left out for brevity ...
    return df

def process_all():
    os.makedirs(DATA_CLEANED_PATH, exist_ok=True)
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    all_data = load_all_data()
    dfs = []
    for ticker, df in all_data.items():
        try:
            df_feat = add_ml_features(df)
            if df_feat.empty:
                continue
            df_feat['ticker'] = ticker
            ordered = ['date', 'ticker'] + [col for col in df_feat.columns if col not in ['date','ticker']]
            df_feat = df_feat[ordered]
            df_feat.to_csv(os.path.join(DATA_PROCESSED_PATH, f"{ticker}_features.csv"), index=False)
            dfs.append(df_feat)
            print(f"Processed {ticker}")
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
    if dfs:
        combined = pd.concat(dfs).sort_values(['ticker','date'])
        combined.to_csv(os.path.join(DATA_CLEANED_PATH, "cleaned_features.csv"), index=False)
        print("Combined dataset saved.")
