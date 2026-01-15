import pandas as pd, os
from .config import *

def merge_membership_and_panel():
    # Load input files
    panel = pd.read_csv(os.path.join(DATA_CLEANED_PATH, "cleaned_features.csv"), parse_dates=['date'])
    constituents = pd.read_csv(CONSTITUENTS_PATH, parse_dates=['entry_date', 'removal_date'])

    # Merge metadata
    panel = panel.merge(constituents[['ticker', 'entry_date', 'removal_date', 'sector', 'industry', 'cik']],
                        on='ticker', how='left')

    # Tag index membership
    panel['in_index'] = (
        ((panel['entry_date'].isna()) | (panel['date'] >= panel['entry_date'])) &
        ((panel['removal_date'].isna()) | (panel['date'] < panel['removal_date']))
    ).astype(int)

    required_cols = ['open', 'close', 'high', 'low', 'volume']
    panel.dropna(subset=required_cols, inplace=True)

    os.makedirs(DATA_CLEANED_PATH, exist_ok=True)
    panel.to_csv(MASTER_PANEL_PATH, index=False)
    panel[panel['in_index']==1].to_csv(SP500_ONLY_PATH, index=False)
    print(f"Saved enhanced panel: {MASTER_PANEL_PATH} and S&P500-only: {SP500_ONLY_PATH}")
