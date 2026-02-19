import pandas as pd, os
import io
import urllib.request
from pathlib import Path
from .config import *

# Wikipedia returns 403 without a browser User-Agent
WIKI_UA = "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"


def _fetch_wiki_html(url):
    """Fetch Wikipedia page HTML with a browser User-Agent to avoid 403."""
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _dags_dir():
    """DAGS directory (parent of pipeline/) where constituent CSVs live."""
    return Path(__file__).resolve().parent.parent.parent


def build_nasdaq100_constituents():
    """
    Scrape Wikipedia 'Nasdaq-100' page for Current components table; save
    NASDAQ100_constituents.csv to DAGS directory (ticker, company).
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _fetch_wiki_html(url)
    tables = pd.read_html(io.StringIO(html))
    # Find table with Ticker and Company columns (Current components)
    for df in tables:
        cols = [c for c in df.columns if isinstance(c, str)]
        if 'Ticker' in cols and 'Company' in cols:
            out = df[['Ticker', 'Company']].copy()
            out.columns = ['ticker', 'company']
            out['ticker'] = out['ticker'].astype(str).str.strip().str.replace('.', '-', regex=False)
            out['company'] = out['company'].astype(str).str.strip()
            out = out[out['ticker'].str.len() > 0].drop_duplicates(subset=['ticker'])
            path = _dags_dir() / 'NASDAQ100_constituents.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            print(f"Saved {len(out)} NASDAQ-100 constituents to {path}")
            return
    print("Could not find Ticker/Company table on Wikipedia Nasdaq-100 page")


def build_russell2000_constituents():
    """
    Scrape Wikipedia 'Russell_2000_Index' for Example Members table; save
    RUSSELL2000_constituents.csv. Note: Wikipedia only lists ~11 example members.
    For a full ~2000 list, use iShares IWM holdings or another source and
    save RUSSELL2000_constituents.csv (ticker, company) in src/DAGS/ manually.
    """
    url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
    html = _fetch_wiki_html(url)
    tables = pd.read_html(io.StringIO(html))
    for df in tables:
        cols = [c for c in df.columns if isinstance(c, str)]
        # Example table has Company, Symbol
        if 'Symbol' in cols and 'Company' in cols:
            out = df[['Symbol', 'Company']].copy()
            out.columns = ['ticker', 'company']
            out['ticker'] = out['ticker'].astype(str).str.strip().str.replace('.', '-', regex=False)
            out['company'] = out['company'].astype(str).str.strip()
            out = out[out['ticker'].str.len() > 0].drop_duplicates(subset=['ticker'])
            path = _dags_dir() / 'RUSSELL2000_constituents.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            print(f"Saved {len(out)} Russell 2000 example constituents to {path}")
            print("  (Wikipedia only has example members; for full list use IWM holdings or paste CSV.)")
            return
    print("Could not find Symbol/Company table on Wikipedia Russell 2000 page")


def build_sp500_constituents_history():
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(WIKI_URL, header=[0, 1])
    constituents = tables[0]
    constituents.columns = [col[0] if isinstance(col, tuple) else col for col in constituents.columns]
    constituents = constituents.rename(columns={
        'Symbol': 'ticker', 'Security': 'company', 'GICS Sector': 'sector', 'GICS Sub-Industry': 'industry',
        'Date added': 'entry_date', 'CIK': 'cik'
    })
    constituents['ticker'] = constituents['ticker'].str.replace('.', '-', regex=False)
    constituents['entry_date'] = pd.to_datetime(constituents['entry_date'], errors='coerce')
    constituents['cik'] = constituents['cik'].astype(str).str.zfill(10)
    changes = tables[1]
    rename_cols = {
        ('Date', 'Date'): 'date', ('Added', 'Ticker'): 'added', ('Added', 'Security'): 'added_company',
        ('Removed', 'Ticker'): 'removed', ('Removed', 'Security'): 'removed_company',
        ('Reason', 'Reason'): 'reason'
    }
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = [rename_cols.get(col, col[1] if isinstance(col, tuple) else col) for col in changes.columns]
    else:
        changes = changes.rename(columns=rename_cols)
    changes.columns = [c if isinstance(c, str) else c[0] for c in changes.columns]

    def split_and_strip(x):
        if pd.isnull(x):
            return []
        return [str(a).strip() for a in str(x).replace('\n', ',').replace(' and ', ',').split(',') if a.strip()]
    records = []
    for _, row in changes.iterrows():
        dt = pd.to_datetime(row['date'], errors='coerce')
        for t in split_and_strip(row['added']):
            records.append({'ticker': t.replace('.', '-'), 'change_date': dt, 'action': 'add'})
        for t in split_and_strip(row['removed']):
            records.append({'ticker': t.replace('.', '-'), 'change_date': dt, 'action': 'remove'})
    history = pd.DataFrame(records)

    all_tickers = pd.unique(list(constituents['ticker']) + list(history['ticker']))
    rows = []
    for ticker in all_tickers:
        entry = history[(history['ticker'] == ticker) & (history['action'] == 'add')]['change_date']
        removal = history[(history['ticker'] == ticker) & (history['action'] == 'remove')]['change_date']
        entry_date = entry.min() if not entry.empty else None
        removal_date = removal.min() if not removal.empty else None

        meta = constituents[constituents['ticker'] == ticker]
        meta = meta.iloc[0] if len(meta) else {}
        rows.append({
            'ticker': ticker,
            'company': meta['company'] if 'company' in meta else None,
            'sector': meta['sector'] if 'sector' in meta else None,
            'industry': meta['industry'] if 'industry' in meta else None,
            'cik': meta['cik'] if 'cik' in meta else None,
            'entry_date': entry_date if pd.notnull(entry_date) else meta['entry_date'] if 'entry_date' in meta else None,
            'removal_date': removal_date if pd.notnull(removal_date) else None
        })

    combined = pd.DataFrame(rows)
    combined['entry_date'] = pd.to_datetime(combined['entry_date'], errors='coerce')
    combined['removal_date'] = pd.to_datetime(combined['removal_date'], errors='coerce')
    combined = combined.sort_values(['entry_date', 'removal_date', 'ticker']).reset_index(drop=True)

    os.makedirs(os.path.dirname(CONSTITUENTS_PATH), exist_ok=True)
    combined.to_csv(CONSTITUENTS_PATH, index=False)
    print(f"Saved full S&P 500 lifecycle to {CONSTITUENTS_PATH}")

def get_sp500_tickers(current_only=True):
    """
    Get S&P 500 tickers from CSV file

    Args:
        current_only: If True, only return current members (no removal_date)
                     If False, return all historical members (852 tickers)
    """
    import os

    # Use the CSV file in DAGS directory (relative to pipeline/yahoo/)
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'SP500_constituents_with_history.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SP500 CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if current_only:
        # Filter to only current members (removal_date is null/empty)
        df = df[df['removal_date'].isna() | (df['removal_date'] == '')]
        print(f"Loaded {len(df)} CURRENT S&P 500 tickers")
    else:
        print(f"Loaded {len(df)} tickers (including historical)")

    # Get tickers, remove NaN values
    tickers = df['ticker'].dropna().unique().tolist()

    return tickers

def get_tickers_from_arango(current_only=True, index=None):
    """
    Get tickers directly from ArangoDB Company collection.

    Args:
        current_only: If True (default), return only current index members. When index is None,
                      "current" = member of at least one index (sp500_member OR russell2000_member OR nasdaq100_member).
                      If False, return all companies including historical.
        index: Optional. If set to "SP500", "RUSSELL2000", or "NASDAQ100", filter to that index only.
               Ignored when current_only is False.

    Returns:
        List of ticker strings.
    """
    from arango import ArangoClient
    from dotenv import load_dotenv
    import os

    load_dotenv()

    arango_url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
    db_name = os.getenv('ARANGO_DB', 'QUANT_v3')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')

    client = ArangoClient(hosts=arango_url)
    db = client.db(db_name, username=username, password=password)

    if not current_only:
        query = """
        FOR company IN Company
            RETURN company.ticker
        """
        print("Fetching ALL companies (including historical)")
    elif index == "SP500":
        query = """
        FOR company IN Company
            FILTER company.sp500_member == true
            RETURN company.ticker
        """
        print("Fetching CURRENT S&P 500 members only (sp500_member=true)")
    elif index == "RUSSELL2000":
        query = """
        FOR company IN Company
            FILTER company.russell2000_member == true
            RETURN company.ticker
        """
        print("Fetching Russell 2000 members only")
    elif index == "NASDAQ100":
        query = """
        FOR company IN Company
            FILTER company.nasdaq100_member == true
            RETURN company.ticker
        """
        print("Fetching NASDAQ-100 members only")
    else:
        # current_only=True, index=None: any current index member
        query = """
        FOR company IN Company
            FILTER company.sp500_member == true
                OR company.russell2000_member == true
                OR company.nasdaq100_member == true
            RETURN company.ticker
        """
        print("Fetching current members of any index (SP500, Russell 2000, NASDAQ-100)")

    tickers = list(db.aql.execute(query))

    return [t for t in tickers if t]  # Filter out None values
