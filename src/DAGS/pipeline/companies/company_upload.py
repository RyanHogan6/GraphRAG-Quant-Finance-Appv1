"""
Company Collection Upload Script
Reads SP500_constituents_with_history.csv and optionally RUSSELL2000_constituents.csv and
NASDAQ100_constituents.csv from the DAGS directory, merges by ticker, and uploads to ArangoDB
Company collection with index membership flags: sp500_member, russell2000_member, nasdaq100_member.
"""
import pandas as pd
import os
from pathlib import Path
from arango import ArangoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def _dags_dir():
    """DAGS directory containing constituent CSVs (pipeline's parent)."""
    return Path(__file__).resolve().parent.parent.parent


def get_db():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_HOST', '')
    db_name = os.getenv('ARANGO_DB', 'QUANT_v3')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)


def setup_company_collection(db):
    """Create Company collection if it doesn't exist"""
    if not db.has_collection('Company'):
        db.create_collection('Company')
        print("✓ Created Company collection")
    else:
        print("✓ Company collection exists")


def load_merged_constituents(dags_dir=None):
    """
    Load SP500 CSV and optionally Russell 2000 and NASDAQ-100 CSVs; merge into one
    DataFrame per ticker with columns: ticker, company, sector, industry, cik,
    entry_date, removal_date, sp500_member, russell2000_member, nasdaq100_member.

    Returns:
        pd.DataFrame with one row per ticker (union of all indices).
    """
    if dags_dir is None:
        dags_dir = _dags_dir()
    dags_dir = Path(dags_dir)

    sp500_path = dags_dir / 'SP500_constituents_with_history.csv'
    if not sp500_path.exists():
        raise FileNotFoundError(f"SP500 CSV not found at {sp500_path}")

    df_sp = pd.read_csv(sp500_path)
    df_sp['entry_date'] = pd.to_datetime(df_sp['entry_date'], errors='coerce')
    df_sp['removal_date'] = pd.to_datetime(df_sp['removal_date'], errors='coerce')
    df_sp['sp500_member'] = df_sp['removal_date'].isna()
    df_sp['russell2000_member'] = False
    df_sp['nasdaq100_member'] = False

    ticker_to_row = {}
    for _, row in df_sp.iterrows():
        ticker = str(row['ticker']).strip()
        if not ticker or ticker == 'nan':
            continue
        ticker_to_row[ticker] = {
            'ticker': ticker,
            'company': row.get('company'),
            'sector': row.get('sector'),
            'industry': row.get('industry'),
            'cik': row.get('cik'),
            'entry_date': row['entry_date'],
            'removal_date': row['removal_date'],
            'sp500_member': bool(row['sp500_member']),
            'russell2000_member': False,
            'nasdaq100_member': False,
        }

    russell_path = dags_dir / 'RUSSELL2000_constituents.csv'
    if russell_path.exists():
        df_r = pd.read_csv(russell_path)
        for _, row in df_r.iterrows():
            ticker = str(row['ticker']).strip().replace('.', '-')
            if not ticker or ticker == 'nan':
                continue
            if ticker not in ticker_to_row:
                ticker_to_row[ticker] = {
                    'ticker': ticker,
                    'company': row.get('company'),
                    'sector': None,
                    'industry': None,
                    'cik': None,
                    'entry_date': pd.NaT,
                    'removal_date': pd.NaT,
                    'sp500_member': False,
                    'russell2000_member': True,
                    'nasdaq100_member': False,
                }
            else:
                ticker_to_row[ticker]['russell2000_member'] = True
        print(f"  Loaded Russell 2000: {russell_path} ({len(df_r)} tickers)")
    else:
        print(f"  Skip Russell 2000 (not found): {russell_path}")

    nasdaq_path = dags_dir / 'NASDAQ100_constituents.csv'
    if nasdaq_path.exists():
        df_n = pd.read_csv(nasdaq_path)
        for _, row in df_n.iterrows():
            ticker = str(row['ticker']).strip().replace('.', '-')
            if not ticker or ticker == 'nan':
                continue
            if ticker not in ticker_to_row:
                ticker_to_row[ticker] = {
                    'ticker': ticker,
                    'company': row.get('company'),
                    'sector': None,
                    'industry': None,
                    'cik': None,
                    'entry_date': pd.NaT,
                    'removal_date': pd.NaT,
                    'sp500_member': False,
                    'russell2000_member': False,
                    'nasdaq100_member': True,
                }
            else:
                ticker_to_row[ticker]['nasdaq100_member'] = True
        print(f"  Loaded NASDAQ-100: {nasdaq_path} ({len(df_n)} tickers)")
    else:
        print(f"  Skip NASDAQ-100 (not found): {nasdaq_path}")

    return pd.DataFrame(list(ticker_to_row.values()))


def upload_companies(csv_path=None):
    """
    Load merged constituents (SP500 + optional Russell 2000 + NASDAQ-100) and upload to ArangoDB.

    If csv_path is provided, only that CSV is used (legacy single-CSV behavior with sp500_member only;
    russell2000_member and nasdaq100_member will be false). If csv_path is None, load_merged_constituents()
    is used to merge all CSVs from the DAGS directory.
    """
    print("="*80)
    print("COMPANY COLLECTION UPLOAD")
    print("="*80)

    dags_dir = _dags_dir()
    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"ERROR: CSV not found at {csv_path}")
            exit(1)
        print(f"\nReading single CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
        df['removal_date'] = pd.to_datetime(df['removal_date'], errors='coerce')
        df['sp500_member'] = df['removal_date'].isna()
        df['russell2000_member'] = False
        df['nasdaq100_member'] = False
    else:
        print(f"\nLoading merged constituents from: {dags_dir}")
        df = load_merged_constituents(dags_dir)

    print(f"  Total companies: {len(df)}")
    print(f"  SP500 current:   {df['sp500_member'].sum()}")
    print(f"  Russell 2000:    {df['russell2000_member'].sum()}")
    print(f"  NASDAQ-100:      {df['nasdaq100_member'].sum()}")

    # Connect to database
    print("\nConnecting to ArangoDB...")
    db = get_db()
    setup_company_collection(db)

    company_col = db.collection('Company')

    # Prepare documents
    print("\nPreparing documents...")
    documents = []

    for _, row in df.iterrows():
        entry_date = row['entry_date']
        removal_date = row['removal_date']
        entry_date = entry_date.isoformat() if pd.notna(entry_date) else None
        removal_date = removal_date.isoformat() if pd.notna(removal_date) else None

        indices = []
        if row['sp500_member']:
            indices.append('SP500')
        if row['russell2000_member']:
            indices.append('RUSSELL2000')
        if row['nasdaq100_member']:
            indices.append('NASDAQ100')

        doc = {
            '_key': row['ticker'],
            'ticker': row['ticker'],
            'company': row['company'] if pd.notna(row['company']) else None,
            'sector': row['sector'] if pd.notna(row['sector']) else None,
            'industry': row['industry'] if pd.notna(row['industry']) else None,
            'cik': row['cik'] if pd.notna(row['cik']) else None,
            'entry_date': entry_date,
            'removal_date': removal_date,
            'sp500_member': bool(row['sp500_member']),
            'russell2000_member': bool(row['russell2000_member']),
            'nasdaq100_member': bool(row['nasdaq100_member']),
            'indices': indices,
            'lastUpdated': datetime.now().isoformat()
        }

        documents.append(doc)

    # Upload documents (replace existing)
    print(f"\nUploading {len(documents)} companies...")

    try:
        for i in range(0, len(documents), 100):
            batch = documents[i:i+100]
            for doc in batch:
                try:
                    company_col.insert(doc, overwrite=True)
                except Exception as e:
                    print(f"  Warning: Could not insert {doc['_key']}: {e}")

            if (i + 100) % 500 == 0:
                print(f"  Uploaded {i + 100}/{len(documents)}...")

        print(f"\n✓ Successfully uploaded {len(documents)} companies")

        total_count = company_col.count()
        query_sp = "FOR c IN Company FILTER c.sp500_member == true COLLECT WITH COUNT INTO count RETURN count"
        query_r = "FOR c IN Company FILTER c.russell2000_member == true COLLECT WITH COUNT INTO count RETURN count"
        query_n = "FOR c IN Company FILTER c.nasdaq100_member == true COLLECT WITH COUNT INTO count RETURN count"
        sp_count = list(db.aql.execute(query_sp))[0]
        r_count = list(db.aql.execute(query_r))[0]
        n_count = list(db.aql.execute(query_n))[0]

        print(f"\nVerification:")
        print(f"  Total companies in DB: {total_count}")
        print(f"  SP500 current: {sp_count}")
        print(f"  Russell 2000:  {r_count}")
        print(f"  NASDAQ-100:    {n_count}")

        print(f"\nSample documents:")
        sample_query = """
        FOR c IN Company
            FILTER c.sp500_member == true OR c.nasdaq100_member == true OR c.russell2000_member == true
            LIMIT 5
            RETURN c
        """
        samples = list(db.aql.execute(sample_query))
        for s in samples:
            print(f"  {s['ticker']:6} - {str(s.get('company', ''))[:40]:40} | {s.get('indices', [])}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise

    print("\n" + "="*80)
    print("✓ COMPANY UPLOAD COMPLETE")
    print("="*80)


if __name__ == '__main__':
    dags_dir = _dags_dir()
    sp500_path = dags_dir / 'SP500_constituents_with_history.csv'

    if not sp500_path.exists():
        print(f"ERROR: SP500 CSV not found at {sp500_path}")
        exit(1)

    # Merge all indices from DAGS directory (SP500 + Russell 2000 + NASDAQ-100 if present)
    upload_companies(csv_path=None)
