"""
Company Collection Upload Script
Reads SP500_constituents_with_history.csv and uploads to ArangoDB Company collection
"""
import pandas as pd
import os
from pathlib import Path
from arango import ArangoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


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


def upload_companies(csv_path):
    """
    Upload companies from CSV to ArangoDB

    CSV structure:
    - ticker: Stock ticker symbol (_key)
    - company: Full company name
    - sector: GICS sector
    - industry: GICS sub-industry
    - cik: SEC CIK number
    - entry_date: Date added to S&P 500
    - removal_date: Date removed from S&P 500 (null if still active)
    """
    print("="*80)
    print("COMPANY COLLECTION UPLOAD")
    print("="*80)

    # Read CSV
    print(f"\nReading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  Total companies: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Clean data
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['removal_date'] = pd.to_datetime(df['removal_date'], errors='coerce')

    # Determine current S&P 500 members
    df['sp500_member'] = df['removal_date'].isna()

    current_members = df[df['sp500_member']].shape[0]
    historical = df[~df['sp500_member']].shape[0]

    print(f"\n  Current S&P 500 members: {current_members}")
    print(f"  Historical members: {historical}")

    # Connect to database
    print("\nConnecting to ArangoDB...")
    db = get_db()
    setup_company_collection(db)

    company_col = db.collection('Company')

    # Prepare documents
    print("\nPreparing documents...")
    documents = []

    for _, row in df.iterrows():
        # Convert timestamps to ISO strings
        entry_date = row['entry_date'].isoformat() if pd.notna(row['entry_date']) else None
        removal_date = row['removal_date'].isoformat() if pd.notna(row['removal_date']) else None

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
            'lastUpdated': datetime.now().isoformat()
        }

        documents.append(doc)

    # Upload documents (replace existing)
    print(f"\nUploading {len(documents)} companies...")

    try:
        # Use upsert to replace existing documents
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

        # Verify counts
        total_count = company_col.count()

        # Query for current S&P 500 members
        query = """
        FOR c IN Company
            FILTER c.sp500_member == true
            COLLECT WITH COUNT INTO count
            RETURN count
        """
        current_count = list(db.aql.execute(query))[0]

        print(f"\nVerification:")
        print(f"  Total companies in DB: {total_count}")
        print(f"  Current S&P 500 members: {current_count}")

        # Show sample
        print(f"\nSample documents:")
        sample_query = """
        FOR c IN Company
            FILTER c.sp500_member == true
            LIMIT 3
            RETURN c
        """
        samples = list(db.aql.execute(sample_query))
        for s in samples:
            print(f"  {s['ticker']:6} - {s['company']:40} | {s['sector']:25} | {s['industry']}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise

    print("\n" + "="*80)
    print("✓ COMPANY UPLOAD COMPLETE")
    print("="*80)


if __name__ == '__main__':
    # CSV path relative to this script
    csv_path = Path(__file__).parent.parent / 'SP500_constituents_with_history.csv'

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        exit(1)

    upload_companies(csv_path)
