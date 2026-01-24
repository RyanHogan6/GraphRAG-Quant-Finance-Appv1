"""
USASpending Awards Downloader - Incremental Updates
Fetches government contract data for S&P 500 companies
"""
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_sp500_companies():
    """Fetch current S&P 500 companies from Wikipedia"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return list(zip(df['Symbol'], df['Security']))
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return []

def query_usaspending(company_name, start_date, end_date, max_retries=3):
    """
    Query USASpending API with pagination
    Returns list of contract records
    """
    all_records = []
    page = 1
    limit = 100

    while True:
        payload = {
            "filters": {
                "recipient_search_text": [company_name],
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [{
                    "start_date": start_date,
                    "end_date": end_date,
                    "date_type": "action_date"
                }]
            },
            "fields": [
                "Award ID", "Recipient Name", "Start Date", "End Date",
                "Award Amount", "Awarding Agency", "Award Type", "Description",
                "NAICS Code", "Product or Service Code", "Place of Performance",
                "Base and All Options Value", "generated_internal_id"
            ],
            "limit": limit,
            "page": page
        }

        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(
                    "https://api.usaspending.gov/api/v2/search/spending_by_award/",
                    json=payload,
                    timeout=30
                )

                if res.status_code == 429:
                    retry_after = int(res.headers.get('Retry-After', 60))
                    print(f"  Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if res.ok:
                    data = res.json()
                    records = data.get("results", [])
                    all_records.extend(records)

                    if len(records) < limit:
                        return all_records

                    page += 1
                    time.sleep(0.5)
                    break
                else:
                    if attempt == max_retries:
                        return all_records
                    time.sleep(2 ** attempt)

            except Exception as e:
                if attempt >= max_retries:
                    return all_records
                time.sleep(3 * attempt)

        if len(records) < limit:
            break

    return all_records

def fetch_recent_awards(days_back=1):
    """
    Fetch awards from last N days for all S&P 500 companies
    Returns DataFrame with all new contracts
    """
    companies = get_sp500_companies()
    if not companies:
        return pd.DataFrame()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    print(f"Fetching awards from {start_date} to {end_date}")
    print(f"Querying {len(companies)} companies...")

    all_results = []

    for idx, (ticker, company_name) in enumerate(companies, 1):
        records = query_usaspending(company_name, start_date.isoformat(), end_date.isoformat())

        for r in records:
            r['Ticker'] = ticker
            r['SP500_Company'] = company_name

        all_results.extend(records)

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(companies)} - {len(all_results)} awards found")

    df = pd.DataFrame(all_results)
    print(f"✓ Fetched {len(df)} new awards")

    return df
