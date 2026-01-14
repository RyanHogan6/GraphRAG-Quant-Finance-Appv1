"""
Yahoo Finance MarketData ETL Pipeline
Fetches S&P 500 stock prices + technical indicators daily
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from pipeline.yahoo.constituents import get_sp500_tickers
from pipeline.yahoo.downloader import download_stock_data
from pipeline.yahoo.features import engineer_technical_features
from pipeline.yahoo.panel_merge import merge_to_panel

# Add arango uploader
from pipeline.polymarket.arango_uploader import get_arango_connection

def task_fetch_tickers(**context):
    """Get S&P 500 ticker list"""
    tickers = get_sp500_tickers()
    context['ti'].xcom_push(key='tickers', value=tickers)
    print(f"[OK] Fetched {len(tickers)} S&P 500 tickers")

def task_download_data(**context):
    """Download OHLCV data for all tickers"""
    tickers = context['ti'].xcom_pull(key='tickers', task_ids='fetch_tickers')

    # Download last 30 days
    data_df = download_stock_data(tickers, period='1mo')
    context['ti'].xcom_push(key='raw_data', value=data_df.to_dict('records'))
    print(f"[OK] Downloaded {len(data_df)} rows")

def task_engineer_features(**context):
    """Calculate technical indicators"""
    raw_data = context['ti'].xcom_pull(key='raw_data', task_ids='download_data')
    df = pd.DataFrame(raw_data)

    featured_df = engineer_technical_features(df)
    context['ti'].xcom_push(key='featured_data', value=featured_df.to_dict('records'))
    print(f"[OK] Engineered features for {len(featured_df)} rows")

def task_upload_to_arango(**context):
    """Upsert to MarketData collection"""
    featured_data = context['ti'].xcom_pull(key='featured_data', task_ids='engineer_features')
    df = pd.DataFrame(featured_data)

    db = get_arango_connection()
    collection = db.collection('MarketData')

    # Batch upsert
    docs = []
    for _, row in df.iterrows():
        doc = {
            '_key': f"{row['ticker']}_{row['date']}",
            'ticker': row['ticker'],
            'date': row['date'],
            **row.to_dict()
        }
        docs.append(doc)

    # Upsert in batches
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        query = f"""
        FOR doc IN @documents
            UPSERT {{ _key: doc._key }}
            INSERT doc
            UPDATE doc
            IN MarketData
        """
        db.aql.execute(query, bind_vars={'documents': batch})

    print(f"[OK] Upserted {len(docs)} market data records")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 14),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='yahoo_marketdata_etl',
    default_args=default_args,
    description='Daily S&P 500 stock data ingestion',
    schedule=timedelta(days=1),  # Daily at midnight
    catchup=False,
    tags=['yahoo', 'stocks', 'marketdata'],
) as dag:

    fetch_tickers = PythonOperator(
        task_id='fetch_tickers',
        python_callable=task_fetch_tickers,
    )

    download_data = PythonOperator(
        task_id='download_data',
        python_callable=task_download_data,
    )

    engineer_features = PythonOperator(
        task_id='engineer_features',
        python_callable=task_engineer_features,
    )

    upload = PythonOperator(
        task_id='upload_to_arango',
        python_callable=task_upload_to_arango,
    )

    fetch_tickers >> download_data >> engineer_features >> upload
