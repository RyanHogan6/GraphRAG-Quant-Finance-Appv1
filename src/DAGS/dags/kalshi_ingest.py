"""
Kalshi ETL Pipeline
Fetches prediction markets, generates embeddings, uploads to ArangoDB
Runs every 30 minutes
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from pipeline.kalshi.downloader import fetch_all_markets
from pipeline.kalshi.features import engineer_market_features
from pipeline.kalshi.arango_uploader import get_arango_connection, upsert_markets

def task_fetch_markets(**context):
    """Fetch Kalshi markets"""
    print("\n[TASK 1/3] FETCHING KALSHI MARKETS")
    markets_df = fetch_all_markets()
    context['ti'].xcom_push(key='markets_df', value=markets_df.to_dict('records'))
    print(f"[OK] Fetched {len(markets_df)} markets")

def task_engineer_features(**context):
    """Engineer features + embeddings"""
    print("\n[TASK 2/3] ENGINEERING FEATURES")
    markets_dict = context['ti'].xcom_pull(key='markets_df', task_ids='fetch_markets')
    markets_df = pd.DataFrame(markets_dict)

    markets_df = engineer_market_features(markets_df)
    context['ti'].xcom_push(key='markets_df_eng', value=markets_df.to_dict('records'))
    print(f"[OK] Engineered features for {len(markets_df)} markets")

def task_upload_to_arango(**context):
    """Upload to ArangoDB"""
    print("\n[TASK 3/3] UPLOADING TO ARANGODB")
    markets_dict = context['ti'].xcom_pull(key='markets_df_eng', task_ids='engineer_features')
    markets_df = pd.DataFrame(markets_dict)

    db = get_arango_connection()
    inserted, updated, errors = upsert_markets(db, markets_df)

    print(f"[OK] Inserted: {inserted}, Updated: {updated}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 14),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='kalshi_etl_pipeline',
    default_args=default_args,
    description='Kalshi prediction markets with embeddings',
    schedule=timedelta(minutes=30),
    catchup=False,
    tags=['kalshi', 'prediction-markets', 'embeddings'],
) as dag:

    fetch_markets = PythonOperator(
        task_id='fetch_markets',
        python_callable=task_fetch_markets,
    )

    engineer_features = PythonOperator(
        task_id='engineer_features',
        python_callable=task_engineer_features,
    )

    upload = PythonOperator(
        task_id='upload_to_arango',
        python_callable=task_upload_to_arango,
    )

    fetch_markets >> engineer_features >> upload
