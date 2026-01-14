"""
Polymarket ETL Pipeline - Airflow DAG
Orchestrates data fetching, feature engineering, and ArangoDB ingestion
Runs every 10 minutes for real-time market intelligence

Pipeline Flow:
1. Fetch markets from Gamma API
2. Fetch traders from subgraph API (skipped on most runs)
3. Engineer features (markets + traders)
4. Upload to ArangoDB with upsert logic
5. Save price history snapshots
6. Build market→company graph edges (every 6 hours only)

Schedule: Every 10 minutes
Catchup: False (only run for current period)
"""

import sys
import os

# Add pipeline directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from pipeline.polymarket.downloader import fetch_all_markets, fetch_current_prices
from pipeline.polymarket.trader_tracker import fetch_traders_from_subgraph, parse_trader_positions
from pipeline.polymarket.features import engineer_market_features, engineer_trader_features
from pipeline.polymarket.arango_uploader import (
    get_arango_connection,
    upsert_markets,
    upsert_traders
)
from pipeline.polymarket.price_history import save_price_snapshots, cleanup_old_snapshots
from pipeline.polymarket.edge_builder import (
    build_all_edges,
    build_trader_position_edges,
    build_position_market_edges
)

# ============================================================================
# TASK FUNCTIONS (Called by Airflow)
# ============================================================================

def task_fetch_markets(**context):
    """
    Task 1: Fetch all markets and current prices from Polymarket.
    Stores results in XCom for downstream tasks.
    """
    print("\n" + "="*80)
    print("TASK 1/5: FETCHING MARKETS")
    print("="*80)

    # Fetch all markets
    markets_df = fetch_all_markets()

    # Fetch prices for top markets
    if len(markets_df) > 0:
        prices_df = fetch_current_prices(markets_df, top_n=200)

        # Merge prices into markets (update yes_probability with latest)
        if len(prices_df) > 0:
            # Update yes_probability from price snapshot
            price_map = dict(zip(prices_df['market_id'], prices_df['yes_price']))
            markets_df['yes_probability'] = markets_df['market_id'].map(price_map).fillna(markets_df['yes_probability'])

    # Push to XCom (convert to dict for JSON serialization)
    context['ti'].xcom_push(key='markets_df', value=markets_df.to_dict('records'))

    print(f"\n[OK] Task 1 complete: Fetched {len(markets_df):,} markets")


def task_fetch_traders(**context):
    """
    Task 2: Fetch trader data from Polymarket subgraph.
    Requires markets from Task 1 to map positions.

    NOTE: This task is SKIPPED on most runs (traders don't change every 10 min).
    Only runs every 36th execution (~6 hours with 10-min schedule).
    """
    print("\n" + "="*80)
    print("TASK 2/6: FETCHING TRADERS")
    print("="*80)

    # Skip trader fetching on most runs (expensive and slow-changing data)
    execution_date = context['execution_date']
    hour = execution_date.hour
    minute = execution_date.minute

    # Only run at :00 minute of every 6th hour (0:00, 6:00, 12:00, 18:00)
    should_fetch_traders = (minute == 0 and hour % 6 == 0)

    if not should_fetch_traders:
        print("  ⏭️ Skipping trader fetch (only runs every 6 hours)")
        context['ti'].xcom_push(key='traders_df', value=[])
        context['ti'].xcom_push(key='positions_df', value=[])
        return

    print("  ✅ Running full trader fetch (6-hour cycle)")

    # Pull markets from Task 1
    markets_dict = context['ti'].xcom_pull(key='markets_df', task_ids='fetch_markets')

    if not markets_dict:
        print("  ⚠️ No markets found from Task 1 - skipping traders")
        context['ti'].xcom_push(key='traders_df', value=[])
        context['ti'].xcom_push(key='positions_df', value=[])
        return

    # Build condition_id → market _id mapping
    market_condition_map = {}
    for m in markets_dict:
        condition_id = m.get('condition_id')
        market_id = m.get('market_id')
        if condition_id and market_id:
            # Use format expected by edge builder
            market_condition_map[condition_id] = f"prediction_markets_polymarket/{market_id}"

    print(f"  Built condition map: {len(market_condition_map)} markets")

    # Fetch traders from subgraph
    traders_raw = fetch_traders_from_subgraph(min_volume=1000)

    # Parse into DataFrames
    if traders_raw:
        traders_df, positions_df = parse_trader_positions(traders_raw, market_condition_map)

        # Push to XCom
        context['ti'].xcom_push(key='traders_df', value=traders_df.to_dict('records'))
        context['ti'].xcom_push(key='positions_df', value=positions_df.to_dict('records'))

        print(f"\n[OK] Task 2 complete: Fetched {len(traders_df):,} traders, {len(positions_df):,} positions")
    else:
        print("  ⚠️ No traders fetched from subgraph")
        context['ti'].xcom_push(key='traders_df', value=[])
        context['ti'].xcom_push(key='positions_df', value=[])


def task_engineer_features(**context):
    """
    Task 3: Engineer features for markets and traders.
    Adds derived metrics for better analysis.
    """
    print("\n" + "="*80)
    print("TASK 3/5: ENGINEERING FEATURES")
    print("="*80)

    # Pull data from previous tasks
    markets_dict = context['ti'].xcom_pull(key='markets_df', task_ids='fetch_markets')
    traders_dict = context['ti'].xcom_pull(key='traders_df', task_ids='fetch_traders')

    # Convert back to DataFrames
    markets_df = pd.DataFrame(markets_dict)

    # Engineer market features
    markets_df = engineer_market_features(markets_df)

    # Engineer trader features (if available)
    if traders_dict:
        traders_df = pd.DataFrame(traders_dict)
        traders_df = engineer_trader_features(traders_df)
        context['ti'].xcom_push(key='traders_df_eng', value=traders_df.to_dict('records'))
    else:
        context['ti'].xcom_push(key='traders_df_eng', value=[])

    # Push engineered data to XCom
    context['ti'].xcom_push(key='markets_df_eng', value=markets_df.to_dict('records'))

    print(f"\n[OK] Task 3 complete: Engineered features for {len(markets_df):,} markets")


def task_upload_to_arango(**context):
    """
    Task 4: Upload all data to ArangoDB using upsert logic.
    Preserves historical data and enables change tracking.
    Also saves price history snapshots for time-series analysis.
    """
    print("\n" + "="*80)
    print("TASK 4/6: UPLOADING TO ARANGODB")
    print("="*80)

    # Get database connection
    db = get_arango_connection()

    # Pull engineered data from Task 3
    markets_dict = context['ti'].xcom_pull(key='markets_df_eng', task_ids='engineer_features')
    traders_dict = context['ti'].xcom_pull(key='traders_df_eng', task_ids='engineer_features')
    positions_dict = context['ti'].xcom_pull(key='positions_df', task_ids='fetch_traders')

    # Convert to DataFrames
    markets_df = pd.DataFrame(markets_dict)

    # Upload markets (with upsert)
    inserted, updated, errors = upsert_markets(db, markets_df)

    # Save price history snapshots (NEW!)
    snapshots_inserted, snapshots_errors = save_price_snapshots(db, markets_df)

    # Upload traders and positions (if available)
    traders_count = 0
    positions_count = 0
    trader_edges_count = 0
    position_edges_count = 0

    if traders_dict:
        traders_df = pd.DataFrame(traders_dict)
        positions_df = pd.DataFrame(positions_dict) if positions_dict else pd.DataFrame()

        traders_count, positions_count = upsert_traders(db, traders_df, positions_df)

        if len(positions_df) > 0:
            trader_edges_count = build_trader_position_edges(db)
            position_edges_count = build_position_market_edges(db)

    print(f"\n[OK] Task 4 complete:")
    print(f"  - Markets inserted: {inserted:,}")
    print(f"  - Markets updated: {updated:,}")
    print(f"  - Price snapshots: {snapshots_inserted:,}")
    print(f"  - Traders: {traders_count:,}")
    print(f"  - Positions: {positions_count:,}")
    print(f"  - Trader edges: {trader_edges_count:,} + {position_edges_count:,}")


def task_build_edges(**context):
    """
    Task 5: Build market→company graph edges.
    Creates direct mentions, sector links, and macro event connections.

    NOTE: This task is SKIPPED on most runs (expensive keyword matching).
    Only runs every 6 hours along with trader fetching.
    """
    print("\n" + "="*80)
    print("TASK 5/6: BUILDING GRAPH EDGES")
    print("="*80)

    # Skip edge building on most runs (expensive operation)
    execution_date = context['execution_date']
    hour = execution_date.hour
    minute = execution_date.minute

    # Only run at :00 minute of every 6th hour (same as traders)
    should_build_edges = (minute == 0 and hour % 6 == 0)

    if not should_build_edges:
        print("  ⏭️ Skipping edge building (only runs every 6 hours)")
        return

    print("  ✅ Running full edge build (6-hour cycle)")

    # Get database connection
    db = get_arango_connection()

    # Build all edges (loads markets and companies from database)
    stats = build_all_edges(db)

    print(f"\n[OK] Task 5 complete:")
    print(f"  - Direct mentions: {stats['direct']:,}")
    print(f"  - Sector edges: {stats['sector']:,}")
    print(f"  - Macro edges: {stats['macro']:,}")
    print(f"  - TOTAL: {stats['total']:,}")


def task_cleanup_old_data(**context):
    """
    Task 6: Clean up old price history snapshots (runs once daily).
    Removes snapshots older than 90 days to manage storage.
    """
    print("\n" + "="*80)
    print("TASK 6/6: CLEANUP OLD DATA")
    print("="*80)

    # Only run once per day (at midnight)
    execution_date = context['execution_date']
    hour = execution_date.hour
    minute = execution_date.minute

    should_cleanup = (hour == 0 and minute == 0)

    if not should_cleanup:
        print("  ⏭️ Skipping cleanup (only runs daily at midnight)")
        return

    print("  ✅ Running daily cleanup")

    # Get database connection
    db = get_arango_connection()

    # Clean up old snapshots
    deleted_count = cleanup_old_snapshots(db, days_to_keep=90)

    print(f"\n[OK] Task 6 complete: Deleted {deleted_count:,} old snapshots")


# ============================================================================
# DAG DEFINITION
# ============================================================================

# Default arguments for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 8),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='polymarket_etl_pipeline',
    default_args=default_args,
    description='Polymarket ETL: Fetch markets, traders, engineer features, upload to ArangoDB with real-time price tracking',
    schedule=timedelta(hours=8),  # Run every 8 hours
    start_date=datetime(2026, 1, 8),
    catchup=False,  # Don't backfill historical runs
    max_active_runs=1,  # Prevent overlapping runs
    tags=['polymarket', 'prediction-markets', 'etl', 'real-time'],
) as dag:

    # Task 1: Fetch markets
    fetch_markets_task = PythonOperator(
        task_id='fetch_markets',
        python_callable=task_fetch_markets,
    )

    # Task 2: Fetch traders
    fetch_traders_task = PythonOperator(
        task_id='fetch_traders',
        python_callable=task_fetch_traders,
    )

    # Task 3: Engineer features
    engineer_features_task = PythonOperator(
        task_id='engineer_features',
        python_callable=task_engineer_features,
    )

    # Task 4: Upload to ArangoDB
    upload_arango_task = PythonOperator(
        task_id='upload_to_arango',
        python_callable=task_upload_to_arango,
    )

    # Task 5: Build edges
    build_edges_task = PythonOperator(
        task_id='build_edges',
        python_callable=task_build_edges,
    )

    # Task 6: Cleanup old data
    cleanup_task = PythonOperator(
        task_id='cleanup_old_data',
        python_callable=task_cleanup_old_data,
    )

    # Define task dependencies (linear pipeline)
    fetch_markets_task >> fetch_traders_task >> engineer_features_task >> upload_arango_task >> build_edges_task >> cleanup_task


# ============================================================================
# DOCUMENTATION
# ============================================================================

"""
POLYMARKET ETL PIPELINE (REAL-TIME)

Purpose:
--------
Fetches prediction market data from Polymarket (via Gamma API and subgraph),
engineers features, saves price history, and loads into ArangoDB graph database
with company linkages.

Schedule:
---------
▶ EVERY 10 MINUTES: Markets + price history snapshots (real-time tracking)
▶ EVERY 6 HOURS: Traders, graph edges (expensive operations)
▶ DAILY: Cleanup old price snapshots (storage management)

Execution Pattern:
------------------
10-min runs (most runs):
  ✅ Fetch markets
  ⏭️  Skip traders
  ✅ Engineer features
  ✅ Upload markets
  ✅ Save price snapshots
  ⏭️  Skip edge building
  ⏭️  Skip cleanup

6-hour runs (at :00 of 0, 6, 12, 18 hours):
  ✅ Fetch markets
  ✅ Fetch traders (FULL)
  ✅ Engineer features
  ✅ Upload all data
  ✅ Save price snapshots
  ✅ Build graph edges (FULL)
  ⏭️  Skip cleanup (unless midnight)

Daily runs (at 00:00):
  ✅ All of the above
  ✅ Cleanup old snapshots

Key Features:
-------------
1. REAL-TIME PRICE TRACKING: 10-min snapshots for all markets
2. HISTORICAL DATA: Time-series collection for charting
3. UPSERT LOGIC: Preserves historical data, enables change tracking
4. SMART SCHEDULING: Skip expensive ops on fast runs
5. TRADER TRACKING: Uses reliable subgraph API (not orderbooks)
6. GRAPH EDGES: Links markets to companies via keywords, sectors, macro events
7. FEATURE ENGINEERING: Activity scores, probability confidence, trader metrics
8. ERROR HANDLING: Retries with exponential backoff, detailed logging

Data Flow:
----------
Every 10 minutes:
  Polymarket API → Markets DataFrame → Feature Engineering → ArangoDB (upsert)
                                                            → Price History Collection

Every 6 hours (additionally):
  Subgraph API → Traders DataFrame → Feature Engineering → ArangoDB (upsert)
  ArangoDB (markets + companies) → Edge Builder → Graph Edges

Output Collections:
-------------------
- prediction_markets_polymarket: Market documents with engineered features (UPSERT)
- polymarket_price_history: Time-series price snapshots (INSERT, 10-min resolution)
- polymarket_traders: Trader documents with volume ranks, activity levels (6-hour)
- polymarket_positions: Position documents (6-hour)
- market_mentions_company_polymarket: Direct company mention edges (6-hour)
- market_related_to_sector_polymarket: Sector-level edges (6-hour)
- market_affects_company_polymarket: Macro event edges (6-hour)

Monitoring:
-----------
- Check Airflow UI for task status
- Review task logs for detailed execution info
- Verify data freshness: markets should have fetched_at within last 10 minutes
- Check price_history collection size (should grow by ~100 docs every 10 min)
- Monitor for skipped tasks in logs (expected behavior)

Manual Trigger:
---------------
airflow dags trigger polymarket_etl_pipeline

Testing:
--------
See: DAGS/test_pipelines_local.py for local testing without Airflow

Performance:
------------
- 10-min runs: ~30-60 seconds (fast, markets only)
- 6-hour runs: ~3-5 minutes (full ETL with traders and edges)
- Storage: ~14k price snapshots per day (100 markets × 144 runs)
- Cleanup: Removes snapshots older than 90 days automatically
"""