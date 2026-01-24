"""
Polymarket Pipeline Scheduler - Railway Service
Runs the full ETL pipeline every 10 minutes automatically
No Airflow, no Docker - just pure Python
"""

import os
import sys
import time
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Add the pipeline directory to Python path
# From src/scheduler/app.py -> go up to src/ -> into DAGS/pipeline/
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'DAGS', 'pipeline')
sys.path.insert(0, PIPELINE_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Debug: Show paths before importing
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Scheduler file location: {__file__}")
logger.info(f"Pipeline directory: {PIPELINE_DIR}")
logger.info(f"Pipeline directory exists: {os.path.exists(PIPELINE_DIR)}")
if os.path.exists(PIPELINE_DIR):
    logger.info(f"Pipeline directory contents: {os.listdir(PIPELINE_DIR)}")
else:
    logger.error(f"❌ Pipeline directory NOT FOUND at: {PIPELINE_DIR}")
    # Try to find where it actually is
    cwd_contents = os.listdir(os.getcwd())
    logger.info(f"Current directory contents: {cwd_contents}")

# Import pipeline modules
try:
    # Polymarket
    from polymarket.downloader import fetch_all_markets
    from polymarket.features import engineer_market_features
    from polymarket.arango_uploader import get_arango_connection, upsert_markets
    from polymarket.price_history import save_price_snapshots
    from polymarket.trader_tracker import fetch_traders_from_subgraph, parse_trader_positions
    from polymarket.arango_uploader import upsert_traders, create_trader_edges
    from polymarket.edge_builder import build_all_edges
    from polymarket.price_history import cleanup_old_snapshots

    # Yahoo
    from yahoo.constituents import get_sp500_tickers
    from yahoo.downloader import download_stock_data
    from yahoo.features import engineer_technical_features

    # Kalshi
    from kalshi.downloader import fetch_all_markets as fetch_kalshi_markets
    from kalshi.features import engineer_market_features as engineer_kalshi_features
    from kalshi.arango_uploader import upsert_markets as upsert_kalshi_markets

    # Awards (optional - may not be deployed yet)
    try:
        from awards.downloader import fetch_recent_awards
        from awards.features import generate_embeddings
        from awards.arango_uploader import upsert_awards
        AWARDS_AVAILABLE = True
        logger.info("✓ Awards pipeline available")
    except ImportError as e:
        AWARDS_AVAILABLE = False
        logger.warning(f"⚠️ Awards pipeline not available (not deployed): {e}")

    # CFTC (optional)
    try:
        from cftc.downloader import fetch_recent_cftc_data
        from cftc.features import engineer_cftc_features
        from cftc.arango_uploader import upsert_commodity_positions
        # Import commodity company edge creator
        from create_commodity_company_edges import create_commodity_company_edges
        CFTC_AVAILABLE = True
        logger.info("✓ CFTC pipeline available")
    except ImportError as e:
        CFTC_AVAILABLE = False
        logger.warning(f"⚠️ CFTC pipeline not available (not deployed): {e}")

    # EIA (optional)
    try:
        from eia.downloader import fetch_all_eia_data
        from eia.features import engineer_eia_features
        from eia.arango_uploader import upsert_all_eia_data
        EIA_AVAILABLE = True
        logger.info("✓ EIA pipeline available")
    except ImportError as e:
        EIA_AVAILABLE = False
        logger.warning(f"⚠️ EIA pipeline not available (not deployed): {e}")

    # FRED (optional)
    try:
        from fred.downloader import fetch_all_fred_data
        from fred.features import engineer_fred_features
        from fred.arango_uploader import upsert_fred_data
        FRED_AVAILABLE = True
        logger.info("✓ FRED pipeline available")
    except ImportError as e:
        FRED_AVAILABLE = False
        logger.warning(f"⚠️ FRED pipeline not available (not deployed): {e}")
        import traceback
        traceback.print_exc()

    # CME Futures (DISABLED - Yahoo Finance blocks Railway IPs)
    # Commodity prices now come from FRED instead (see fred/downloader.py)
    CME_AVAILABLE = False
    # try:
    #     from cme.downloader import fetch_all_futures_data
    #     from cme.features import engineer_futures_features
    #     from cme.arango_uploader import upsert_futures_data
    #     CME_AVAILABLE = True
    #     logger.info("✓ CME Futures pipeline available")
    # except ImportError as e:
    #     CME_AVAILABLE = False
    #     logger.warning(f"⚠️ CME Futures pipeline not available (not deployed): {e}")

    logger.info("✓ Successfully imported all pipeline modules")
except Exception as e:
    logger.error(f"✗ Failed to import pipeline modules: {e}")
    sys.exit(1)


def run_yahoo_pipeline():
    """Execute Yahoo MarketData ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("YAHOO MARKETDATA PIPELINE")
    logger.info("="*80)

    try:
        # Step 1: Get current S&P 500 tickers only (not historical)
        logger.info("[1/4] Fetching current S&P 500 tickers...")
        tickers = get_sp500_tickers(current_only=True)
        logger.info(f"✓ Fetched {len(tickers)} tickers")

        # Step 2: Download data ONE TICKER AT A TIME to avoid rate limiting
        logger.info("[2/4] Downloading stock data (30 days)...")
        logger.info(f"  Downloading one ticker at a time with 5.0s delay (~{len(tickers) * 5.0 / 60:.1f} min)")
        data_df = download_stock_data(tickers, period='1mo', sleep_between_tickers=5.0)
        logger.info(f"✓ Downloaded {len(data_df)} rows")

        # Check if download succeeded
        if data_df.empty or len(data_df) == 0:
            logger.error("✗ No data downloaded - skipping feature engineering and upload")
            return False

        # Step 3: Engineer features (without fundamentals to avoid extra API calls)
        logger.info("[3/4] Engineering technical indicators...")
        featured_df = engineer_technical_features(data_df, include_fundamentals=False)
        logger.info(f"✓ Engineered {len(featured_df)} rows")

        # Step 4: Upload to ArangoDB
        logger.info("[4/4] Uploading to MarketData collection...")
        db = get_arango_connection()

        docs = []
        for _, row in featured_df.iterrows():
            doc = {
                '_key': f"{row['ticker']}_{row['date']}",
                'ticker': row['ticker'],
                'date': row['date'],
                **{k: v for k, v in row.to_dict().items() if k not in ['ticker', 'date']}
            }
            docs.append(doc)

        # Batch upsert
        for i in range(0, len(docs), 250):
            batch = docs[i:i+250]
            db.aql.execute(
                "FOR doc IN @docs UPSERT {_key: doc._key} INSERT doc UPDATE doc IN MarketData",
                bind_vars={'docs': batch}
            )

        logger.info(f"✓ Uploaded {len(docs)} records")
        return True

    except Exception as e:
        logger.error(f"✗ Yahoo pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_kalshi_pipeline():
    """Execute Kalshi ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("KALSHI PIPELINE")
    logger.info("="*80)

    try:
        # Step 1: Fetch markets (only open/active ones for scheduler)
        logger.info("[1/3] Fetching Kalshi markets (open only)...")
        markets_df = fetch_kalshi_markets(status_filter='open', limit=2000)
        logger.info(f"✓ Fetched {len(markets_df)} markets")

        # Step 2: Engineer features + embeddings (skip embeddings - use standalone)
        logger.info("[2/3] Engineering features (skipping embeddings)...")
        # Note: engineer_market_features will call generate_title_embeddings
        # but it skips if embeddings already exist
        markets_df = engineer_kalshi_features(markets_df)
        logger.info(f"✓ Engineered {len(markets_df)} markets")

        # Step 3: Upload to ArangoDB (don't truncate - update existing)
        logger.info("[3/3] Uploading to ArangoDB...")
        db = get_arango_connection()
        inserted, updated, errors = upsert_kalshi_markets(db, markets_df, truncate_first=False)
        logger.info(f"✓ Inserted: {inserted}, Updated: {updated}")

        return True

    except Exception as e:
        logger.error(f"✗ Kalshi pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_fred_pipeline():
    """Execute FRED Federal Reserve Economic Data ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("FRED ECONOMIC DATA PIPELINE")
    logger.info("="*80)

    try:
        # Check for historical backfill mode
        historical_years = os.getenv('FRED_HISTORICAL_YEARS')
        if historical_years:
            years = int(historical_years)
            logger.info(f"[1/3] Fetching FRED data (last {years} years - HISTORICAL BACKFILL)...")
            fred_df = fetch_all_fred_data(years_back=years)
        else:
            # Incremental: last 90 days (catches all monthly releases)
            logger.info("[1/3] Fetching FRED data (last 90 days)...")
            fred_df = fetch_all_fred_data(days_back=90)

        if fred_df.empty:
            logger.info("✓ No new FRED data found")
            return True

        logger.info(f"✓ Fetched {len(fred_df)} records")

        # Step 2: Engineer features (pivot to wide format)
        logger.info("[2/3] Engineering features...")
        fred_df = engineer_fred_features(fred_df)
        logger.info(f"✓ Processed {len(fred_df)} dates")

        # Step 3: Upload to ArangoDB with graph edges
        logger.info("[3/3] Uploading to ArangoDB and creating graph edges...")
        db = get_arango_connection()
        inserted, updated, edge_counts = upsert_fred_data(db, fred_df)

        logger.info(f"✓ Inserted: {inserted}, Updated: {updated}")
        for edge_type, count in edge_counts.items():
            logger.info(f"  {edge_type}: {count}")

        return True

    except Exception as e:
        logger.error(f"✗ FRED pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_eia_pipeline():
    """Execute EIA Energy Information Administration ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("EIA ENERGY DATA PIPELINE")
    logger.info("="*80)

    try:
        # Check for historical backfill mode
        historical_years = os.getenv('EIA_HISTORICAL_YEARS')
        if historical_years:
            years = int(historical_years)
            logger.info(f"[1/3] Fetching EIA energy data (last {years} years - HISTORICAL BACKFILL)...")
            datasets = fetch_all_eia_data(years_back=years)
        else:
            # Step 1: Fetch all EIA datasets (last 4 weeks for incremental updates)
            logger.info("[1/3] Fetching EIA energy data (last 4 weeks)...")
            datasets = fetch_all_eia_data(weeks_back=4)

        total_records = sum(len(df) for df in datasets.values() if not df.empty)
        if total_records == 0:
            logger.info("✓ No new EIA data found")
            return True

        logger.info(f"✓ Fetched {total_records} total records")

        # Step 2: Engineer features for each dataset
        logger.info("[2/3] Engineering features...")
        for dataset_key, df in datasets.items():
            if not df.empty:
                datasets[dataset_key] = engineer_eia_features(df, dataset_key)
        logger.info(f"✓ Processed {total_records} records")

        # Step 3: Upload to ArangoDB with graph edges
        logger.info("[3/3] Uploading to ArangoDB and creating graph edges...")
        db = get_arango_connection()
        results = upsert_all_eia_data(db, datasets)

        # Summary
        total_inserted = sum(r['inserted'] for r in results.values())
        total_updated = sum(r['updated'] for r in results.values())
        total_edges = sum(r['edges'] for r in results.values())

        logger.info(f"✓ Inserted: {total_inserted}, Updated: {total_updated}, Edges: {total_edges}")

        return True

    except Exception as e:
        logger.error(f"✗ EIA pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_cftc_pipeline():
    """Execute CFTC Commitments of Traders ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("CFTC COMMITMENTS OF TRADERS PIPELINE")
    logger.info("="*80)

    try:
        # Step 1: Fetch recent CFTC data (last 4 weeks)
        logger.info("[1/4] Fetching recent CFTC data (last 4 weeks)...")
        cftc_df = fetch_recent_cftc_data(weeks_back=4)

        if cftc_df.empty:
            logger.info("✓ No new CFTC data found")
            return True

        logger.info(f"✓ Fetched {len(cftc_df)} records")

        # Step 2: Engineer features
        logger.info("[2/4] Engineering features...")
        cftc_df = engineer_cftc_features(cftc_df)
        logger.info(f"✓ Processed {len(cftc_df)} records")

        # Step 3: Upload to ArangoDB
        logger.info("[3/4] Uploading to ArangoDB...")
        db = get_arango_connection()
        inserted, updated = upsert_commodity_positions(db, cftc_df)
        logger.info(f"✓ Inserted: {inserted}, Updated: {updated}")

        # Step 4: Create Company → Commodity edges
        logger.info("[4/4] Creating Company → Commodity edges...")
        try:
            edges = create_commodity_company_edges()
            logger.info(f"✓ Created {len(edges):,} Company → futures_prices edges")
        except Exception as e:
            logger.warning(f"⚠️ Edge creation failed (non-fatal): {e}")
            # Don't fail the whole pipeline if edge creation fails

        return True

    except Exception as e:
        logger.error(f"✗ CFTC pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_cme_pipeline():
    """Execute CME/NYMEX Futures Prices ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("CME/NYMEX FUTURES PRICES PIPELINE")
    logger.info("="*80)

    try:
        # Check for historical backfill mode
        historical_years = os.getenv('CME_HISTORICAL_YEARS')
        if historical_years:
            years = int(historical_years)
            days = years * 365
            logger.info(f"[1/3] Fetching futures data (last {years} years - HISTORICAL BACKFILL)...")
            futures_df = fetch_all_futures_data(days_back=days)
        else:
            # Incremental: last 90 days (matches CFTC weekly updates)
            logger.info("[1/3] Fetching futures data (last 90 days)...")
            futures_df = fetch_all_futures_data(days_back=90)

        if futures_df.empty:
            logger.info("✓ No new futures data found")
            return True

        logger.info(f"✓ Fetched {len(futures_df)} records across {futures_df['commodity'].nunique()} commodities")

        # Step 2: Engineer features (technical indicators, momentum, volatility)
        logger.info("[2/3] Engineering features...")
        futures_df = engineer_futures_features(futures_df)
        logger.info(f"✓ Processed {len(futures_df)} records")

        # Step 3: Upload to ArangoDB with graph edges
        logger.info("[3/3] Uploading to ArangoDB and creating graph edges...")
        db = get_arango_connection()
        inserted, updated, edge_counts = upsert_futures_data(db, futures_df)

        logger.info(f"✓ Inserted: {inserted}, Updated: {updated}")
        for edge_type, count in edge_counts.items():
            logger.info(f"  {edge_type}: {count}")

        return True

    except Exception as e:
        logger.error(f"✗ CME pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_awards_pipeline():
    """Execute Awards (USASpending) ETL pipeline"""
    logger.info("\n" + "="*80)
    logger.info("AWARDS (USASPENDING) PIPELINE")
    logger.info("="*80)

    try:
        # Step 1: Fetch recent awards (last 2 days to ensure we don't miss any)
        logger.info("[1/3] Fetching recent awards (last 2 days)...")
        awards_df = fetch_recent_awards(days_back=2)

        if awards_df.empty:
            logger.info("✓ No new awards found")
            return True

        logger.info(f"✓ Fetched {len(awards_df)} awards")

        # Step 2: Generate embeddings
        logger.info("[2/3] Generating FinBERT embeddings...")
        awards_df = generate_embeddings(awards_df, batch_size=64)
        logger.info(f"✓ Generated embeddings for {len(awards_df)} awards")

        # Step 3: Upload to ArangoDB
        logger.info("[3/3] Uploading to ArangoDB...")
        db = get_arango_connection()
        inserted, updated, edges = upsert_awards(db, awards_df)
        logger.info(f"✓ Inserted: {inserted}, Updated: {updated}, Edges: {edges}")

        return True

    except Exception as e:
        logger.error(f"✗ Awards pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_polymarket_pipeline(skip_embeddings=False):
    """
    Execute the Polymarket ETL pipeline

    Args:
        skip_embeddings: If True, skips embedding generation (run separately via standalone script)

    This runs:
    1. Fetch markets (always)
    2. Fetch traders (every 6 hours)
    3. Engineer features (conditionally skip embeddings)
    4. Upload to ArangoDB (always)
    5. Save price snapshots (always)
    6. Build graph edges (every 6 hours)
    7. Cleanup old data (daily at midnight)
    """

    start_time = datetime.now()
    logger.info("\n" + "="*80)
    logger.info(f"POLYMARKET PIPELINE STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    if skip_embeddings:
        logger.info("⚠️  EMBEDDINGS SKIPPED - Run generate_embeddings_standalone.py locally")

    try:
        # Determine what to run based on time
        hour = start_time.hour
        minute = start_time.minute

        # Every 6 hours at :00 (0:00, 6:00, 12:00, 18:00)
        # TEMPORARY: Force traders to fetch on every run for testing
        should_fetch_traders = True  # CHANGE BACK TO: (minute == 0 and hour % 6 == 0)
        should_build_edges = should_fetch_traders

        # Daily at midnight
        should_cleanup = (hour == 0 and minute == 0)

        logger.info(f"Execution mode: traders={should_fetch_traders}, edges={should_build_edges}, cleanup={should_cleanup}")

        # Step 1: Fetch markets from Polymarket API
        logger.info("\n[1/7] Fetching markets from Polymarket API...")
        markets_df = fetch_all_markets()
        logger.info(f"✓ Fetched {len(markets_df):,} markets")

        # Step 2: Fetch traders (conditional)
        traders_df = None
        positions_df = None

        if should_fetch_traders:
            logger.info("\n[2/7] Fetching traders (6-hour cycle)...")
            try:
                # Build condition map
                market_condition_map = {}
                for _, row in markets_df.iterrows():
                    if 'condition_id' in row.index and 'market_id' in row.index:
                        cond_id = row['condition_id']
                        market_id = row['market_id']
                        if cond_id and market_id:
                            market_condition_map[cond_id] = f"prediction_markets_polymarket/{market_id}"

                # Fetch traders
                traders_raw = fetch_traders_from_subgraph(min_volume=1000)
                if traders_raw:
                    traders_df, positions_df = parse_trader_positions(traders_raw, market_condition_map)
                    logger.info(f"✓ Fetched {len(traders_df):,} traders, {len(positions_df):,} positions")
                else:
                    logger.warning("No traders fetched from subgraph")
            except Exception as e:
                logger.error(f"✗ Trader fetching failed: {e}")
        else:
            logger.info("\n[2/7] Skipping trader fetch (not 6-hour cycle)")

        # Step 3: Connect to database (EARLY - needed for incremental embedding saves)
        logger.info("\n[3/7] Connecting to ArangoDB...")
        db = get_arango_connection()
        logger.info("✓ Connected to database")

        # Step 4: Engineer features (with db connection for crash recovery)
        logger.info("\n[4/7] Engineering features...")
        markets_df = engineer_market_features(markets_df, db=db, skip_embeddings=skip_embeddings)
        logger.info(f"✓ Engineered features for {len(markets_df):,} markets")

        if traders_df is not None and len(traders_df) > 0:
            from polymarket.features import engineer_trader_features
            traders_df = engineer_trader_features(traders_df)
            logger.info(f"✓ Engineered features for {len(traders_df):,} traders")

        # Step 5: Upload markets
        logger.info("\n[5/7] Uploading markets to ArangoDB...")
        inserted, updated, errors = upsert_markets(db, markets_df)
        logger.info(f"✓ Markets - Inserted: {inserted:,}, Updated: {updated:,}, Errors: {errors}")

        # Upload traders (conditional)
        if traders_df is not None and len(traders_df) > 0:
            logger.info("Uploading traders and positions...")
            traders_count, positions_count = upsert_traders(db, traders_df, positions_df)
            logger.info(f"✓ Traders: {traders_count:,}, Positions: {positions_count:,}")

            # Create trader edges
            if positions_df is not None and len(positions_df) > 0:
                logger.info("Creating trader edges...")
                trader_edges = create_trader_edges(db, positions_df)
                logger.info(f"✓ Created trader edges: {trader_edges}")

        # Step 6: Save price history snapshots
        logger.info("\n[6/7] Saving price history snapshots...")
        snapshots_inserted, snapshots_errors = save_price_snapshots(db, markets_df)
        logger.info(f"✓ Price snapshots - Inserted: {snapshots_inserted:,}, Errors: {snapshots_errors}")

        # Step 7: Build graph edges (conditional)
        if should_build_edges:
            logger.info("\n[7/7] Building graph edges (6-hour cycle)...")
            try:
                stats = build_all_edges(db)
                logger.info(f"✓ Graph edges - Direct: {stats['direct']:,}, Sector: {stats['sector']:,}, Macro: {stats['macro']:,}")
            except Exception as e:
                logger.error(f"✗ Edge building failed: {e}")
        else:
            logger.info("\n[7/7] Skipping edge building (not 6-hour cycle)")

        # Step 8: Cleanup old data (conditional)
        if should_cleanup:
            logger.info("\n[8/7] Cleaning up old data (daily cycle)...")
            try:
                deleted_count = cleanup_old_snapshots(db, days_to_keep=90)
                logger.info(f"✓ Deleted {deleted_count:,} old snapshots")
            except Exception as e:
                logger.error(f"✗ Cleanup failed: {e}")

        # Success summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Markets: {inserted:,} inserted, {updated:,} updated")
        logger.info(f"Price snapshots: {snapshots_inserted:,}")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        return False


def run_pipeline():
    """
    Master pipeline orchestrator - runs all pipelines in sequence
    Order: Yahoo → Kalshi → Polymarket (skip embeddings)

    Embeddings are generated separately via standalone script to avoid Railway timeouts
    """
    start_time = datetime.now()
    logger.info("\n" + "="*80)
    logger.info(f"MASTER PIPELINE STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("Execution order: FRED (includes commodity prices) → CFTC → EIA → Awards → Kalshi → Polymarket")
    logger.info("Note: Yahoo & CME disabled (Railway IPs blocked)")
    logger.info("="*80 + "\n")

    results = {
        'yahoo': False,
        'kalshi': False,
        'polymarket': False,
        'awards': False,
        'cftc': False,
        'eia': False,
        'fred': False
    }

    # Pipeline 1: FRED Economic Data (includes macro + commodity prices)
    if FRED_AVAILABLE:
        try:
            results['fred'] = run_fred_pipeline()
        except Exception as e:
            logger.error(f"FRED pipeline crashed: {e}")
            results['fred'] = False

    # Pipeline 3: CFTC Commitments of Traders
    if CFTC_AVAILABLE:
        try:
            results['cftc'] = run_cftc_pipeline()
        except Exception as e:
            logger.error(f"CFTC pipeline crashed: {e}")
            results['cftc'] = False

    # Pipeline 4: EIA Energy Data
    if EIA_AVAILABLE:
        try:
            results['eia'] = run_eia_pipeline()
        except Exception as e:
            logger.error(f"EIA pipeline crashed: {e}")
            results['eia'] = False

    # Pipeline 5: Awards (USASpending)
    if AWARDS_AVAILABLE:
        try:
            results['awards'] = run_awards_pipeline()
        except Exception as e:
            logger.error(f"Awards pipeline crashed: {e}")
            results['awards'] = False

    # Pipeline 6: Kalshi
    try:
        results['kalshi'] = run_kalshi_pipeline()
    except Exception as e:
        logger.error(f"Kalshi pipeline crashed: {e}")
        results['kalshi'] = False

    # Pipeline 7: Polymarket (skip embeddings - use standalone script)
    try:
        results['polymarket'] = run_polymarket_pipeline(skip_embeddings=True)
    except Exception as e:
        logger.error(f"Polymarket pipeline crashed: {e}")
        results['polymarket'] = False

    # Pipeline 8: Yahoo MarketData (DISABLED - Railway IPs blocked by Yahoo)
    logger.info("Skipping Yahoo (blocked on Railway datacenter IPs)")
    results['yahoo'] = True  # Skip but don't fail pipeline

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "="*80)
    logger.info("MASTER PIPELINE COMPLETE")
    logger.info("="*80)
    if FRED_AVAILABLE:
        logger.info(f"FRED (with commodity prices): {'✓ SUCCESS' if results['fred'] else '✗ FAILED'}")
    if CFTC_AVAILABLE:
        logger.info(f"CFTC: {'✓ SUCCESS' if results['cftc'] else '✗ FAILED'}")
    if EIA_AVAILABLE:
        logger.info(f"EIA: {'✓ SUCCESS' if results['eia'] else '✗ FAILED'}")
    if AWARDS_AVAILABLE:
        logger.info(f"Awards: {'✓ SUCCESS' if results['awards'] else '✗ FAILED'}")
    logger.info(f"Kalshi: {'✓ SUCCESS' if results['kalshi'] else '✗ FAILED'}")
    logger.info(f"Polymarket: {'✓ SUCCESS' if results['polymarket'] else '✗ FAILED'}")
    logger.info(f"Yahoo & CME: SKIPPED (Railway IPs blocked)")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
    logger.info("="*80)


def health_check():
    """Simple health check that runs every minute"""
    logger.info(f"[HEALTH] Service running - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """
    Main entry point - sets up scheduler and runs forever
    """
    logger.info("\n" + "="*80)
    logger.info("POLYMARKET PIPELINE SCHEDULER - STARTING")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Schedule: Every 12 hours")
    logger.info(f"Environment: Railway")
    logger.info("="*80 + "\n")

    # Verify environment variables (support both ARANGO_HOST and ARANGO_URL)
    required_vars = ['ARANGO_DATABASE', 'ARANGO_USERNAME', 'ARANGO_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    # Check for either ARANGO_HOST or ARANGO_URL
    if not (os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')):
        missing_vars.append('ARANGO_URL or ARANGO_HOST')

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set them in Railway's environment variables section")
        sys.exit(1)

    logger.info("✓ All environment variables set")

    # Run pipeline immediately on startup
    logger.info("\n→ Running initial pipeline execution...")
    run_pipeline()

    # Set up scheduler
    scheduler = BlockingScheduler(timezone='UTC')

    # Main pipeline job - every 12 hours
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(hours=12),
        id='pipeline_job',
        name='Polymarket ETL Pipeline',
        replace_existing=True
    )

    # Health check job - every 1 minute
    scheduler.add_job(
        health_check,
        trigger=IntervalTrigger(minutes=1),
        id='health_check',
        name='Health Check',
        replace_existing=True
    )

    logger.info("\n✓ Scheduler configured:")
    logger.info("  - Pipeline: Every 12 hours")
    logger.info("  - Health check: Every 1 minute")
    logger.info("\n→ Scheduler starting (Ctrl+C to stop)...\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n→ Scheduler stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
