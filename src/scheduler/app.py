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
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'DAGS', 'pipeline')
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
    from polymarket.downloader import fetch_all_markets
    from polymarket.features import engineer_market_features
    from polymarket.arango_uploader import get_arango_connection, upsert_markets
    from polymarket.price_history import save_price_snapshots
    from polymarket.trader_tracker import fetch_traders_from_subgraph, parse_trader_positions
    from polymarket.arango_uploader import upsert_traders, create_trader_edges
    from polymarket.edge_builder import build_all_edges
    from polymarket.price_history import cleanup_old_snapshots

    logger.info("✓ Successfully imported all pipeline modules")
except Exception as e:
    logger.error(f"✗ Failed to import pipeline modules: {e}")
    sys.exit(1)


def run_pipeline():
    """
    Execute the full Polymarket ETL pipeline

    This runs:
    1. Fetch markets (always)
    2. Fetch traders (every 6 hours)
    3. Engineer features (always)
    4. Upload to ArangoDB (always)
    5. Save price snapshots (always)
    6. Build graph edges (every 6 hours)
    7. Cleanup old data (daily at midnight)
    """

    start_time = datetime.now()
    logger.info("="*80)
    logger.info(f"PIPELINE EXECUTION STARTED: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    try:
        # Determine what to run based on time
        hour = start_time.hour
        minute = start_time.minute

        # Every 6 hours at :00 (0:00, 6:00, 12:00, 18:00)
        should_fetch_traders = (minute == 0 and hour % 6 == 0)
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

        # Step 3: Engineer features
        logger.info("\n[3/7] Engineering features...")
        markets_df = engineer_market_features(markets_df)
        logger.info(f"✓ Engineered features for {len(markets_df):,} markets")

        if traders_df is not None and len(traders_df) > 0:
            from polymarket.features import engineer_trader_features
            traders_df = engineer_trader_features(traders_df)
            logger.info(f"✓ Engineered features for {len(traders_df):,} traders")

        # Step 4: Connect to database
        logger.info("\n[4/7] Connecting to ArangoDB...")
        db = get_arango_connection()
        logger.info("✓ Connected to database")

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
    logger.info(f"Schedule: Every 10 minutes")
    logger.info(f"Environment: Railway")
    logger.info("="*80 + "\n")

    # Verify environment variables
    required_vars = ['ARANGO_HOST', 'ARANGO_DATABASE', 'ARANGO_USERNAME', 'ARANGO_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

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

    # Main pipeline job - every 10 minutes
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(minutes=10),
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
    logger.info("  - Pipeline: Every 10 minutes")
    logger.info("  - Health check: Every 1 minute")
    logger.info("\n→ Scheduler starting (Ctrl+C to stop)...\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n→ Scheduler stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
