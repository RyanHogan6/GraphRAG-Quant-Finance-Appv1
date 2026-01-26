"""
Options Flow Pipeline - Local Execution Script
Run this on your local machine to bypass Railway IP blocking

Usage:
    python run_options_local.py                    # Fetch top 50 liquid stocks
    python run_options_local.py --max-tickers 100  # Fetch 100 stocks
    python run_options_local.py --delay 2.0        # 2 second delay between tickers
"""
import sys
import os
from datetime import datetime
import argparse

# Add pipeline directory to path (parent of options/)
# This allows: from options.downloader import ...
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Fetch options flow data locally and upload to ArangoDB')
    parser.add_argument('--max-tickers', type=int, help='Number of tickers to fetch (default: all from Company collection)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between tickers in seconds (default: 1.0)')
    parser.add_argument('--all', action='store_true', help='Fetch ALL tickers from Company collection (no limit)')
    args = parser.parse_args()

    print("="*80)
    print("OPTIONS FLOW PIPELINE - LOCAL EXECUTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Delay: {args.delay}s between requests")
    print("="*80 + "\n")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Verify ArangoDB credentials
    arango_url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
    arango_db = os.getenv('ARANGO_DATABASE') or os.getenv('ARANGO_DB')
    arango_password = os.getenv('ARANGO_PASSWORD')

    if not all([arango_url, arango_db, arango_password]):
        print("❌ Missing ArangoDB credentials in .env file!")
        print("Required: ARANGO_URL, ARANGO_DATABASE, ARANGO_PASSWORD")
        sys.exit(1)

    print(f"✓ ArangoDB: {arango_url}")
    print(f"✓ Database: {arango_db}\n")

    # Get tickers from Company collection
    from options.downloader import get_tickers_from_company_collection
    all_tickers = get_tickers_from_company_collection()

    print(f"✓ Found {len(all_tickers)} tickers in Company collection")

    # Determine how many to fetch
    if args.all:
        max_tickers = len(all_tickers)
        print(f"✓ Fetching ALL {max_tickers} tickers")
    elif args.max_tickers:
        max_tickers = args.max_tickers
        print(f"✓ Limiting to first {max_tickers} tickers")
    else:
        # Default: fetch all if <= 100, otherwise limit to 100
        if len(all_tickers) <= 100:
            max_tickers = len(all_tickers)
            print(f"✓ Fetching all {max_tickers} tickers (under 100)")
        else:
            max_tickers = 100
            print(f"⚠️  Limiting to first {max_tickers} tickers (use --all to fetch all {len(all_tickers)})")

    tickers_to_fetch = all_tickers[:max_tickers]

    estimated_time = (max_tickers * args.delay) / 60
    print(f"⏱️  Estimated time: {estimated_time:.1f} minutes")
    print()

    # Step 1: Fetch options data
    print("[1/3] Fetching options flow data...")
    print("-" * 80)

    try:
        from options.downloader import fetch_options_for_tickers

        options_df = fetch_options_for_tickers(
            tickers=tickers_to_fetch,
            delay=args.delay
        )

        if options_df.empty:
            print("✗ No options data fetched!")
            print("Note: Yahoo Finance may be throttling. Try:")
            print(f"  - Reduce --max-tickers (currently: {max_tickers})")
            print("  - Increase --delay to 2.0 or 3.0")
            print("  - Wait a few minutes and try again")
            print("  - Some tickers may not have options (small caps, recent IPOs)")
            sys.exit(1)

        print(f"\n✓ Fetched {len(options_df)} tickers")
        print(f"✓ Date: {options_df['date'].iloc[0]}")

        # Show summary stats
        print("\nSummary statistics:")
        print(f"  Total volume: {options_df['total_volume'].sum():,}")
        print(f"  Avg put/call ratio: {options_df['put_call_volume_ratio'].mean():.2f}")
        print(f"  Tickers with unusual activity: {(options_df['total_volume'] > options_df['total_volume'].median() * 2).sum()}")

        # Show top 5 by volume
        print("\nTop 5 by options volume:")
        top5 = options_df.nlargest(5, 'total_volume')[['ticker', 'total_volume', 'put_call_volume_ratio', 'call_premium', 'put_premium']]
        print(top5.to_string(index=False))

    except Exception as e:
        print(f"✗ Failed to fetch options data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Engineer features
    print("\n[2/3] Engineering features...")
    print("-" * 80)

    try:
        from options.features import engineer_options_features

        # Note: First run won't have historical data for averages
        # Features like unusual_volume require multiple days of data
        featured_df = engineer_options_features(options_df)

        print(f"✓ Engineered {len(featured_df)} records")
        print(f"✓ Features: {len(featured_df.columns)} columns")

        # Show detected signals
        bullish = (featured_df['bullish_signal'] == 1).sum()
        bearish = (featured_df['bearish_signal'] == 1).sum()
        call_sweeps = (featured_df['potential_call_sweep'] == 1).sum()
        put_sweeps = (featured_df['potential_put_sweep'] == 1).sum()

        print(f"\nSignals detected:")
        print(f"  Bullish signals: {bullish}")
        print(f"  Bearish signals: {bearish}")
        print(f"  Potential call sweeps: {call_sweeps}")
        print(f"  Potential put sweeps: {put_sweeps}")

        if call_sweeps > 0:
            print("\nPotential call sweeps:")
            sweeps = featured_df[featured_df['potential_call_sweep'] == 1][
                ['ticker', 'call_volume', 'put_call_volume_ratio', 'call_premium']
            ]
            print(sweeps.to_string(index=False))

    except Exception as e:
        print(f"✗ Failed to engineer features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Upload to ArangoDB
    print("\n[3/3] Uploading to ArangoDB and creating graph edges...")
    print("-" * 80)

    try:
        from options.arango_uploader import get_arango_connection, upsert_options_data

        print("Connecting to ArangoDB...")
        db = get_arango_connection()

        print("Upserting options data...")
        inserted, updated, edge_counts = upsert_options_data(db, featured_df)

        print(f"\n✓ Inserted: {inserted:,}")
        print(f"✓ Updated: {updated:,}")

        if edge_counts:
            print("\nGraph edges created:")
            for edge_type, count in edge_counts.items():
                print(f"  {edge_type}: {count:,}")
        else:
            print("\n⚠️  No graph edges created yet")
            print("Edges will be created as you accumulate:")
            print("  - MarketData for the same dates")
            print("  - Awards announced after unusual options activity")
            print("  - SEC filings following unusual activity")

    except Exception as e:
        print(f"\n✗ Failed to upload data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Success summary
    end_time = datetime.now()
    print("\n" + "="*80)
    print("✓ OPTIONS FLOW PIPELINE COMPLETE")
    print("="*80)
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tickers: {len(featured_df):,}")
    print(f"Database: {arango_db}")
    print(f"Collection: options_flow")
    print("\nNext steps:")
    print("  1. Run daily to build historical options data")
    print("  2. After 20+ days, unusual volume detection will work")
    print("  3. Query for insider activity patterns")
    print("="*80)


if __name__ == "__main__":
    main()
