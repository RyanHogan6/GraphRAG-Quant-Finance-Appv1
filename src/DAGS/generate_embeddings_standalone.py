"""
Standalone Embedding Generator - Run Locally
Generates embeddings for Polymarket markets with crash recovery
Can be safely stopped and resumed at any time
"""

import sys
import os
from datetime import datetime

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))

from polymarket.arango_uploader import get_arango_connection
from polymarket.features import get_openai_client
import pandas as pd
from tqdm import tqdm

def load_markets_without_embeddings(db):
    """Load all markets that don't have embeddings yet"""
    print("\n[1/4] Loading markets from ArangoDB...")

    query = """
    FOR market IN prediction_markets_polymarket
        FILTER market.question_embedding == null
        RETURN {
            market_id: market._key,
            question: market.question,
            description: market.description
        }
    """

    cursor = db.aql.execute(query)
    markets = list(cursor)
    df = pd.DataFrame(markets)

    print(f"✓ Found {len(df):,} markets without embeddings")
    return df

def generate_and_save_embeddings(db, markets_df, batch_size=100):
    """Generate embeddings and save each batch immediately"""
    print(f"\n[2/4] Generating embeddings in batches of {batch_size}...")
    print("=" * 80)

    if len(markets_df) == 0:
        print("No markets to process!")
        return

    # Prepare texts
    texts_to_embed = []
    for idx, row in markets_df.iterrows():
        question = str(row.get('question', '')).strip()
        description = str(row.get('description', '')).strip()

        # Combine question + description (first 500 chars)
        combined_text = question
        if description and description not in ['nan', 'None', '']:
            desc_snippet = description[:500]
            combined_text = f"{question} {desc_snippet}"

        texts_to_embed.append(combined_text)

    # Get OpenAI client
    client = get_openai_client()

    # Process in batches with progress bar
    total_batches = (len(texts_to_embed) - 1) // batch_size + 1
    successful_batches = 0
    failed_batches = 0

    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches", total=total_batches):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_indices = list(range(i, min(i+batch_size, len(texts_to_embed))))

        try:
            # Generate embeddings
            response = client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-small"
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]

            # Save to database immediately
            for idx, embedding in zip(batch_indices, embeddings):
                market_id = markets_df.iloc[idx]['market_id']
                db.aql.execute(
                    """
                    UPDATE {_key: @key}
                    WITH {question_embedding: @emb}
                    IN prediction_markets_polymarket
                    """,
                    bind_vars={'key': str(market_id), 'emb': embedding}
                )

            successful_batches += 1

        except Exception as e:
            print(f"\n✗ Batch {i//batch_size + 1} failed: {e}")
            failed_batches += 1
            continue

    print("\n" + "=" * 80)
    print(f"✓ Completed: {successful_batches} batches successful, {failed_batches} failed")
    print(f"✓ Processed ~{successful_batches * batch_size:,} markets")

def verify_embeddings(db):
    """Check final embedding status"""
    print("\n[3/4] Verifying embeddings...")

    query = """
    RETURN {
        total: LENGTH(prediction_markets_polymarket),
        with_embeddings: LENGTH(
            FOR m IN prediction_markets_polymarket
                FILTER m.question_embedding != null
                RETURN 1
        ),
        without_embeddings: LENGTH(
            FOR m IN prediction_markets_polymarket
                FILTER m.question_embedding == null
                RETURN 1
        )
    }
    """

    result = db.aql.execute(query).next()

    print(f"  Total markets: {result['total']:,}")
    print(f"  With embeddings: {result['with_embeddings']:,}")
    print(f"  Without embeddings: {result['without_embeddings']:,}")

    completion_pct = (result['with_embeddings'] / result['total'] * 100) if result['total'] > 0 else 0
    print(f"  Completion: {completion_pct:.1f}%")

    return result

def main():
    """Main execution"""
    print("=" * 80)
    print("POLYMARKET EMBEDDING GENERATOR - STANDALONE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script can be safely stopped and resumed at any time.")
    print("Progress is saved to the database after each batch.\n")

    try:
        # Connect to database
        print("[0/4] Connecting to ArangoDB...")
        db = get_arango_connection()
        print("✓ Connected")

        # Load markets needing embeddings
        markets_df = load_markets_without_embeddings(db)

        if len(markets_df) == 0:
            print("\n✓ All markets already have embeddings!")
            verify_embeddings(db)
            return

        # Generate and save embeddings
        generate_and_save_embeddings(db, markets_df, batch_size=100)

        # Verify results
        verify_embeddings(db)

        print("\n" + "=" * 80)
        print(f"✓ COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user - progress saved to database")
        print("Run this script again to resume from where you left off")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
