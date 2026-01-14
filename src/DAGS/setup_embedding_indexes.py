"""
Setup ArangoDB Indexes for Embedding Fields
Run this ONCE after first embedding generation to optimize COSINE_SIMILARITY queries

Usage:
    python setup_embedding_indexes.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline', 'polymarket'))

from pipeline.polymarket.arango_uploader import get_arango_connection


def create_embedding_indexes():
    """
    Create persistent indexes on embedding fields for faster similarity search.

    Index types:
    - Persistent index: Fast lookups for non-null embedding fields

    Performance impact:
    - Without index: O(n) scan of all documents
    - With index: O(log n) lookup + O(k) similarity computation

    Expected speedup: 5-10x for queries on large collections
    """

    print("\n" + "="*80)
    print("ARANGODB EMBEDDING INDEX SETUP")
    print("="*80)

    # Get database connection
    try:
        db = get_arango_connection()
        print("\n✓ Connected to ArangoDB")
    except Exception as e:
        print(f"\n✗ Failed to connect to ArangoDB: {e}")
        return False

    indexes_created = []
    indexes_existed = []
    errors = []

    # Define indexes to create
    index_definitions = [
        {
            "collection": "prediction_markets_polymarket",
            "field": "question_embedding",
            "index_name": "idx_question_embedding",
            "description": "Polymarket question embeddings for semantic search"
        },
        {
            "collection": "Award",
            "field": "description_embedding",
            "index_name": "idx_description_embedding",
            "description": "Award description embeddings for semantic search"
        }
    ]

    # Create indexes
    for idx_def in index_definitions:
        collection_name = idx_def["collection"]
        field = idx_def["field"]
        index_name = idx_def["index_name"]
        description = idx_def["description"]

        print(f"\n[{collection_name}] Creating index '{index_name}' on '{field}'...")
        print(f"  Purpose: {description}")

        try:
            # Get collection
            if not db.has_collection(collection_name):
                print(f"  ⚠ WARNING: Collection '{collection_name}' does not exist! Skipping...")
                errors.append(f"{collection_name}: Collection not found")
                continue

            collection = db.collection(collection_name)

            # Check if index already exists
            existing_indexes = collection.indexes()
            index_exists = any(
                idx.get('name') == index_name
                for idx in existing_indexes
            )

            if index_exists:
                print(f"  ℹ Index '{index_name}' already exists. Skipping...")
                indexes_existed.append(index_name)
                continue

            # Create persistent index
            collection.add_persistent_index(
                fields=[field],
                unique=False,
                name=index_name
            )

            print(f"  ✓ Index '{index_name}' created successfully!")
            indexes_created.append(index_name)

        except Exception as e:
            print(f"  ✗ Failed to create index: {e}")
            errors.append(f"{collection_name}.{field}: {str(e)}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if indexes_created:
        print(f"\n✓ Created {len(indexes_created)} new indexes:")
        for idx_name in indexes_created:
            print(f"  - {idx_name}")

    if indexes_existed:
        print(f"\n ℹ {len(indexes_existed)} indexes already existed:")
        for idx_name in indexes_existed:
            print(f"  - {idx_name}")

    if errors:
        print(f"\n✗ {len(errors)} errors occurred:")
        for error in errors:
            print(f"  - {error}")

    total_success = len(indexes_created) + len(indexes_existed)
    total_attempts = total_success + len(errors)

    print(f"\n[RESULT] {total_success}/{total_attempts} indexes ready")
    print("="*80 + "\n")

    return len(errors) == 0


def verify_embeddings_exist():
    """
    Verify that embeddings actually exist in the database before creating indexes.
    """

    print("\n" + "="*80)
    print("VERIFYING EMBEDDINGS")
    print("="*80)

    try:
        db = get_arango_connection()
    except Exception as e:
        print(f"\n✗ Failed to connect: {e}")
        return False

    checks = [
        {
            "collection": "prediction_markets_polymarket",
            "field": "question_embedding",
            "query": "FOR doc IN prediction_markets_polymarket FILTER doc.question_embedding != null LIMIT 1 RETURN 1"
        },
        {
            "collection": "Award",
            "field": "description_embedding",
            "query": "FOR doc IN Award FILTER doc.description_embedding != null LIMIT 1 RETURN 1"
        }
    ]

    all_verified = True

    for check in checks:
        collection = check["collection"]
        field = check["field"]
        query = check["query"]

        print(f"\n[{collection}] Checking for '{field}' embeddings...")

        try:
            result = list(db.aql.execute(query))

            if result:
                print(f"  ✓ Embeddings found! Ready for indexing.")
            else:
                print(f"  ✗ WARNING: No embeddings found in {collection}.{field}")
                print(f"  ℹ Run the Airflow DAG to generate embeddings first:")
                print(f"    airflow dags trigger polymarket_etl_pipeline")
                all_verified = False

        except Exception as e:
            print(f"  ✗ Error checking embeddings: {e}")
            all_verified = False

    print("\n" + "="*80)
    return all_verified


if __name__ == "__main__":
    print("\n🔧 ArangoDB Embedding Index Setup Script")
    print("=" * 80)

    # Step 1: Verify embeddings exist
    print("\nStep 1: Verifying embeddings exist in database...")
    embeddings_ready = verify_embeddings_exist()

    if not embeddings_ready:
        print("\n⚠ WARNING: Some embeddings are missing!")
        print("You can still create indexes, but they won't improve performance until embeddings are generated.")

        response = input("\nContinue anyway? (y/N): ").lower()
        if response != 'y':
            print("\n❌ Aborted. Generate embeddings first, then run this script again.")
            sys.exit(1)

    # Step 2: Create indexes
    print("\nStep 2: Creating indexes...")
    success = create_embedding_indexes()

    # Exit code
    if success:
        print("\n✅ Setup complete! Semantic search queries will now be faster.")
        print("\nNext steps:")
        print("1. Test a semantic query via backend API:")
        print('   curl -X POST http://localhost:8000/api/query/execute -H "Content-Type: application/json" -d \'{"question": "Find prediction markets about artificial intelligence"}\'')
        print("\n2. Monitor query performance in backend logs")
        print("3. Run analytics to compare before/after performance")
        sys.exit(0)
    else:
        print("\n⚠ Setup completed with errors. Check output above for details.")
        sys.exit(1)
