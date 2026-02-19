"""
SEC Exhibits Extraction Script
Parses full-submission.txt files, extracts exhibits, calculates sentiment, uploads to ArangoDB

Exhibits extracted:
- EX-10.* : Material contracts (credit agreements, employment, supply contracts)
- EX-4.*  : Debt instruments (bonds, notes, indentures)
- EX-99.* : Additional exhibits (press releases, financial statements)
- EX-21.* : Subsidiaries
- EX-31/32: SOX certifications (less valuable, optional)

Runtime: ~2-3 hours for 7,495 filings
"""
import os
import re
from pathlib import Path
from datetime import datetime
from arango import ArangoClient
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

load_dotenv()

# File paths
OLD_FORMS_DIR = Path(r"D:/Users/Ryan/Desktop/QUANT/tests/sec_filings/sec-edgar-filings/")
NEW_FORMS_DIR = Path(r"D:/Users/Ryan/Desktop/QUANT/tests/sec_filings/sec-edgar-filings-extended/")

# Batch configuration
BATCH_SIZE = 100  # Insert exhibits in batches of 100 (adjust up/down for speed vs memory)
PROCESS_BATCH = 50  # Process N filings before bulk inserting exhibits

# Exhibit types to extract (skip certifications unless you want them)
VALUABLE_EXHIBIT_TYPES = [
    'EX-10',   # Material contracts (MOST VALUABLE)
    'EX-4',    # Debt instruments
    'EX-99',   # Additional exhibits
    'EX-21',   # Subsidiaries
    # 'EX-31', # SOX certifications (skip - boilerplate)
    # 'EX-32', # SOX certifications (skip - boilerplate)
]

# Sentiment word lists (simple Loughran-McDonald)
NEGATIVE_WORDS = set([
    'loss', 'losses', 'decline', 'declined', 'decrease', 'decreased', 'decreasing',
    'adverse', 'adversely', 'fail', 'failed', 'failing', 'failure', 'risk', 'risks',
    'uncertain', 'uncertainty', 'negative', 'negatively', 'volatility', 'volatile',
    'default', 'defaulted', 'impair', 'impaired', 'impairment', 'litigation',
    'weakness', 'weaknesses', 'difficult', 'difficulty', 'challenges', 'challenging',
    'termination', 'breach', 'penalty', 'penalties', 'covenant', 'covenants'
])

POSITIVE_WORDS = set([
    'profit', 'profits', 'profitable', 'gain', 'gains', 'increase', 'increased',
    'growth', 'growing', 'improve', 'improved', 'improvement', 'strong', 'stronger',
    'opportunity', 'opportunities', 'success', 'successful', 'revenue', 'revenues',
    'earnings', 'achievement', 'achievements', 'favorable', 'favorably', 'positive',
    'extension', 'renewal', 'amended', 'expand', 'expansion'
])

# Keywords for contract classification
CONTRACT_KEYWORDS = {
    'credit_agreement': ['credit agreement', 'credit facility', 'loan agreement', 'revolving credit', 'term loan'],
    'employment': ['employment agreement', 'executive compensation', 'severance', 'change of control'],
    'supply': ['supply agreement', 'purchase agreement', 'vendor', 'supplier'],
    'partnership': ['joint venture', 'strategic alliance', 'partnership', 'collaboration'],
    'real_estate': ['lease agreement', 'real estate', 'property lease'],
    'acquisition': ['merger agreement', 'acquisition', 'purchase agreement', 'stock purchase'],
    'licensing': ['license agreement', 'intellectual property', 'patent', 'trademark'],
    'settlement': ['settlement agreement', 'consent decree', 'legal settlement']
}


def get_db():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_HOST', '')
    db_name = os.getenv('ARANGO_DATABASE', 'QUANT_v3')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)


def setup_collections(db):
    """Create sec_exhibits collection and edges"""
    print("\nSetting up collections...")

    # Document collection
    if not db.has_collection('sec_exhibits'):
        db.create_collection('sec_exhibits')
        print("  ✓ Created collection: sec_exhibits")
    else:
        print("  ⊙ Collection sec_exhibits already exists")

    # Edge collection
    if not db.has_collection('has_exhibit'):
        db.create_collection('has_exhibit', edge=True)
        print("  ✓ Created edge collection: has_exhibit")
    else:
        print("  ⊙ Edge collection has_exhibit already exists")

    # Add to graph
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        existing_edges = [ed['edge_collection'] for ed in graph.edge_definitions()]

        if 'has_exhibit' not in existing_edges:
            try:
                graph.create_edge_definition(
                    edge_collection='has_exhibit',
                    from_vertex_collections=['sec_filings'],
                    to_vertex_collections=['sec_exhibits']
                )
                print("  ✓ Added has_exhibit to graph")
            except Exception as e:
                print(f"  ⚠ Could not add has_exhibit: {e}")

    print()


def parse_documents_from_submission(file_path):
    """
    Parse full-submission.txt and extract all <DOCUMENT> sections

    Returns: List of {type, filename, description, text}
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  ✗ Error reading {file_path}: {e}")
        return []

    # Split by <DOCUMENT> tags
    documents = []
    doc_pattern = re.compile(
        r'<DOCUMENT>\s*'
        r'<TYPE>(.*?)\s*'
        r'<SEQUENCE>(.*?)\s*'
        r'<FILENAME>(.*?)\s*'
        r'(?:<DESCRIPTION>(.*?)\s*)?'
        r'<TEXT>\s*(.*?)\s*</TEXT>',
        re.DOTALL | re.IGNORECASE
    )

    for match in doc_pattern.finditer(content):
        doc_type = match.group(1).strip()
        sequence = match.group(2).strip()
        filename = match.group(3).strip()
        description = match.group(4).strip() if match.group(4) else doc_type
        text = match.group(5).strip()

        documents.append({
            'type': doc_type,
            'sequence': sequence,
            'filename': filename,
            'description': description,
            'text': text
        })

    return documents


def is_valuable_exhibit(doc_type):
    """Check if exhibit type is valuable (skip certifications)"""
    doc_type_upper = doc_type.upper()
    return any(doc_type_upper.startswith(vtype) for vtype in VALUABLE_EXHIBIT_TYPES)


def strip_html(text):
    """Remove HTML tags and extra whitespace"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove XBRL tags
    text = re.sub(r'<\?xml[^>]*\?>', '', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&#58;', ':')
    text = text.replace('&#59;', ';')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_sentiment(text):
    """Calculate sentiment scores based on word counts"""
    words = re.findall(r'\b\w+\b', text.lower())
    n_tokens = len(words)

    if n_tokens == 0:
        return {
            'n_tokens': 0,
            'negative_per_1k': 0,
            'positive_per_1k': 0,
            'finbert_score': 0.0
        }

    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)

    return {
        'n_tokens': n_tokens,
        'negative_per_1k': int((negative_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'positive_per_1k': int((positive_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'finbert_score': (positive_count - negative_count) / n_tokens if n_tokens > 0 else 0.0
    }


def classify_contract_type(text):
    """Classify contract by keywords"""
    text_lower = text.lower()

    for contract_type, keywords in CONTRACT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return contract_type

    return 'other'


def process_filing_exhibits(file_path, ticker, form_type, db):
    """Extract exhibits from a single SEC filing"""

    # Parse all documents from full-submission.txt
    documents = parse_documents_from_submission(file_path)

    if not documents:
        return 0

    # Find the filing document to get filing_key
    filing_key = None
    for doc in documents:
        if doc['type'] in ['10-K', '10-Q', '8-K', '4', '5', 'S-1', 'SC 13D', 'SC 13G', 'DEF 14A']:
            # Extract accession from path
            parts = file_path.parts
            accession = parts[-2]
            # Database uses "full-submission" not "full-submission_txt"
            filing_key = f"{ticker}_{form_type}_{accession}_full-submission"
            break

    if not filing_key:
        return 0

    # Check if filing exists in database
    filings_col = db.collection('sec_filings')
    if not filings_col.has(filing_key):
        # Filing doesn't exist - skip exhibits (run main ingestion first)
        return 0

    # Process exhibits
    exhibits_col = db.collection('sec_exhibits')
    edges_col = db.collection('has_exhibit')
    exhibits_added = 0

    for doc in documents:
        if not is_valuable_exhibit(doc['type']):
            continue

        # Extract and clean text
        raw_text = doc['text']
        clean_text = strip_html(raw_text)

        if len(clean_text) < 100:
            continue  # Skip empty/tiny exhibits

        # Create exhibit key
        exhibit_key = f"{filing_key}_{doc['type']}_{doc['sequence']}".replace('.', '_').replace(' ', '_')

        # Skip if already exists
        if exhibits_col.has(exhibit_key):
            continue

        # Calculate sentiment
        sentiment = calculate_sentiment(clean_text)

        # Classify contract type (for EX-10 exhibits)
        contract_type = classify_contract_type(clean_text) if doc['type'].upper().startswith('EX-10') else None

        # Truncate text for storage (keep first 50k chars, full text is too large)
        stored_text = clean_text[:50000] if len(clean_text) > 50000 else clean_text

        # Create exhibit document
        exhibit_doc = {
            '_key': exhibit_key,
            'ticker': ticker,
            'filing_type': form_type,
            'filing_key': filing_key,
            'exhibit_type': doc['type'],
            'sequence': doc['sequence'],
            'filename': doc['filename'],
            'description': doc['description'],
            'text': stored_text,
            'text_length': len(clean_text),
            'truncated': len(clean_text) > 50000,
            'contract_type': contract_type,
            'n_tokens': sentiment['n_tokens'],
            'negative_per_1k': sentiment['negative_per_1k'],
            'positive_per_1k': sentiment['positive_per_1k'],
            'finbert_score': sentiment['finbert_score'],
            'created_at': datetime.now().isoformat()
        }

        try:
            exhibits_col.insert(exhibit_doc)

            # Create edge from filing to exhibit
            edge_doc = {
                '_from': f"sec_filings/{filing_key}",
                '_to': f"sec_exhibits/{exhibit_key}",
                'exhibit_type': doc['type']
            }
            edges_col.insert(edge_doc)

            exhibits_added += 1

        except Exception as e:
            print(f"    ✗ Failed to insert exhibit {exhibit_key}: {e}")

    return exhibits_added


def process_all_filings(db, index_filter=None):
    """Process all filings and extract exhibits with batch inserts"""
    print("\nScanning for SEC filings...")

    # Find all full-submission.txt files
    all_files = []

    for forms_dir in [OLD_FORMS_DIR, NEW_FORMS_DIR]:
        if forms_dir.exists():
            for filing_file in forms_dir.rglob('full-submission.txt'):
                # Extract ticker and form type from path
                # Path: .../TICKER/FORM_TYPE/ACCESSION/full-submission.txt
                parts = filing_file.parts
                ticker = parts[-4]
                form_type = parts[-3]
                all_files.append((filing_file, ticker, form_type))

    print(f"Found {len(all_files)} filings")

    if index_filter:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from yahoo.constituents import get_tickers_from_arango
        allowed = set(get_tickers_from_arango(current_only=True, index=index_filter))
        all_files = [(f, t, ft) for f, t, ft in all_files if t in allowed]
        print(f"Filtered to {len(all_files)} filings (index: {index_filter})")

    print(f"Batch size: {BATCH_SIZE} exhibits per insert\n")

    if len(all_files) == 0:
        print("⚠ No filings found. Check directory paths.")
        return

    # Process filings with batching
    total_exhibits = 0
    exhibits_col = db.collection('sec_exhibits')
    edges_col = db.collection('has_exhibit')

    exhibit_batch = []
    edge_batch = []

    pbar = tqdm(all_files, desc="Processing filings", unit=" filings")

    for filing_file, ticker, form_type in pbar:
        try:
            exhibits_added = process_filing_exhibits(filing_file, ticker, form_type, db)
            total_exhibits += exhibits_added

            if exhibits_added > 0:
                pbar.set_postfix({'total_exhibits': total_exhibits})

        except Exception as e:
            print(f"\n  ✗ Error processing {ticker} {form_type}: {e}")

    pbar.close()

    print(f"\n✓ Processing complete!")
    print(f"  Total exhibits extracted: {total_exhibits:,}")


def verify_exhibits(db):
    """Verify exhibits were uploaded correctly"""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # Count exhibits by type
    query = """
    FOR e IN sec_exhibits
        COLLECT exhibit_type = e.exhibit_type WITH COUNT INTO count
        SORT count DESC
        RETURN {exhibit_type, count}
    """
    results = list(db.aql.execute(query))

    print("\nExhibits by type:")
    for r in results:
        print(f"  {r['exhibit_type']}: {r['count']:,}")

    # Count exhibits by contract type (EX-10 only)
    query = """
    FOR e IN sec_exhibits
        FILTER e.contract_type != null
        COLLECT contract_type = e.contract_type WITH COUNT INTO count
        SORT count DESC
        RETURN {contract_type, count}
    """
    results = list(db.aql.execute(query))

    print("\nContract types (EX-10):")
    for r in results:
        print(f"  {r['contract_type']}: {r['count']:,}")

    # Sample exhibits
    query = """
    FOR e IN sec_exhibits
        FILTER e.exhibit_type =~ "EX-10"
        LIMIT 5
        RETURN {
            ticker: e.ticker,
            type: e.exhibit_type,
            contract_type: e.contract_type,
            description: SUBSTRING(e.description, 0, 60),
            sentiment: e.finbert_score,
            length: e.text_length
        }
    """
    samples = list(db.aql.execute(query))

    print("\nSample EX-10 exhibits:")
    for s in samples:
        print(f"  [{s['ticker']}] {s['type']} - {s['contract_type']}")
        print(f"    {s['description']}")
        print(f"    Sentiment: {s['sentiment']:.3f}, Length: {s['length']:,} chars\n")

    print("=" * 80)
    print("✓ Exhibits ready for analysis!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Update backend/app/llm/prompts.py with exhibit examples")
    print("  2. Add sec_exhibits to frontend schema")
    print("  3. Test queries through LLM system")
    print("\nExample query:")
    print("""
    // Find credit agreements for defense contractors
    FOR c IN Company
      FILTER c.sector == "Industrials"
      FOR f IN OUTBOUND c HAS_FILING
        FOR e IN OUTBOUND f has_exhibit
          FILTER e.contract_type == "credit_agreement"
          SORT f.filing_date DESC
          LIMIT 10
          RETURN {
            ticker: c.ticker,
            company: c.company,
            filing_date: f.filing_date,
            exhibit: e.exhibit_type,
            description: e.description,
            sentiment: e.finbert_score
          }
    """)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract exhibits from SEC filings')
    parser.add_argument('--verify-only', action='store_true', help='Just run verification, skip extraction')
    parser.add_argument('--index', type=str, default=None, choices=['SP500', 'RUSSELL2000', 'NASDAQ100'],
                        help='Process only tickers in this index (from Company collection)')
    args = parser.parse_args()

    print("=" * 80)
    print("SEC EXHIBITS EXTRACTION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Connect to database
        db = get_db()
        print(f"✓ Connected to {os.getenv('ARANGO_DATABASE')}")

        if args.verify_only:
            verify_exhibits(db)
        else:
            # Setup collections
            setup_collections(db)

            # Process all filings (optionally filtered by index)
            process_all_filings(db, index_filter=args.index)

            # Verify
            verify_exhibits(db)

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✓ SUCCESS!")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("  Progress has been saved. Run again to continue from where you left off.")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
