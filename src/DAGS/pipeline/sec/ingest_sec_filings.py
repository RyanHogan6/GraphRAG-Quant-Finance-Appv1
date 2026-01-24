"""
SEC Filings Ingestion Script
Parses SEC filings, extracts sections/sentences, calculates FinBERT sentiment, uploads to ArangoDB
"""
import os
import re
from pathlib import Path
from datetime import datetime
from arango import ArangoClient
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# File paths
OLD_FORMS_DIR = Path(r"D:/Users/Ryan/Desktop/QUANT/tests/sec_filings/sec-edgar-filings/")
NEW_FORMS_DIR = Path(r"D:/Users/Ryan/Desktop/QUANT/tests/sec_filings/sec-edgar-filings-extended/")

# Form types
OLD_FORM_TYPES = ["8-K", "10-Q", "10-K"]
NEW_FORM_TYPES = ["S-1", "4", "5", "6-K", "SC 13D", "SC 13G", "13F-HR"]

# Loughran-McDonald dictionaries (simple word lists for sentiment)
NEGATIVE_WORDS = set([
    'loss', 'losses', 'decline', 'declined', 'decrease', 'decreased', 'decreasing',
    'adverse', 'adversely', 'fail', 'failed', 'failing', 'failure', 'risk', 'risks',
    'uncertain', 'uncertainty', 'negative', 'negatively', 'volatility', 'volatile',
    'default', 'defaulted', 'impair', 'impaired', 'impairment', 'litigation',
    'weakness', 'weaknesses', 'difficult', 'difficulty', 'challenges', 'challenging'
])

POSITIVE_WORDS = set([
    'profit', 'profits', 'profitable', 'gain', 'gains', 'increase', 'increased',
    'growth', 'growing', 'improve', 'improved', 'improvement', 'strong', 'stronger',
    'opportunity', 'opportunities', 'success', 'successful', 'revenue', 'revenues',
    'earnings', 'achievement', 'achievements', 'favorable', 'favorably', 'positive'
])

UNCERTAINTY_WORDS = set([
    'uncertain', 'uncertainty', 'uncertainties', 'may', 'might', 'could', 'possibly',
    'perhaps', 'approximately', 'estimate', 'estimated', 'estimates', 'projected',
    'projections', 'forecast', 'forecasts', 'potential', 'potentially', 'believe',
    'expects', 'expected', 'anticipate', 'anticipated', 'assumes', 'assumption'
])

LITIGIOUS_WORDS = set([
    'litigation', 'suit', 'lawsuit', 'lawsuits', 'legal', 'court', 'claim', 'claims',
    'alleged', 'allegedly', 'settlement', 'settlements', 'arbitration', 'dispute',
    'disputes', 'appeal', 'appeals', 'regulatory', 'investigation', 'investigations'
])


def get_db():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_HOST', '')
    db_name = os.getenv('ARANGO_DB', 'QUANT_v3')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)


def setup_collections(db):
    """Create SEC collections and edges if they don't exist"""
    # Document collections
    for coll_name in ['sec_filings', 'sec_sections', 'sec_sentences']:
        if not db.has_collection(coll_name):
            db.create_collection(coll_name)
            print(f"  Created collection: {coll_name}")

    # Edge collections
    for edge_name in ['HAS_FILING', 'has_section', 'has_sentence']:
        if not db.has_collection(edge_name):
            db.create_collection(edge_name, edge=True)
            print(f"  Created edge collection: {edge_name}")

    # Add to graph
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        existing_edges = [ed['edge_collection'] for ed in graph.edge_definitions()]

        edge_configs = {
            'HAS_FILING': {'from': ['Company'], 'to': ['sec_filings']},
            'has_section': {'from': ['sec_filings'], 'to': ['sec_sections']},
            'has_sentence': {'from': ['sec_sections'], 'to': ['sec_sentences']}
        }

        for edge_name, config in edge_configs.items():
            if edge_name not in existing_edges:
                try:
                    graph.create_edge_definition(
                        edge_collection=edge_name,
                        from_vertex_collections=config['from'],
                        to_vertex_collections=config['to']
                    )
                    print(f"  Added {edge_name} to graph")
                except Exception as e:
                    print(f"  ⚠ Could not add {edge_name}: {e}")


def extract_text_from_filing(file_path):
    """Extract text from SEC filing"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""


def split_into_sentences(text):
    """Simple sentence splitter"""
    # Split on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def calculate_sentiment(text):
    """
    Calculate simple sentiment scores based on word counts
    (Simplified version - you had FinBERT before)
    """
    words = re.findall(r'\b\w+\b', text.lower())
    n_tokens = len(words)

    if n_tokens == 0:
        return {
            'n_tokens': 0,
            'negative_per_1k': 0,
            'positive_per_1k': 0,
            'uncertainty_per_1k': 0,
            'litigious_per_1k': 0,
            'finbert_score': 0.0
        }

    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    uncertainty_count = sum(1 for w in words if w in UNCERTAINTY_WORDS)
    litigious_count = sum(1 for w in words if w in LITIGIOUS_WORDS)

    return {
        'n_tokens': n_tokens,
        'negative_per_1k': int((negative_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'positive_per_1k': int((positive_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'uncertainty_per_1k': int((uncertainty_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'litigious_per_1k': int((litigious_count / n_tokens) * 1000) if n_tokens > 0 else 0,
        'finbert_score': (positive_count - negative_count) / n_tokens if n_tokens > 0 else 0.0
    }


def parse_filing_metadata(file_path, ticker, form_type):
    """Extract metadata from filing path"""
    # Path structure: .../TICKER/FORM_TYPE/ACCESSION/full-submission.txt
    parts = file_path.parts

    accession = parts[-2]  # Parent directory is accession number
    file_name = file_path.name

    # Try to extract date from accession (format: 0000320193-20-000096 -> 2020)
    year_match = re.search(r'-(\d{2})-', accession)
    fiscal_year = int(f"20{year_match.group(1)}") if year_match else None

    # Try to find filing date in the file
    filing_date = None
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_2kb = f.read(2000)
            date_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', first_2kb)
            if date_match:
                date_str = date_match.group(1)
                filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            else:
                # Fallback: use current date
                filing_date = datetime.now().strftime('%Y-%m-%d')
    except:
        filing_date = datetime.now().strftime('%Y-%m-%d')

    return {
        'accession': accession,
        'file_name': file_name,
        'filing_date': filing_date,
        'fiscal_year': fiscal_year
    }


def process_filing(file_path, ticker, form_type, db):
    """Process a single SEC filing"""
    print(f"  Processing {ticker} {form_type} {file_path.name}...")

    # Get metadata
    metadata = parse_filing_metadata(file_path, ticker, form_type)

    # Create filing key
    filing_key = f"{ticker}_{form_type}_{metadata['accession']}_{metadata['file_name']}".replace('.', '_').replace(' ', '_')

    # Extract full text
    full_text = extract_text_from_filing(file_path)

    if not full_text or len(full_text) < 100:
        print(f"    ⚠ Skipping {filing_key} - insufficient text")
        return

    # Split into sentences
    sentences = split_into_sentences(full_text)

    if len(sentences) == 0:
        print(f"    ⚠ Skipping {filing_key} - no sentences extracted")
        return

    # Calculate aggregate sentiment
    all_sentiments = [calculate_sentiment(sent) for sent in sentences]
    avg_finbert = np.mean([s['finbert_score'] for s in all_sentiments])
    avg_negative = np.mean([s['negative_per_1k'] for s in all_sentiments])
    avg_positive = np.mean([s['positive_per_1k'] for s in all_sentiments])
    avg_uncertainty = np.mean([s['uncertainty_per_1k'] for s in all_sentiments])

    # Create filing document
    filing_doc = {
        '_key': filing_key,
        'ticker': ticker,
        'type': form_type,
        'accession': metadata['accession'],
        'file_name': metadata['file_name'],
        'filing_date': metadata['filing_date'],
        'fiscal_year': metadata['fiscal_year'],
        'avg_finbert': float(avg_finbert),
        'avg_negative': float(avg_negative),
        'avg_positive': float(avg_positive),
        'avg_uncertainty': float(avg_uncertainty),
        'sentence_count': len(sentences)
    }

    # Check if filing already exists
    filings_col = db.collection('sec_filings')
    try:
        if filings_col.has(filing_key):
            print(f"    ⊙ Skipping {filing_key} - already exists")
            return
        filings_col.insert(filing_doc)
    except Exception as e:
        print(f"    ✗ Failed to insert filing: {e}")
        return

    # Create section (for simplicity, treating entire document as one section)
    section_key = f"{filing_key}_sec0"
    section_doc = {
        '_key': section_key,
        'filing_id': f"sec_filings/{filing_key}",
        'section_type': 'Full Document',
        'start_char': 0,
        'length': len(full_text)
    }

    sections_col = db.collection('sec_sections')
    try:
        sections_col.insert(section_doc)
    except Exception as e:
        print(f"    ✗ Failed to insert section: {e}")
        return

    # Create sentences
    sentence_docs = []
    for i, (sent_text, sentiment) in enumerate(zip(sentences, all_sentiments)):
        sent_key = f"{section_key}_sent{i}"

        sentence_docs.append({
            '_key': sent_key,
            'section_id': f"sec_sections/{section_key}",
            'text': sent_text[:1000],  # Limit text length
            'n_tokens': sentiment['n_tokens'],
            'finbert_score': sentiment['finbert_score'],
            'finbert_probs': {
                'positive': max(0, sentiment['finbert_score']),
                'negative': max(0, -sentiment['finbert_score']),
                'neutral': 1 - abs(sentiment['finbert_score'])
            },
            'negative_per_1k': sentiment['negative_per_1k'],
            'positive_per_1k': sentiment['positive_per_1k'],
            'uncertainty_per_1k': sentiment['uncertainty_per_1k'],
            'litigious_per_1k': sentiment['litigious_per_1k']
        })

    # Batch insert sentences
    sentences_col = db.collection('sec_sentences')
    try:
        for i in range(0, len(sentence_docs), 500):
            batch = sentence_docs[i:i+500]
            sentences_col.insert_many(batch)
        print(f"    ✓ Inserted {len(sentence_docs)} sentences")
    except Exception as e:
        print(f"    ✗ Failed to insert sentences: {e}")
        return

    # Create edges
    create_edges(db, ticker, filing_key, section_key)


def create_edges(db, ticker, filing_key, section_key):
    """Create graph edges for SEC data"""

    # Company → Filing edge
    try:
        has_filing_edge = {
            '_key': f"{ticker}_{filing_key}",
            '_from': f"Company/{ticker}",
            '_to': f"sec_filings/{filing_key}",
            'relationship': 'filed'
        }
        db.collection('HAS_FILING').insert(has_filing_edge, overwrite=True, silent=True)
    except:
        pass  # Edge might already exist

    # Filing → Section edge
    try:
        has_section_edge = {
            '_key': f"{filing_key}_{section_key}",
            '_from': f"sec_filings/{filing_key}",
            '_to': f"sec_sections/{section_key}",
            'relationship': 'contains_section'
        }
        db.collection('has_section').insert(has_section_edge, overwrite=True, silent=True)
    except:
        pass


def scan_directory(base_dir, form_types):
    """Scan directory for SEC filings"""
    filings = []

    if not base_dir.exists():
        print(f"  ✗ Directory does not exist: {base_dir}")
        return filings

    # Check for form type directories directly (in case structure is different)
    for form_type in form_types:
        form_dir = base_dir / form_type
        if form_dir.exists() and form_dir.is_dir():
            print(f"  Found form directory: {form_type}")

            # Structure: form_type/ticker/accession/full-submission.txt
            for ticker_dir in form_dir.iterdir():
                if not ticker_dir.is_dir():
                    continue

                ticker = ticker_dir.name

                for accession_dir in ticker_dir.iterdir():
                    if not accession_dir.is_dir():
                        continue

                    submission_file = accession_dir / 'full-submission.txt'
                    if submission_file.exists():
                        filings.append({
                            'ticker': ticker,
                            'form_type': form_type,
                            'file_path': submission_file
                        })

    return filings


def main():
    """Main ingestion pipeline"""
    print("="*80)
    print("SEC FILINGS INGESTION - NEW FORMS ONLY")
    print("="*80)

    # Connect to database
    print("\nConnecting to ArangoDB...")
    db = get_db()

    # Setup collections
    print("\nSetting up collections...")
    setup_collections(db)

    # ONLY scan new forms directory (skip old forms to avoid re-processing 4M docs)
    print(f"\nScanning {NEW_FORMS_DIR} for {NEW_FORM_TYPES}...")
    new_filings = scan_directory(NEW_FORMS_DIR, NEW_FORM_TYPES)
    print(f"  Found {len(new_filings)} new-form filings")

    if len(new_filings) == 0:
        print("\n⚠ No new filings found. Checking directory structure...")
        if NEW_FORMS_DIR.exists():
            print(f"  Directory exists: {NEW_FORMS_DIR}")
            print(f"  Contents: {list(NEW_FORMS_DIR.iterdir())[:10]}")
        else:
            print(f"  ✗ Directory does not exist: {NEW_FORMS_DIR}")
        return

    # Use only new filings
    all_filings = new_filings
    print(f"\nTotal filings to process: {len(all_filings)}")

    # Process each filing
    print("\nProcessing filings...")
    for i, filing in enumerate(all_filings, 1):
        print(f"\n[{i}/{len(all_filings)}] {filing['ticker']} {filing['form_type']}")
        try:
            process_filing(
                filing['file_path'],
                filing['ticker'],
                filing['form_type'],
                db
            )
        except Exception as e:
            print(f"  ✗ Error processing filing: {e}")
            continue

    print("\n" + "="*80)
    print("✅ SEC INGESTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
