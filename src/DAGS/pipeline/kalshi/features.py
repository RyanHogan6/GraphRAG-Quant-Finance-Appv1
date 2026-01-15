"""Kalshi feature engineering with embeddings"""
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_client():
    """Get or create OpenAI client lazily"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Temporarily remove proxy environment variables to avoid httpx conflicts
    old_http_proxy = os.environ.pop('HTTP_PROXY', None)
    old_https_proxy = os.environ.pop('HTTPS_PROXY', None)
    old_http_proxy_lower = os.environ.pop('http_proxy', None)
    old_https_proxy_lower = os.environ.pop('https_proxy', None)

    try:
        client = OpenAI(api_key=api_key)
        return client
    finally:
        # Restore proxy settings
        if old_http_proxy:
            os.environ['HTTP_PROXY'] = old_http_proxy
        if old_https_proxy:
            os.environ['HTTPS_PROXY'] = old_https_proxy
        if old_http_proxy_lower:
            os.environ['http_proxy'] = old_http_proxy_lower
        if old_https_proxy_lower:
            os.environ['https_proxy'] = old_https_proxy_lower

def generate_title_embeddings(markets_df: pd.DataFrame, batch_size=100) -> pd.DataFrame:
    """Generate embeddings for Kalshi market titles"""
    print("\n[EMBEDDINGS] Generating title embeddings...")

    if len(markets_df) == 0:
        return markets_df

    df = markets_df.copy()

    # Check existing embeddings
    if 'title_embedding' in df.columns:
        needs_embedding = df['title_embedding'].isna()
        to_embed_count = needs_embedding.sum()
        if to_embed_count == 0:
            print(f"[SKIP] All {len(df)} markets have embeddings")
            return df
        df_to_embed = df[needs_embedding].copy()
    else:
        df['title_embedding'] = None
        df_to_embed = df.copy()

    # Prepare texts
    texts = [str(row['title']).strip() for _, row in df_to_embed.iterrows()]

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"[BATCH {i//batch_size + 1}] Embedding {len(batch)} markets...")

        try:
            client = get_openai_client()
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
            all_embeddings.extend([None] * len(batch))

    # Assign embeddings back to dataframe
    for i, idx in enumerate(df_to_embed.index):
        if i < len(all_embeddings):
            df.at[idx, 'title_embedding'] = all_embeddings[i]

    print(f"[OK] Generated {df['title_embedding'].notna().sum()}/{len(df)} embeddings")
    return df

def engineer_market_features(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for Kalshi markets"""
    print("\n[FEATURES] Engineering Kalshi market features...")

    df = markets_df.copy()

    # Generate embeddings
    df = generate_title_embeddings(df)

    print(f"[OK] Engineered features for {len(df)} markets")
    return df
