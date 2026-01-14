"""Kalshi feature engineering with embeddings"""
import pandas as pd
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
            all_embeddings.extend([None] * len(batch))

    df.loc[df_to_embed.index, 'title_embedding'] = all_embeddings
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
