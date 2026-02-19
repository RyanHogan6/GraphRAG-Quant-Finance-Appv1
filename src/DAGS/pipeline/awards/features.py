"""
Awards Feature Engineering - FinBERT Embeddings
"""
import torch
from transformers import AutoTokenizer

# Use explicit BertModel import so model loading works across transformers versions
# (AutoModel.from_pretrained can fail with "Could not find BertModel" on some installs)
try:
    from transformers import BertModel
except ImportError:
    from transformers.models.bert.modeling_bert import BertModel

_tokenizer = None
_model = None

def _get_finbert():
    """Lazy load FinBERT model"""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # faster conv-style ops
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = BertModel.from_pretrained("ProsusAI/finbert").to(device)
        _model.eval()
        # Log so user knows GPU is used
        import logging
        logging.getLogger(__name__).info("FinBERT embeddings device: %s", device)

    return _tokenizer, _model

def generate_embeddings(df, batch_size=None):
    """Generate FinBERT embeddings for award descriptions.
    batch_size: None = auto (256 on GPU, 64 on CPU), or set explicitly."""
    if df.empty or 'Description' not in df.columns:
        return df

    tokenizer, model = _get_finbert()
    device = next(model.parameters()).device
    if batch_size is None:
        batch_size = 256 if device.type == 'cuda' else 64

    descriptions = df['Description'].fillna('').tolist()
    embeddings = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        valid_batch = [d if d and len(d.strip()) > 10 else "" for d in batch]

        if not any(valid_batch):
            embeddings.extend([None] * len(batch))
            continue

        inputs = tokenizer(valid_batch, truncation=True, padding=True,
                          return_tensors="pt", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()
            embeddings.extend(batch_embeddings)

    df['description_embedding'] = embeddings
    return df
