# Generating Polymarket Embeddings Locally

The embedding generation process has been moved out of the Railway scheduler due to timeouts and memory constraints. Instead, run this standalone script locally to generate embeddings.

## Why Run Locally?

- **Crash Recovery**: Railway's time/memory limits cause interruptions at batch ~104/240
- **Incremental Progress**: Script saves to database after every batch
- **Resumable**: Can stop and resume at any time - progress is preserved
- **Cost Efficiency**: Only generates embeddings for new markets (skips existing)

## Prerequisites

1. Python 3.11+ with required packages:
   ```bash
   pip install pandas python-arango openai python-dotenv tqdm
   ```

2. Environment variables (create `.env` file in DAGS directory):
   ```
   OPENAI_API_KEY=sk-proj-...
   ARANGO_HOST=http://your-arango-host:8529
   ARANGO_DATABASE=QUANT_v3
   ARANGO_USERNAME=root
   ARANGO_PASSWORD=your_password
   ```

## Running the Script

```bash
cd src/DAGS
python generate_embeddings_standalone.py
```

## What It Does

1. **Loads Markets** - Fetches all Polymarket markets without embeddings from ArangoDB
2. **Generates Embeddings** - Processes in batches of 100 using OpenAI API
3. **Saves Incrementally** - Each batch is saved to database immediately
4. **Shows Progress** - Progress bar shows current batch and completion %

## Example Output

```
================================================================================
POLYMARKET EMBEDDING GENERATOR - STANDALONE
================================================================================
Started: 2026-01-14 15:30:00

This script can be safely stopped and resumed at any time.
Progress is saved to the database after each batch.

[0/4] Connecting to ArangoDB...
✓ Connected

[1/4] Loading markets from ArangoDB...
✓ Found 34,805 markets without embeddings

[2/4] Generating embeddings in batches of 100...
================================================================================
Embedding batches: 100%|████████████████| 348/348 [45:23<00:00, 7.82s/it]

================================================================================
✓ Completed: 348 batches successful, 0 failed
✓ Processed ~34,800 markets

[3/4] Verifying embeddings...
  Total markets: 45,054
  With embeddings: 45,054
  Without embeddings: 0
  Completion: 100.0%

================================================================================
✓ COMPLETE - 2026-01-14 16:15:23
================================================================================
```

## Interrupting and Resuming

**To stop the script:**
- Press `Ctrl+C` - progress is automatically saved

**To resume:**
- Just run the script again
- It will skip markets that already have embeddings
- Only processes remaining markets

## Cost Estimation

- **Model**: text-embedding-3-small
- **Cost**: ~$0.02 per 1,000 markets
- **Total for 45,000 markets**: ~$0.90

## Verifying Results in ArangoDB

After completion, verify in ArangoDB web UI:

```aql
RETURN {
  total: LENGTH(prediction_markets_polymarket),
  with_embeddings: LENGTH(
    FOR m IN prediction_markets_polymarket
      FILTER m.question_embedding != null
      RETURN 1
  )
}
```

Expected: Both numbers should match (e.g., 45,054 / 45,054)

## Troubleshooting

**"No module named 'openai'"**
```bash
pip install openai==1.58.1
```

**"OPENAI_API_KEY environment variable not set"**
- Check your `.env` file exists in the DAGS directory
- Ensure `OPENAI_API_KEY` is set correctly

**"Connection to ArangoDB failed"**
- Verify ArangoDB is running and accessible
- Check host/port/credentials in `.env` file

**Embeddings not appearing in database**
- Check Railway logs - scheduler may be overwriting them
- Wait for scheduler to complete its cycle before checking

## After Running

Once embeddings are complete:
1. Verify completion in ArangoDB (query above)
2. Test semantic search in your app (e.g., "markets about AI")
3. Future scheduler runs will skip embedding generation (fast!)

## Schedule

- **Railway Scheduler**: Runs every 12 hours (no embeddings)
- **This Script**: Run manually when needed (new markets added)
- **Recommended**: Run this script once per week to catch new markets
