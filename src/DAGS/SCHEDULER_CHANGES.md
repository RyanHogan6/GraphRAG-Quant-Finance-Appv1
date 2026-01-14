# Scheduler Architecture Changes

## Summary

The scheduler has been restructured to run multiple pipelines in sequence and handle embedding generation separately to avoid Railway timeouts.

## New Pipeline Order

**Railway Scheduler (runs every 12 hours):**
1. **Yahoo MarketData** (placeholder - not yet implemented)
2. **Kalshi** (placeholder - not yet implemented)
3. **Polymarket** (✓ active, **skips embeddings**)

## Why Skip Embeddings in Scheduler?

**Problem:** Embedding generation was breaking at batch 104/240 on Railway due to:
- Time limits (~10-15 min execution timeout)
- Memory constraints
- Lost all progress on restart (0 embeddings saved)

**Solution:**
- Scheduler skips embeddings entirely
- Run `generate_embeddings_standalone.py` **locally** instead
- Incremental saves after each batch (100 markets)
- Resumable - can stop/restart without losing progress

## Files Changed

### 1. `generate_embeddings_standalone.py` (NEW)
Standalone script for local embedding generation:
- Loads markets without embeddings from ArangoDB
- Generates embeddings in batches of 100
- Saves each batch immediately to database
- Shows progress bar with estimated completion
- Fully resumable (interrupt with Ctrl+C, resume by running again)

**Usage:**
```bash
cd src/DAGS
python generate_embeddings_standalone.py
```

### 2. `scheduler/app.py` (MODIFIED)
Restructured to support multiple pipelines:
- `run_yahoo_pipeline()` - Placeholder for Yahoo implementation
- `run_kalshi_pipeline()` - Placeholder for Kalshi implementation
- `run_polymarket_pipeline(skip_embeddings)` - Existing Polymarket pipeline
- `run_pipeline()` - Master orchestrator that runs all three in sequence

### 3. `pipeline/polymarket/features.py` (MODIFIED)
Added embedding skip capability:
- `engineer_market_features(db=None, skip_embeddings=False)` - New parameter
- `generate_market_embeddings(db=None)` - Accepts db for incremental saves
- Loads existing embeddings from database at start (crash recovery)
- Saves each batch immediately after generation

### 4. `pipeline/kalshi/downloader.py` (MODIFIED)
Fixed probability fields:
- Changed from `yes_bid`/`no_bid` (don't exist) to `last_price`
- Converts to `yes_probability`/`no_probability` (0-1 range)

### 5. `pipeline/kalshi/arango_uploader.py` (MODIFIED)
Updated field names to match downloader changes

### 6. `backend/app/llm/prompts.py` (MODIFIED)
Updated Kalshi schema to reflect correct field names

## Current Status

### Embeddings
- **Total markets:** 45,054
- **With embeddings:** 10,249 (22.7%)
- **Without embeddings:** 34,805 (77.3%)

**Next Step:** Run `generate_embeddings_standalone.py` locally to complete remaining 34,805 markets

### Scheduler
- Runs every 12 hours on Railway
- Executes: Yahoo (placeholder) → Kalshi (placeholder) → Polymarket (no embeddings)
- Estimated runtime: 5-10 minutes (without embeddings)

## Workflow Going Forward

### Initial Setup (One Time)
```bash
# Run locally to generate all embeddings
cd src/DAGS
python generate_embeddings_standalone.py
# Takes ~45-60 minutes for 34,805 markets
# Can be interrupted and resumed
```

### Regular Operations
- **Scheduler (Railway):** Runs automatically every 12 hours
  - Fetches new markets
  - Updates existing markets
  - Saves price snapshots
  - Builds graph edges (every 6 hours)

- **Embeddings (Local):** Run manually when needed
  - Once per week recommended
  - Only processes new markets (incremental)
  - Takes 1-5 minutes for typical weekly additions

## Deployment

Commit these changes:
```bash
git add src/DAGS/generate_embeddings_standalone.py
git add src/DAGS/README_EMBEDDINGS.md
git add src/DAGS/SCHEDULER_CHANGES.md
git add src/scheduler/app.py
git add src/DAGS/pipeline/polymarket/features.py
git add src/DAGS/pipeline/kalshi/downloader.py
git add src/DAGS/pipeline/kalshi/arango_uploader.py
git add backend/app/llm/prompts.py

git commit -m "refactor: separate embedding generation + multi-pipeline scheduler

- Create standalone embedding script with crash recovery
- Restructure scheduler for Yahoo → Kalshi → Polymarket order
- Skip embeddings in Railway scheduler (timeout issues)
- Fix Kalshi probability fields (yes_probability/no_probability)
- Add incremental embedding saves to database"

git push origin main
```

## Benefits

✅ **No more lost progress** - Embeddings save after each batch
✅ **Faster scheduler runs** - Completes in 5-10 min (vs 45+ min timeout)
✅ **Resumable embeddings** - Stop/start anytime without losing work
✅ **Multi-pipeline support** - Easy to add Yahoo/Kalshi implementations
✅ **Cost efficient** - Only embeds new markets (incremental updates)
✅ **Local control** - Run embeddings when you want, not on Railway's schedule
