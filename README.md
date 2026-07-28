# KARGA Markets

**AI-powered natural language interface to a multi-source financial knowledge graph.**

Ask questions in plain English across stocks, government contracts, SEC filings, prediction markets, commodities, and more — KARGA translates them into graph queries (AQL), runs them against ArangoDB, and synthesizes the results.

> Traditional platforms keep financial data in silos. KARGA connects it so you can ask things like: *"Which S&P 500 companies with significant government contracts are mentioned in prediction markets?"*

---

## Features

- **Natural language → graph query** — GPT-4 plans and generates AQL; results come from the graph, not document retrieval
- **Hybrid answers** — graph data + optional Perplexity web search for current events
- **Interactive UI** — charts, tables, graph explorer, company workups, whale tracker, query builder
- **Alpha signals** — contract momentum, options–filing convergence, graph centrality
- **Automated ETL** — scheduled pipelines keep Polymarket, Kalshi, Yahoo, FRED, CFTC, EIA, CME, awards, options, and SEC data fresh

---

## Architecture

```
┌─────────────────────┐
│  Next.js Frontend   │  Natural language UI, charts, graph viz
└──────────┬──────────┘
           │ REST
┌──────────▼──────────┐
│  FastAPI Backend    │  Intent → AQL planning → execution → synthesis
└──────────┬──────────┘
           │ AQL
┌──────────▼──────────┐
│  ArangoDB Graph     │  Companies ↔ market data ↔ awards ↔ SEC ↔
│  (QUANT_v3)         │  prediction markets ↔ commodities ↔ options
└─────────────────────┘
           ▲
┌──────────┴──────────┐
│  ETL Scheduler      │  APScheduler pipelines (Railway)
└─────────────────────┘
```

| Layer | Stack |
|-------|--------|
| Frontend | Next.js, React, TypeScript, Tailwind, Framer Motion, React Flow |
| Backend | FastAPI, OpenAI (GPT-4 / embeddings), Perplexity |
| Database | ArangoDB (document + graph + vector) |
| Pipelines | Python, APScheduler, pandas, yfinance, FRED API |

---

## Data Sources

| Source | What it covers |
|--------|----------------|
| **Yahoo Finance** | S&P 500 (and related) OHLCV + technical features |
| **USASpending / Awards** | Government contract awards linked to companies |
| **SEC** | Filings, sections, sentence-level embeddings |
| **Polymarket / Kalshi** | Prediction markets, traders, graph edges to companies |
| **CME** | Commodity futures |
| **CFTC** | Commodity positioning (COT-style) |
| **EIA** | Energy data |
| **FRED** | Macro / economic series |
| **Options flow** | Unusual activity vs baselines |

Exact volumes change as pipelines run; see the in-app **About** page for current graph stats.

---

## Project Structure

```
├── frontend/          # Next.js app (Vercel-ready)
│   ├── app/           # Pages: home, markets, signals, database, about
│   ├── components/    # Query UI, charts, graph explorer, etc.
│   └── lib/           # API client, types
├── backend/           # FastAPI KARGA Query API
│   ├── app/
│   │   ├── api/       # /api/query, /markets, /database, /signals
│   │   ├── llm/       # Planning, AQL, validation, synthesis, web search
│   │   ├── analytics/ # Query logging
│   │   └── cache/     # Query cache
│   ├── config.py
│   └── main.py
└── src/
    ├── DAGS/pipeline/ # ETL modules per data source
    └── scheduler/     # APScheduler service (runs pipelines on a schedule)
```

---

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- ArangoDB instance (local or cloud) with the `QUANT_v3` graph loaded
- OpenAI API key (required for query planning / synthesis)
- Optional: Perplexity API key (web-augmented answers)

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env` (or a root `.env` loaded by your process):

```env
ARANGO_URL=http://localhost:8529
ARANGO_DB=QUANT_v3
ARANGO_USERNAME=root
ARANGO_PASSWORD=your_password

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
PERPLEXITY_API_KEY=          # optional

CORS_ORIGINS=http://localhost:3000
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

Start the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `http://localhost:8000/health`
- OpenAPI docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 3. Data pipelines (optional)

Pipelines live under `src/DAGS/pipeline/` and are orchestrated by `src/scheduler/app.py` (designed for Railway / APScheduler).

```bash
cd src
pip install -r requirements.txt
# Configure Arango + source API credentials in .env, then:
# python scheduler/app.py
```

Individual modules (Polymarket, Yahoo, Kalshi, etc.) can also be run locally for backfills.

---

## Environment Variables

### Backend

| Variable | Description |
|----------|-------------|
| `ARANGO_URL` / `ARANGO_HOST` | ArangoDB endpoint |
| `ARANGO_DB` | Database name (default `QUANT_v3`) |
| `ARANGO_USERNAME` / `ARANGO_PASSWORD` | Credentials |
| `OPENAI_API_KEY` | LLM + embeddings |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `PERPLEXITY_API_KEY` | Optional web search |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `SENTRY_DSN` | Optional error tracking |
| `DAILY_API_BUDGET` | Soft daily spend cap (USD) |

### Frontend

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend base URL |

---

## API Overview

| Prefix | Purpose |
|--------|---------|
| `POST /api/query/*` | NL query planning, execution, streaming synthesis |
| `/api/markets` | Prediction / market endpoints |
| `/api/database` | Schema / collection exploration |
| `/api/*` (signals) | Alpha signal endpoints |
| `GET /health` | Liveness + DB check |

Full interactive docs at `/docs` when the backend is running.

---

## Deployment

The repo is set up for a typical split deploy:

- **Frontend** → Vercel (`frontend/vercel.json`)
- **Backend** → Railway (`backend/railway.toml`, `Procfile`)
- **Scheduler / ETL** → Railway (`src/railway.toml`, `src/scheduler`)

Point `NEXT_PUBLIC_API_URL` at your deployed API and set `CORS_ORIGINS` accordingly.

---

## Example Questions

- Which companies received the largest government contracts in the last 90 days?
- What are Polymarket odds saying about Tesla?
- Show SEC filing language related to cybersecurity for defense contractors
- Which tickers have unusual options activity near recent filings?
- Compare prediction-market sentiment vs equity performance for AI-related names

---

## Disclaimer

KARGA is for research and informational purposes only. It is **not** investment advice. Markets, contracts, and prediction-market data can be incomplete, delayed, or wrong. Always verify critical decisions with primary sources and licensed professionals.

---

## Contact

Feedback / beta notes: [karga.analytics@gmail.com](mailto:karga.analytics@gmail.com)

---

## License

Private / unlicensed unless otherwise stated by the repository owner. Ask before redistributing.


<img width="860" height="806" alt="Screenshot 2026-01-28 225751" src="https://github.com/user-attachments/assets/c48c3f6b-4e89-46d3-a0a4-73f3c0e9a77d" />
<img width="726" height="635" alt="Screenshot 2026-02-17 225538" src="https://github.com/user-attachments/assets/e823b88e-c80b-44d8-9c9f-9ccbeeb0197f" />
<img width="704" height="679" alt="Screenshot 2026-02-17 225612" src="https://github.com/user-attachments/assets/b7e9fefd-ba2a-46da-ba69-372d6741d406" />
