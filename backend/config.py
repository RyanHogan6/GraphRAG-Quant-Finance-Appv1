"""
Backend configuration - loads from environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ArangoDB Configuration
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
GRAPH_NAME = "QUANT_v3_FinanceGraph"
DB_NAME = os.getenv("ARANGO_DB", "QUANT_v3")
USERNAME = os.getenv("ARANGO_USERNAME", "root")
PASSWORD = os.getenv("ARANGO_PASSWORD", "")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Perplexity Configuration (for web search/current events)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "360"))

# FastAPI Configuration
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# Error Tracking & Monitoring
SENTRY_DSN = os.getenv("SENTRY_DSN", "")  # Sentry error tracking
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

# Cost & Abuse Protection
DAILY_API_BUDGET = float(os.getenv("DAILY_API_BUDGET", "50.0"))  # $50/day default
MAX_QUERY_COMPLEXITY = int(os.getenv("MAX_QUERY_COMPLEXITY", "3"))  # Max FOR loops in builder mode

# Document Collections
COMPANY_COL = "Company"
MARKETDATA_COL = "MarketData"
AWARD_COL = "Award"
ECONOMIC_COL = "EconomicData"
COMMODITY_COL = "commodity_positions"
SEC_FILING_COL = "sec_filings"
SEC_SECTION_COL = "sec_sections"
SEC_SENTENCE_COL = "sec_sentences"
POLYMARKET_COL = "prediction_markets_polymarket"
KALSHIMARKET_COL = "prediction_markets_kalshi"

# Edge Collections - Company Relationships
EDGE_MARKETDATA = "HAS_MARKETDATA"
EDGE_AWARD = "HAS_AWARD"
EDGE_COMMODITY = "HAS_COMMODITY_POSITION"
EDGE_FILING = "HAS_FILING"

# Edge Collections - SEC Hierarchy
EDGE_SECTION = "has_section"
EDGE_SENTENCE = "has_sentence"

# Edge Collections - Prediction Markets (Polymarket)
EDGE_POLYMARKET_DIRECT = "market_mentions_company_polymarket"
EDGE_POLYMARKET_SECTOR = "market_related_to_sector_polymarket"
EDGE_POLYMARKET_MACRO = "market_affects_company_polymarket"

# Edge Collections - Prediction Markets (Kalshi)
EDGE_KALSHI_DIRECT = "market_mentions_company_kalshi"
EDGE_KALSHI_SECTOR = "market_related_to_sector_kalshi"
