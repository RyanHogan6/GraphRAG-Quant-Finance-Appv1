"""Kalshi configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

# Kalshi API
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")

# ArangoDB
DB_NAME = os.getenv("ARANGO_DB", "QUANT_v3")
USERNAME = os.getenv("ARANGO_USERNAME", "root")
PASSWORD = os.getenv("ARANGO_PASSWORD", "")
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")

# Collections
MARKET_COL = "prediction_markets_kalshi"
COMPANY_COL = "Company"
EDGE_MENTIONS = "market_mentions_company_kalshi"
EDGE_SECTOR = "market_related_to_sector_kalshi"
