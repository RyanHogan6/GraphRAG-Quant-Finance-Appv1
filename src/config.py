import streamlit as st
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 

# Configuration
ARANGO_URL = st.secrets["arangodb"]["host"] #"http://localhost:8529" 
GRAPH_NAME = "QUANT_v3_FinanceGraph"
DB_NAME = st.secrets["arangodb"]["database"]
USERNAME = st.secrets["arangodb"]["username"]
PASSWORD =  st.secrets["arangodb"]["password"]
openai.api_key =  st.secrets["arangodb"]['open_api_key'] 


# Document Collections
COMPANY_COL = "Company"
MARKETDATA_COL = "MarketData"
AWARD_COL = "Award"
ECONOMIC_COL = "EconomicData"  # Updated from FREDData
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

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS = 1500
TEMPERATURE = 0.1
QUERY_TIMEOUT = 360
