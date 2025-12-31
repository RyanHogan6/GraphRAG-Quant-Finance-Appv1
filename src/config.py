import streamlit as st
from arango import ArangoClient
import torch
import re
import openai
from datetime import datetime
from dotenv import load_dotenv
import os 
import pandas as pd 
import json

# Load environment variables
load_dotenv() 

# Configuration
ARANGO_URL = "https://e11129e8c5ae.arangodb.cloud:8529"   #  "http://localhost:8529"
GRAPH_NAME = "QUANT_v1_FinanceGraph"
DB_NAME = "QUANT_v1"
USERNAME = "root"
PASSWORD = os.getenv('PASSWORD')
COMPANY_COL = "Company"
MARKETDATA_COL = "MarketData"
EDGE_MARKETDATA_COL = "HAS_MARKETDATA"
AWARD_COL = "Award"
EDGE_AWARD_COL = "HAS_AWARD"
FRED_COL = "FREDData"
COMMODITY_COL = "CommodityPosition"
SEC_COL = "sec_filings"

openai.api_key = os.getenv('OPENAI_API_KEY')

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS = 1500
TEMPERATURE = 0.1
QUERY_TIMEOUT = 30