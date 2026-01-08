import streamlit as st
import pandas as pd
import config as cfg
import database as arango_db
import llm as llm
import ui as ui
import time
from query_logger import get_logger
from datetime import datetime
import torch

st.set_page_config(page_title="GraphRAG", page_icon="▓", layout="centered")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Custom CSS - Vaporwave/Hacker aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap');

    /* Dark hacker background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        font-family: 'IBM Plex Mono', monospace;
    }

    .block-container {
        padding-top: 5rem;
        max-width: 700px;
    }

    /* Hide everything */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stElementToolbar"] {display: none;}

    /* Centered header - sleek monospace with subtle glow */
    h1 {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 300;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: 0.15em;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.2);
    }

    /* Tabs - minimal text with gold underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        padding: 0;
        justify-content: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 3rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #555555;
        font-weight: 300;
        padding: 10px 20px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #888888;
    }

    .stTabs [aria-selected="true"] {
        background: transparent;
        border: none;
        border-bottom: 1px solid #d4af37;
        color: #d4af37;
    }

    /* Search input - compact terminal style */
    .stTextInput input {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(212, 175, 55, 0.2);
        color: #d4af37;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 300;
        height: 32px;
    }

    .stTextInput input:focus {
        border-color: rgba(212, 175, 55, 0.5);
        box-shadow: 0 0 0 1px rgba(212, 175, 55, 0.2);
        outline: none;
    }

    .stTextInput input::placeholder {
        color: #444444;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Text input spacing */
    .stTextInput {
        margin-bottom: 1rem;
    }

    /* Search button - compact gold */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.15) 0%, rgba(212, 175, 55, 0.25) 100%);
        border: 1px solid rgba(212, 175, 55, 0.5);
        color: #d4af37;
        border-radius: 4px;
        font-weight: 400;
        padding: 6px 24px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-family: 'IBM Plex Mono', monospace;
        width: 100%;
        transition: all 0.2s ease;
        height: 32px;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.25) 0%, rgba(212, 175, 55, 0.35) 100%);
        border-color: #d4af37;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
    }

    /* Regular buttons - compact sample questions */
    .stButton > button {
        background: transparent;
        border: 1px solid rgba(212, 175, 55, 0.15);
        color: #666666;
        border-radius: 4px;
        font-weight: 300;
        padding: 6px 12px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        height: 32px;
    }

    .stButton > button:hover {
        background: rgba(212, 175, 55, 0.05);
        border-color: rgba(212, 175, 55, 0.3);
        color: #d4af37;
    }

    /* Data tables - terminal style */
    [data-testid="stDataFrame"] {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #d4af37;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 300;
    }

    /* Captions */
    .stCaption {
        color: #555555;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Alerts */
    .stAlert {
        background: rgba(0, 0, 0, 0.3);
        border-left: 2px solid #d4af37;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Simple centered header
st.markdown("<h1>GraphRAG</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["AI Query", "Database"])

# ==================== AI QUERY TAB ====================
with tab1:
    # Single compact search bar
    user_question = st.text_input(
        "query",
        placeholder="Ask about stocks, SEC filings, government contracts...",
        label_visibility="collapsed",
        key="main_query"
    )

    # Single search button
    search_clicked = st.button("Search", type="primary", use_container_width=True, disabled=not user_question)

    # Query execution
    if search_clicked and user_question:
        logger = get_logger()
        start_time = time.time()

        try:
            # Step 1: Intent check
            with st.spinner("[ ANALYZING QUERY ]"):
                intent = llm.quick_intent_check(user_question, use_local=False)

            # Step 2: Generate query plan
            with st.spinner("[ GENERATING AQL ]"):
                query_plan = llm.plan_query_with_llm(user_question, intent_hint=intent, use_local=False)

            # Step 3: Execute
            with st.spinner("[ EXECUTING ]"):
                results, error = llm.execute_with_retry(query_plan, max_retries=2)

            execution_time = time.time() - start_time

            if error:
                st.error(f"ERROR: {error}")
            elif results:
                st.success(f"✓ {len(results)} results | {execution_time:.2f}s")

                # Display results
                df = pd.DataFrame(results)
                cols = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
                if cols:
                    st.dataframe(df[cols], use_container_width=True, hide_index=True)

                # Log query
                st.session_state.query_history.append({
                    'question': user_question,
                    'plan': query_plan,
                    'result_count': len(results),
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                st.warning("No results found")

        except Exception as e:
            st.error(f"ERROR: {str(e)}")


# ==================== DATABASE TAB ====================
with tab2:
    ui.render_database_browser_tab()
