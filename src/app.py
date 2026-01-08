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

st.set_page_config(page_title="GraphRAG", page_icon="📊", layout="wide")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Custom CSS - Minimal ChatGPT-like design
st.markdown("""
    <style>
    /* Clean dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }

    .block-container {
        padding-top: 4rem;
        max-width: 800px;
    }

    /* Hide everything */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stElementToolbar"] {display: none;}

    /* Centered header */
    h1 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    /* Tabs - text only with underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        padding: 0;
        justify-content: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 3rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #666666;
        font-weight: 400;
        padding: 12px 24px;
        font-size: 0.95rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #999999;
    }

    .stTabs [aria-selected="true"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid #d4af37;
        color: #d4af37;
    }

    /* Large centered search input */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 1rem;
    }

    .stTextInput input:focus {
        border-color: rgba(212, 175, 55, 0.4);
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
    }

    .stTextInput input::placeholder {
        color: #555555;
    }

    /* Buttons - gold accent */
    .stButton > button {
        background: transparent;
        border: 1px solid rgba(212, 175, 55, 0.3);
        color: #d4af37;
        border-radius: 8px;
        font-weight: 400;
        padding: 8px 20px;
    }

    .stButton > button:hover {
        background: rgba(212, 175, 55, 0.1);
        border-color: rgba(212, 175, 55, 0.5);
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #d4af37;
    }
    </style>
""", unsafe_allow_html=True)

# Simple centered header
st.markdown("<h1>GraphRAG</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["AI Query", "Database"])

# ==================== AI QUERY TAB ====================
with tab1:
    # Main search input
    user_question = st.text_input(
        "query",
        placeholder="Ask anything about stocks, SEC filings, or government contracts...",
        label_visibility="collapsed",
        key="main_search"
    )

    # Sample questions
    if not user_question:
        st.caption("Example questions:")
        col1, col2 = st.columns(2)

        samples_left = [
            "Tesla closing price on 2020-06-15",
            "Top defense contracts in 2024",
            "Cybersecurity risks in SEC filings"
        ]

        samples_right = [
            "AAPL EBITDA in March 2017",
            "Tech companies with negative sentiment",
            "Government contracts with positive sentiment"
        ]

        with col1:
            for q in samples_left:
                if st.button(q, key=f"s1_{q[:15]}", use_container_width=True):
                    st.session_state.current_question = q
                    st.rerun()

        with col2:
            for q in samples_right:
                if st.button(q, key=f"s2_{q[:15]}", use_container_width=True):
                    st.session_state.current_question = q
                    st.rerun()

    # Query execution
    if user_question or st.session_state.current_question:
        if st.session_state.current_question and not user_question:
            user_question = st.session_state.current_question

        logger = get_logger()
        start_time = time.time()

        try:
            # Step 1: Intent check
            with st.spinner("Understanding query..."):
                intent = llm.quick_intent_check(user_question, use_local=False)

            # Step 2: Generate query plan
            with st.spinner("Planning query..."):
                query_plan = llm.plan_query_with_llm(user_question, intent_hint=intent, use_local=False)

            # Step 3: Execute
            with st.spinner("Fetching results..."):
                results, error = llm.execute_with_retry(query_plan, max_retries=2)

            execution_time = time.time() - start_time

            if error:
                st.error(f"Query failed: {error}")
            elif results:
                st.success(f"Found {len(results)} results ({execution_time:.1f}s)")

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

                # Clear current question
                st.session_state.current_question = ""
            else:
                st.warning("No results found")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ==================== DATABASE TAB ====================
with tab2:
    ui.render_database_browser_tab()
