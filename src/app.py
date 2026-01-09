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

st.set_page_config(page_title="Finna Go Alpha", page_icon="▓", layout="centered")

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

    /* Data tables - enhanced contrast terminal style */
    [data-testid="stDataFrame"] {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
    }

    /* Table headers */
    [data-testid="stDataFrame"] thead tr {
        background: rgba(212, 175, 55, 0.15) !important;
        border-bottom: 2px solid rgba(212, 175, 55, 0.4);
    }

    [data-testid="stDataFrame"] thead th {
        color: #d4af37 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 12px 8px !important;
        font-size: 0.75rem !important;
        border-right: 1px solid rgba(212, 175, 55, 0.1);
    }

    /* Table cells */
    [data-testid="stDataFrame"] tbody td {
        color: #cccccc !important;
        padding: 10px 8px !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Zebra striping */
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background: rgba(255, 255, 255, 0.02);
    }

    [data-testid="stDataFrame"] tbody tr:hover {
        background: rgba(212, 175, 55, 0.08) !important;
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
tab1, tab2, tab3 = st.tabs(["AI Query", "Markets", "Database"])

# ==================== AI QUERY TAB ====================
with tab1:
    # Single compact search bar
    user_question = st.text_input(
        "query",
        placeholder="Ask about stocks, SEC filings, prediction markets, whale traders...",
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

                # Display results with enhanced styling
                df = pd.DataFrame(results)
                cols = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
                if cols:
                    # Apply pandas styling for better visual hierarchy
                    styled_df = df[cols].style.set_properties(**{
                        'background-color': 'rgba(0, 0, 0, 0.4)',
                        'color': '#cccccc',
                        'border': '1px solid rgba(255, 255, 255, 0.1)',
                        'padding': '8px',
                        'font-family': 'IBM Plex Mono, monospace',
                        'font-size': '0.8rem'
                    }).set_table_styles([
                        {'selector': 'thead th',
                         'props': [
                             ('background-color', 'rgba(212, 175, 55, 0.2)'),
                             ('color', '#d4af37'),
                             ('font-weight', '500'),
                             ('text-transform', 'uppercase'),
                             ('letter-spacing', '0.1em'),
                             ('padding', '12px 8px'),
                             ('border', '1px solid rgba(212, 175, 55, 0.3)'),
                             ('font-size', '0.75rem')
                         ]},
                        {'selector': 'tbody tr:nth-child(even)',
                         'props': [('background-color', 'rgba(255, 255, 255, 0.02)')]},
                        {'selector': 'tbody tr:hover',
                         'props': [('background-color', 'rgba(212, 175, 55, 0.1)')]}
                    ])

                    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)

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


# ==================== MARKETS TAB ====================
with tab2:
    st.markdown("### Prediction Markets")

    # Get database connection
    try:
        db = arango_db.get_arango_connection()

        # Fetch key metrics
        metrics_query = """
        LET total_markets = LENGTH(
            FOR m IN prediction_markets_polymarket
                FILTER m.closed == false
                RETURN 1
        )

        LET total_whales = LENGTH(
            FOR t IN polymarket_traders
                FILTER t.is_whale == true
                RETURN 1
        )

        LET total_volume_24h = SUM(
            FOR m IN prediction_markets_polymarket
                FILTER m.closed == false AND m.volume_24h != null
                RETURN m.volume_24h
        )

        RETURN {
            total_markets: total_markets,
            total_whales: total_whales,
            total_volume_24h: total_volume_24h
        }
        """

        metrics = list(db.aql.execute(metrics_query))[0]

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Active Markets",
                f"{metrics['total_markets']:,}",
                help="Open prediction markets on Polymarket"
            )

        with col2:
            st.metric(
                "Whale Traders",
                f"{metrics['total_whales']:,}",
                help="Traders with >$50k volume"
            )

        with col3:
            st.metric(
                "24h Volume",
                f"${metrics['total_volume_24h']/1e6:.1f}M",
                help="Total 24-hour trading volume"
            )

        st.markdown("---")

        # Top Markets by Volume
        st.markdown("#### 🔥 Top Markets by Volume")

        top_markets_query = """
        FOR market IN prediction_markets_polymarket
            FILTER market.closed == false
            FILTER market.volume_24h > 0
            SORT market.volume_24h DESC
            LIMIT 20
            RETURN {
                question: market.question,
                yes_prob: FLOOR(market.yes_probability * 100),
                volume_24h: market.volume_24h,
                liquidity: market.liquidity,
                category: market.category,
                end_date: market.end_date
            }
        """

        top_markets = list(db.aql.execute(top_markets_query))

        if top_markets:
            markets_df = pd.DataFrame(top_markets)
            markets_df.columns = ['Question', 'Yes %', '24h Volume', 'Liquidity', 'Category', 'End Date']
            markets_df['24h Volume'] = markets_df['24h Volume'].apply(lambda x: f"${x:,.0f}")
            markets_df['Liquidity'] = markets_df['Liquidity'].apply(lambda x: f"${x:,.0f}")

            # Style the dataframe
            styled_markets = markets_df.style.set_properties(**{
                'background-color': 'rgba(0, 0, 0, 0.4)',
                'color': '#cccccc',
                'border': '1px solid rgba(255, 255, 255, 0.1)',
                'padding': '8px',
                'font-family': 'IBM Plex Mono, monospace',
                'font-size': '0.75rem',
                'text-align': 'left'
            }).set_table_styles([
                {'selector': 'thead th',
                 'props': [
                     ('background-color', 'rgba(212, 175, 55, 0.2)'),
                     ('color', '#d4af37'),
                     ('font-weight', '500'),
                     ('text-transform', 'uppercase'),
                     ('letter-spacing', '0.1em'),
                     ('padding', '10px 8px'),
                     ('border', '1px solid rgba(212, 175, 55, 0.3)'),
                     ('font-size', '0.7rem'),
                     ('text-align', 'left')
                 ]},
                {'selector': 'tbody tr:nth-child(even)',
                 'props': [('background-color', 'rgba(255, 255, 255, 0.02)')]},
                {'selector': 'tbody tr:hover',
                 'props': [('background-color', 'rgba(212, 175, 55, 0.08)')]}
            ])

            st.dataframe(styled_markets, use_container_width=True, hide_index=True, height=600)

        st.markdown("---")

        # Top Whale Traders
        st.markdown("#### 🐋 Top Whale Traders")

        top_whales_query = """
        FOR trader IN polymarket_traders
            FILTER trader.is_whale == true
            SORT trader.total_volume DESC
            LIMIT 15
            RETURN {
                address: CONCAT(SUBSTRING(trader.address, 0, 6), "...", SUBSTRING(trader.address, -4)),
                volume: trader.total_volume,
                profit: trader.total_profit,
                trades: trader.total_trades,
                activity: trader.activity_level
            }
        """

        top_whales = list(db.aql.execute(top_whales_query))

        if top_whales:
            whales_df = pd.DataFrame(top_whales)
            whales_df.columns = ['Address', 'Total Volume', 'Total Profit', 'Trades', 'Activity']
            whales_df['Total Volume'] = whales_df['Total Volume'].apply(lambda x: f"${x:,.0f}")
            whales_df['Total Profit'] = whales_df['Total Profit'].apply(lambda x: f"${x:,.0f}")

            # Style the dataframe
            styled_whales = whales_df.style.set_properties(**{
                'background-color': 'rgba(0, 0, 0, 0.4)',
                'color': '#cccccc',
                'border': '1px solid rgba(255, 255, 255, 0.1)',
                'padding': '8px',
                'font-family': 'IBM Plex Mono, monospace',
                'font-size': '0.75rem',
                'text-align': 'left'
            }).set_table_styles([
                {'selector': 'thead th',
                 'props': [
                     ('background-color', 'rgba(212, 175, 55, 0.2)'),
                     ('color', '#d4af37'),
                     ('font-weight', '500'),
                     ('text-transform', 'uppercase'),
                     ('letter-spacing', '0.1em'),
                     ('padding', '10px 8px'),
                     ('border', '1px solid rgba(212, 175, 55, 0.3)'),
                     ('font-size', '0.7rem'),
                     ('text-align', 'left')
                 ]},
                {'selector': 'tbody tr:nth-child(even)',
                 'props': [('background-color', 'rgba(255, 255, 255, 0.02)')]},
                {'selector': 'tbody tr:hover',
                 'props': [('background-color', 'rgba(212, 175, 55, 0.08)')]}
            ])

            st.dataframe(styled_whales, use_container_width=True, hide_index=True, height=400)

    except Exception as e:
        st.error(f"Error loading markets data: {str(e)}")


# ==================== DATABASE TAB ====================
with tab3:
    ui.render_database_browser_tab()
