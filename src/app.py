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
import html

st.set_page_config(page_title="Finna Go Alpha", page_icon="▓", layout="wide")

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
    st.markdown("### 📊 Prediction Markets")

    # Get database connection
    try:
        db = arango_db.get_arango_connection()

        # Fetch categories for filter
        categories_query = """
        FOR m IN prediction_markets_polymarket
            FILTER m.closed == false AND m.category != null
            COLLECT category = m.category WITH COUNT INTO count
            SORT count DESC
            RETURN {category: category, count: count}
        """
        categories_data = list(db.aql.execute(categories_query))
        categories = ["All"] + [c['category'] for c in categories_data]

        # Filters row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

        with col1:
            selected_category = st.selectbox(
                "Category",
                categories,
                index=0,
                key="market_category_filter"
            )

        with col2:
            min_volume = st.number_input(
                "Min 24h Volume ($)",
                min_value=0,
                max_value=1000000,
                value=0,
                step=1000,
                key="min_volume_filter"
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Volume (High to Low)", "Volume (Low to High)", "Probability (High to Low)", "Probability (Low to High)", "Liquidity (High to Low)"],
                index=0,
                key="market_sort"
            )

        with col4:
            limit = st.selectbox(
                "Show",
                [10, 20, 50, 100],
                index=1,
                key="markets_tab_limit"
            )

        st.markdown("<br>", unsafe_allow_html=True)

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

        # Display metrics in columns with custom styling
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.markdown(f"""
                <div style="background: rgba(212, 175, 55, 0.05); border-left: 3px solid #d4af37; padding: 15px; border-radius: 4px;">
                    <div style="color: #666; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">Active Markets</div>
                    <div style="color: #d4af37; font-size: 1.8rem; font-weight: 300; font-family: 'IBM Plex Mono', monospace;">{metrics['total_markets']:,}</div>
                </div>
            """, unsafe_allow_html=True)

        with metric_col2:
            st.markdown(f"""
                <div style="background: rgba(212, 175, 55, 0.05); border-left: 3px solid #d4af37; padding: 15px; border-radius: 4px;">
                    <div style="color: #666; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">Whale Traders</div>
                    <div style="color: #d4af37; font-size: 1.8rem; font-weight: 300; font-family: 'IBM Plex Mono', monospace;">{metrics['total_whales']:,}</div>
                </div>
            """, unsafe_allow_html=True)

        with metric_col3:
            st.markdown(f"""
                <div style="background: rgba(212, 175, 55, 0.05); border-left: 3px solid #d4af37; padding: 15px; border-radius: 4px;">
                    <div style="color: #666; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">24h Volume</div>
                    <div style="color: #d4af37; font-size: 1.8rem; font-weight: 300; font-family: 'IBM Plex Mono', monospace;">${metrics['total_volume_24h']/1e6:.1f}M</div>
                </div>
            """, unsafe_allow_html=True)

        # Build dynamic query based on filters FIRST (need count for metric)
        category_filter = f"FILTER market.category == '{selected_category}'" if selected_category != "All" else ""
        volume_filter = f"FILTER market.volume_24h >= {min_volume}" if min_volume > 0 else ""

        # Determine sort field and direction
        sort_mapping = {
            "Volume (High to Low)": ("market.volume_24h", "DESC"),
            "Volume (Low to High)": ("market.volume_24h", "ASC"),
            "Probability (High to Low)": ("market.yes_probability", "DESC"),
            "Probability (Low to High)": ("market.yes_probability", "ASC"),
            "Liquidity (High to Low)": ("market.liquidity", "DESC")
        }
        sort_field, sort_dir = sort_mapping[sort_by]

        top_markets_query = f"""
        FOR market IN prediction_markets_polymarket
            FILTER market.closed == false
            FILTER market.volume_24h > 0
            {category_filter}
            {volume_filter}
            SORT {sort_field} {sort_dir}
            LIMIT {limit}
            RETURN {{
                question: market.question,
                yes_prob: FLOOR(market.yes_probability * 100),
                volume_24h: market.volume_24h,
                liquidity: market.liquidity,
                category: market.category,
                end_date: market.end_date
            }}
        """

        top_markets = list(db.aql.execute(top_markets_query))
        filtered_count = len(top_markets)

        # Now display metric with actual count
        with metric_col4:
            st.markdown(f"""
                <div style="background: rgba(212, 175, 55, 0.05); border-left: 3px solid #d4af37; padding: 15px; border-radius: 4px;">
                    <div style="color: #666; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">Showing</div>
                    <div style="color: #d4af37; font-size: 1.8rem; font-weight: 300; font-family: 'IBM Plex Mono', monospace;">{filtered_count:,}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🔥 Top Markets")

        if top_markets:
            # Create custom HTML table with better styling
            table_html = """<style>
.market-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.88rem;
                margin-top: 20px;
                background: rgba(0, 0, 0, 0.4);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .market-table thead {
                background: rgba(212, 175, 55, 0.2);
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .market-table th {
                padding: 16px 20px;
                text-align: left;
                color: #d4af37;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-size: 0.7rem;
                border-bottom: 2px solid rgba(212, 175, 55, 0.4);
            }
            .market-table td {
                padding: 16px 20px;
                color: #cccccc;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                vertical-align: middle;
            }
            .market-table tbody tr {
                transition: all 0.2s ease;
            }
            .market-table tbody tr:hover {
                background: rgba(212, 175, 55, 0.12);
                cursor: pointer;
                transform: translateX(2px);
            }
            .market-question {
                line-height: 1.5;
                color: #ffffff;
                font-size: 0.92rem;
            }
            .market-prob {
                color: #d4af37;
                font-weight: 600;
                font-size: 1rem;
            }
            .market-volume {
                color: #88ccff;
                font-weight: 500;
            }
            .market-category {
                display: inline-block;
                padding: 5px 12px;
                background: rgba(212, 175, 55, 0.15);
                border: 1px solid rgba(212, 175, 55, 0.4);
                border-radius: 4px;
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #d4af37;
            }
            </style>
            <div style="overflow-x: auto; max-height: 800px; overflow-y: auto;">
            <table class="market-table">
                <thead>
                    <tr>
                        <th style="width: 48%;">Market Question</th>
                        <th style="width: 10%; text-align: center;">Yes %</th>
                        <th style="width: 14%; text-align: right;">24h Volume</th>
                        <th style="width: 14%; text-align: right;">Liquidity</th>
                        <th style="width: 14%; text-align: center;">Category</th>
                    </tr>
                </thead>
                <tbody>
            """

            for market in top_markets:
                question_raw = market['question'][:180] + "..." if len(market['question']) > 180 else market['question']
                question = html.escape(question_raw)
                yes_prob = market['yes_prob']
                volume = f"${market['volume_24h']/1000:.1f}k" if market['volume_24h'] < 100000 else f"${market['volume_24h']/1000000:.2f}M"
                liquidity = f"${market['liquidity']/1000:.1f}k" if market.get('liquidity') and market['liquidity'] < 100000 else (f"${market['liquidity']/1000000:.2f}M" if market.get('liquidity') else "N/A")
                category = html.escape(market['category'] or "Other")

                table_html += f"""
                    <tr>
                        <td class="market-question">{question}</td>
                        <td class="market-prob" style="text-align: center;">{yes_prob}%</td>
                        <td class="market-volume" style="text-align: right;">{volume}</td>
                        <td style="text-align: right;">{liquidity}</td>
                        <td style="text-align: center;"><span class="market-category">{category}</span></td>
                    </tr>
                """

            table_html += """
                </tbody>
            </table>
            </div>
            """

            st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Top Whale Traders
        st.markdown("#### 🐋 Top Whale Traders")

        top_whales_query = """
        FOR trader IN polymarket_traders
            FILTER trader.is_whale == true
            SORT trader.total_volume DESC
            LIMIT 20
            RETURN {
                address: trader.address,
                volume: trader.total_volume,
                profit: trader.total_profit,
                trades: trader.total_trades,
                activity: trader.activity_level,
                profit_ratio: trader.profit_ratio
            }
        """

        top_whales = list(db.aql.execute(top_whales_query))

        if top_whales:
            # Create custom HTML table for whales
            whale_table_html = """
            <style>
            .whale-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.88rem;
                margin-top: 20px;
                background: rgba(0, 0, 0, 0.4);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .whale-table thead {
                background: rgba(212, 175, 55, 0.2);
            }
            .whale-table th {
                padding: 16px 20px;
                text-align: left;
                color: #d4af37;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-size: 0.7rem;
                border-bottom: 2px solid rgba(212, 175, 55, 0.4);
            }
            .whale-table td {
                padding: 16px 20px;
                color: #cccccc;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                vertical-align: middle;
            }
            .whale-table tbody tr {
                transition: all 0.2s ease;
            }
            .whale-table tbody tr:hover {
                background: rgba(212, 175, 55, 0.12);
                cursor: pointer;
                transform: translateX(2px);
            }
            .whale-address {
                font-family: 'Courier New', monospace;
                color: #88ccff;
                font-size: 0.85rem;
                letter-spacing: 0.02em;
            }
            .whale-profit-positive {
                color: #00ff88;
                font-weight: 600;
            }
            .whale-profit-negative {
                color: #ff6b6b;
                font-weight: 600;
            }
            .whale-activity {
                display: inline-block;
                padding: 5px 12px;
                background: rgba(212, 175, 55, 0.15);
                border: 1px solid rgba(212, 175, 55, 0.4);
                border-radius: 4px;
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #d4af37;
            }
            </style>
            <div style="overflow-x: auto; max-height: 700px; overflow-y: auto;">
            <table class="whale-table">
                <thead>
                    <tr>
                        <th style="width: 22%;">Address</th>
                        <th style="width: 16%; text-align: right;">Total Volume</th>
                        <th style="width: 16%; text-align: right;">Total Profit</th>
                        <th style="width: 13%; text-align: center;">Profit Ratio</th>
                        <th style="width: 13%; text-align: center;">Trades</th>
                        <th style="width: 20%; text-align: center;">Activity</th>
                    </tr>
                </thead>
                <tbody>
            """

            for whale in top_whales:
                address = whale['address'][:10] + "..." + whale['address'][-8:]
                volume_val = whale['volume']
                volume = f"${volume_val/1000:.1f}k" if volume_val < 100000 else f"${volume_val/1000000:.2f}M"
                profit = whale['profit']
                profit_abs = abs(profit)
                profit_formatted = f"${profit_abs/1000:.1f}k" if profit_abs < 100000 else f"${profit_abs/1000000:.2f}M"
                profit_class = "whale-profit-positive" if profit >= 0 else "whale-profit-negative"
                profit_sign = "+" if profit >= 0 else "-"
                profit_ratio = f"{whale['profit_ratio']*100:.1f}%" if whale.get('profit_ratio') else "N/A"
                trades = f"{whale['trades']:,}"
                activity = whale['activity'] or "unknown"

                whale_table_html += f"""
                    <tr>
                        <td class="whale-address">{address}</td>
                        <td style="color: #d4af37; font-weight: 500; text-align: right;">{volume}</td>
                        <td class="{profit_class}" style="text-align: right;">{profit_sign}{profit_formatted}</td>
                        <td class="{profit_class}" style="text-align: center; font-size: 0.92rem;">{profit_ratio}</td>
                        <td style="text-align: center;">{trades}</td>
                        <td style="text-align: center;"><span class="whale-activity">{activity}</span></td>
                    </tr>
                """

            whale_table_html += """
                </tbody>
            </table>
            </div>
            """

            st.markdown(whale_table_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading markets data: {str(e)}")


# ==================== DATABASE TAB ====================
with tab3:
    ui.render_database_browser_tab()
