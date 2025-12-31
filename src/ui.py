import streamlit as st
from datetime import datetime
import pandas as pd 
import database as arango_db


# ============================================================================
# COLLECTION METADATA
# ============================================================================

COLLECTION_CONFIG = {
    'Company': {
        'display_name': 'Companies',
        'icon': '🏢',
        'description': 'Corporate entity information and profiles',
        'field_mappings': {
            'ticker': 'Ticker',
            'name': 'Company Name',
            'sector': 'Sector',
            'industry': 'Industry',
            'sharesOutstanding': 'Shares Outstanding',
            'marketCap': 'Market Cap',
            'website': 'Website',
            'description': 'Description',
            'country': 'Country',
            'city': 'City',
            'state': 'State',
            'employees': 'Employees'
        },
        'numeric_fields': ['sharesOutstanding', 'marketCap', 'employees'],
        'searchable_fields': ['ticker', 'name', 'sector', 'industry']
    },
    'MarketData': {
        'display_name': 'S&P 500 Market Data',
        'icon': '📊',
        'description': 'Historical and real-time market prices',
        'field_mappings': {
            'ticker': 'Ticker',
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'marketCap': 'Market Cap',
            'targetMeanPrice': 'Target Mean Price',
            'forwardEps': 'Forward EPS',
            'returnOnEquity': 'Return On Equity',
            'dividendRate': 'Dividend Rate',
            'grossMargins': 'Gross Margins',
            'ebitdaMargins': 'EBITDA Margins',
            'returnOnAssets': 'Return On Assets',
            'sharesOutstanding': 'Shares Outstanding'
        },
        'numeric_fields': ['open', 'high', 'low', 'close', 'volume', 'marketCap', 
                          'targetMeanPrice', 'forwardEps', 'returnOnEquity', 'dividendRate',
                          'grossMargins', 'ebitdaMargins', 'returnOnAssets', 'sharesOutstanding'],
        'percentage_fields': ['returnOnEquity', 'dividendRate', 'grossMargins', 'ebitdaMargins', 'returnOnAssets'],
        'price_fields': ['open', 'high', 'low', 'close', 'targetMeanPrice', 'forwardEps'],
        'searchable_fields': ['ticker', 'date']
    },
    'Award': {
        'display_name': 'Government Contract Awards',
        'icon': '🏛️',
        'description': 'Federal government contract data',
        'field_mappings': {
            'award_id': 'Award ID',
            'internal_id': 'Internal ID',
            'recipient_name': 'Recipient Name',
            'recipient_uei': 'Recipient UEI',
            'recipient_duns': 'Recipient DUNS',
            'recipient_parent_name': 'Recipient Parent Name',
            'recipient_parent_uei': 'Recipient Parent UEI',
            'recipient_parent_duns': 'Recipient Parent DUNS',
            'recipient_country': 'Recipient Country',
            'recipient_state': 'Recipient State',
            'recipient_city': 'Recipient City',
            'recipient_zip': 'Recipient ZIP',
            'recipient_congressional_district': 'Congressional District',
            'start_date': 'Start Date',
            'end_date': 'End Date',
            'award_amount': 'Award Amount',
            'total_obligation': 'Total Obligation',
            'base_obligation': 'Base Obligation',
            'awarding_agency': 'Agency',
            'awarding_agency_code': 'Agency Code',
            'awarding_sub_agency': 'Sub Agency',
            'awarding_office': 'Awarding Office',
            'funding_agency': 'Funding Agency',
            'funding_sub_agency': 'Funding Sub Agency',
            'funding_office': 'Funding Office',
            'contract_type': 'Contract Type',
            'award_type': 'Award Type',
            'idv_type': 'IDV Type',
            'naics_code': 'NAICS Code',
            'naics_description': 'NAICS Description',
            'product_service_code': 'Product Service Code',
            'product_service_description': 'Product Service Description',
            'place_of_performance_city': 'Performance City',
            'place_of_performance_state': 'Performance State',
            'place_of_performance_country': 'Performance Country',
            'place_of_performance_zip': 'Performance ZIP',
            'ticker': 'Ticker',
            'description': 'Description',
            'period_of_performance_start': 'Performance Start',
            'period_of_performance_end': 'Performance End',
            'solicitation_date': 'Solicitation Date',
            'action_date': 'Action Date',
            'fiscal_year': 'Fiscal Year',
            'contract_award_unique_key': 'Contract Award Key',
            'transaction_number': 'Transaction Number',
            'modification_number': 'Modification Number'
        },
        'numeric_fields': ['award_amount', 'total_obligation', 'base_obligation'],
        'currency_fields': ['award_amount', 'total_obligation', 'base_obligation'],
        'excluded_fields': ['source', 'description_embedding'],  # Fields to never display
        'searchable_fields': ['ticker', 'award_id', 'recipient_name', 'awarding_agency', 'naics_description']
    },

    'CommodityPosition': {
        'display_name': 'Commodity Positions',
        'icon': '📦',
        'description': 'CFTC commodity positions',
        'field_mappings': {},
        'numeric_fields': [],
        'searchable_fields': []
    },
    'FREDData': {
        'display_name': 'Federal Reserve Economic Data (FRED)',
        'icon': '📈',
        'description': 'Macroeconomic indicators',
        'field_mappings': {},
        'numeric_fields': [],
        'searchable_fields': []
    },
    'sec_filings': {
        'display_name': 'SEC Filings (8-K, 10-K, 10-Q)',
        'icon': '📄',
        'description': 'SEC regulatory filings',
        'field_mappings': {},
        'numeric_fields': [],
        'searchable_fields': []
    }
}


def get_collection_config(collection_name):
    """Get display configuration for a collection"""
    return COLLECTION_CONFIG.get(collection_name, {
        'display_name': collection_name,
        'icon': '📁',
        'description': 'Database collection',
        'field_mappings': {},
        'numeric_fields': [],
        'searchable_fields': []
    })


# ============================================================================
# STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS"""
    st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
            border-right: 2px solid #6366f1;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #9ca3af;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
        }
        
        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.02);
        }
        
        /* Remove extra padding */
        .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value):
    """Format currency values - FIXED to handle non-numeric input"""
    try:
        # Convert to float if it's a string
        if isinstance(value, str):
            value = float(value)
        
        if pd.isna(value) or value == 0:
            return "N/A"
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        if value >= 1e6:
            return f"${value/1e6:.2f}M"
        if value >= 1e3:
            return f"${value/1e3:.1f}K"
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_number(value):
    """Format large numbers - FIXED to handle non-numeric input"""
    try:
        # Convert to float if it's a string
        if isinstance(value, str):
            value = float(value)
        
        if pd.isna(value):
            return "N/A"
        if value >= 1e9:
            return f"{value/1e9:.2f}B"
        if value >= 1e6:
            return f"{value/1e6:.2f}M"
        if value >= 1e3:
            return f"{value/1e3:.1f}K"
        return f"{value:,.0f}"
    except (ValueError, TypeError):
        return "N/A"

def rename_dataframe_columns(df, field_mappings):
    """Rename dataframe columns using field mappings"""
    rename_dict = {}
    for col in df.columns:
        if col in field_mappings:
            rename_dict[col] = field_mappings[col]
        else:
            # Auto-convert any unmapped column names from snake_case to Title Case
            rename_dict[col] = col.replace('_', ' ').title()
    return df.rename(columns=rename_dict)


def format_dataframe_values(df, config):
    """Format numeric values in dataframe based on field type"""
    df_formatted = df.copy()
    
    field_mappings = config.get('field_mappings', {})
    numeric_fields = config.get('numeric_fields', [])
    percentage_fields = config.get('percentage_fields', [])
    price_fields = config.get('price_fields', [])
    
    for field in numeric_fields:
        # Use either original or mapped name
        display_name = field_mappings.get(field, field)
        
        if field in df_formatted.columns:
            col_name = field
        elif display_name in df_formatted.columns:
            col_name = display_name
        else:
            continue
        
        # Convert to numeric first
        try:
            df_formatted[col_name] = pd.to_numeric(df_formatted[col_name], errors='coerce')
        except:
            continue
        
        # Format based on field type
        if field in percentage_fields:
            # Format as percentage
            df_formatted[col_name] = df_formatted[col_name].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
        elif field in price_fields:
            # Format as price with 2 decimals
            df_formatted[col_name] = df_formatted[col_name].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
        elif 'marketCap' in field.lower() or 'amount' in field.lower():
            # Format as currency
            df_formatted[col_name] = df_formatted[col_name].apply(
                lambda x: format_currency(x) if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
        elif 'shares' in field.lower() or 'volume' in field.lower() or 'employees' in field.lower():
            # Format with commas (no decimals for large numbers)
            df_formatted[col_name] = df_formatted[col_name].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
        else:
            # Default: add commas for readability
            df_formatted[col_name] = df_formatted[col_name].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
    
    return df_formatted
def render_awards_collection(db):
    """Render enhanced Government Contract Awards collection"""
    config = get_collection_config('Award')
    
    # Search and limit
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            "🔍 Search Government Contract Awards:",
            placeholder="Enter search term",
            key="awards_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Limit:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="awards_limit"
        )
    
    # Build query
    if search_term:
        filter_conditions = " OR ".join([
            f"LOWER(doc.{field}) LIKE LOWER(@search)"
            for field in config['searchable_fields']
        ])
        aql_query = f"""
        FOR doc IN Award
            FILTER {filter_conditions}
            SORT doc.start_date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = """
        FOR doc IN Award
            SORT doc.start_date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if not results:
        st.warning("No awards found")
        return
    
    st.success(f"✅ Loaded {len(results)} documents")
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Get excluded fields list
    excluded_fields = config.get('excluded_fields', [])
    excluded_fields.extend(['_key', '_id', '_rev'])  # Always exclude internal ArangoDB fields
    
    # Filter out excluded fields and internal fields
    cols_to_show = [c for c in df.columns if c not in excluded_fields and not c.startswith('_')]
    
    # Preferred column order (display these first if they exist)
    preferred_order = [
        'award_id', 'recipient_name', 'ticker', 'start_date', 'end_date', 
        'award_amount', 'awarding_agency', 'contract_type', 'naics_description',
        'place_of_performance_state', 'description'
    ]
    
    # Reorder columns: preferred first, then the rest
    ordered_cols = [c for c in preferred_order if c in cols_to_show]
    remaining_cols = [c for c in cols_to_show if c not in ordered_cols]
    cols_to_show = ordered_cols + remaining_cols
    
    df_display = df[cols_to_show].copy()
    
    # Rename columns using field mappings
    df_display = rename_dataframe_columns(df_display, config['field_mappings'])
    
    # Format currency fields
    currency_fields = config.get('currency_fields', [])
    for field in currency_fields:
        display_name = config['field_mappings'].get(field, field)
        if display_name in df_display.columns:
            df_display[display_name] = pd.to_numeric(df_display[display_name], errors='coerce')
            df_display[display_name] = df_display[display_name].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "N/A"
            )
    
    # Truncate long descriptions
    if 'Description' in df_display.columns:
        df_display['Description'] = df_display['Description'].apply(
            lambda x: str(x)[:100] + "..." if pd.notna(x) and len(str(x)) > 100 else str(x) if pd.notna(x) else ""
        )
    
    # Truncate NAICS Description if too long
    if 'NAICS Description' in df_display.columns:
        df_display['NAICS Description'] = df_display['NAICS Description'].apply(
            lambda x: str(x)[:60] + "..." if pd.notna(x) and len(str(x)) > 60 else str(x) if pd.notna(x) else ""
        )
    
    # Display table
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"government_awards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def render_market_data_collection(db):
    """Render enhanced S&P 500 Market Data collection"""
    config = get_collection_config('MarketData')
    
    st.caption(config['description'])
    
    # Search and limit
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            "🔍 Search by ticker:",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            key="market_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Limit:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="market_limit"
        )
    
    # Build query
    if search_term:
        aql_query = """
        FOR doc IN MarketData
            FILTER LOWER(doc.ticker) LIKE LOWER(@search)
            SORT doc.date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = """
        FOR doc IN MarketData
            SORT doc.date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if not results:
        st.warning("No market data found")
        return
    
    st.success(f"✅ Loaded {len(results)} documents")
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Remove internal fields and embeddings
    cols_to_show = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
    
    df_display = df[cols_to_show].copy()
    
    # Rename columns using field mappings
    df_display = rename_dataframe_columns(df_display, config['field_mappings'])
    
    # Format values
    df_display = format_dataframe_values(df_display, config)
    
    # Display table
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# ============================================================================
# COMPANY COLLECTION RENDERER
# ============================================================================
def render_company_collection(db):
    """Render enhanced Companies collection view - COMPACT TOP 5 LIST"""
    config = get_collection_config('Company')
    
    # Compact search row
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            "🔍 Search companies by ticker, name, sector, or industry:",
            placeholder="e.g., AAPL, Healthcare, Technology",
            key="company_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Results limit:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="company_limit"
        )
    
    # Build query based on search
    if search_term:
        aql_query = """
        FOR doc IN Company
            FILTER LOWER(doc.ticker) LIKE LOWER(@search)
                OR LOWER(doc.name) LIKE LOWER(@search)
                OR LOWER(doc.sector) LIKE LOWER(@search)
                OR LOWER(doc.industry) LIKE LOWER(@search)
            SORT doc.marketCap DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = """
        FOR doc IN Company
            SORT doc.marketCap DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if not results:
        st.warning("No companies found")
        return
    
    # Calculate summary metrics
    df_raw = pd.DataFrame(results)
    total_companies = len(results)
    total_market_cap = df_raw['marketCap'].sum() if 'marketCap' in df_raw.columns else 0
    unique_sectors = df_raw['sector'].nunique() if 'sector' in df_raw.columns else 0
    unique_industries = df_raw['industry'].nunique() if 'industry' in df_raw.columns else 0
    
    # TWO COLUMN LAYOUT: Overview metrics (left) + COMPACT Top 5 List (right)
    col_overview, col_top5 = st.columns([2.5, 2.5])
    
    with col_overview:
        st.markdown("### 📊 Overview")
        
        # 2x2 grid of metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Companies", f"{total_companies:,}")
            st.metric("Sectors", f"{unique_sectors}")
        with m2:
            st.metric("Market Cap", format_currency(total_market_cap))
            st.metric("Industries", f"{unique_industries}")
    
    with col_top5:
        st.markdown("### 🏆 Top 5 by Market Cap")
        
        # Get top 5 - COMPACT LIST
        if 'marketCap' in df_raw.columns and 'ticker' in df_raw.columns:
            top_5 = df_raw.nlargest(5, 'marketCap')[['ticker', 'marketCap']]
            
            # Create compact list of clickable companies
            for idx, (_, row) in enumerate(top_5.iterrows()):
                ticker = row['ticker']
                market_cap = row['marketCap']
                
                # Streamlit button styled to look like a list row
                if st.button(
                    f"🏢 {ticker} · {format_currency(market_cap)}",
                    key=f"top_company_{ticker}",
                    use_container_width=True
                ):
                    st.session_state.show_stock_overview = True
                    st.session_state.selected_ticker = ticker
                    st.rerun()
    
    st.divider()
    
    # Data table and selector side-by-side
    col_table, col_select = st.columns([4, 1])
    
    with col_table:
        st.markdown("### 📋 Company Data")
        
        # Prepare data
        cols_to_show = ['ticker', 'sector', 'industry', 'marketCap', 'sharesOutstanding']
        cols_available = [c for c in cols_to_show if c in df_raw.columns]
        
        df_display = df_raw[cols_available].copy()
        
        # Rename columns
        df_display = rename_dataframe_columns(df_display, config['field_mappings'])
        
        # Format values
        df_display = format_dataframe_values(df_display, config)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=450
        )
    
    with col_select:
        st.markdown("### 🔍 View Details")
        
        if 'ticker' in df_raw.columns:
            selected_ticker = st.selectbox(
                "Select company:",
                options=df_raw['ticker'].tolist(),
                key="company_detail_select"
            )
            
            if st.button("Load Overview", type="primary", use_container_width=True):
                st.session_state.show_stock_overview = True
                st.session_state.selected_ticker = selected_ticker
                st.rerun()
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_generic_collection(db, collection_name):
    """Render generic collection view"""
    config = get_collection_config(collection_name)
    
    st.caption(config['description'])
    
    # Simple search and limit
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            f"🔍 Search {config['display_name']}:",
            placeholder="Enter search term",
            key=f"{collection_name}_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Limit:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key=f"{collection_name}_limit"
        )
    
    # Build query
    if search_term and config.get('searchable_fields'):
        filter_conditions = " OR ".join([
            f"LOWER(doc.{field}) LIKE LOWER(@search)"
            for field in config['searchable_fields']
        ])
        aql_query = f"""
        FOR doc IN {collection_name}
            FILTER {filter_conditions}
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = f"""
        FOR doc IN {collection_name}
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading data..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    if not results:
        st.warning("No data found")
        return
    
    st.success(f"✅ Loaded {len(results)} documents")
    
    # Display data
    df = pd.DataFrame(results)
    cols_to_show = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
    
    if cols_to_show:
        df_display = df[cols_to_show].copy()
        
        # Rename columns
        df_display = rename_dataframe_columns(df_display, config['field_mappings'])
        
        # Format values
        df_display = format_dataframe_values(df_display, config)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Download
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )



# ============================================================================
# GENERIC COLLECTION RENDERER (for other collections)
# ============================================================================
def render_generic_collection(db, collection_name):
    """Render generic collection view"""
    config = get_collection_config(collection_name)
    
    st.caption(config['description'])
    
    # Simple search and limit
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            f"🔍 Search {config['display_name']}:",
            placeholder="Enter search term",
            key=f"{collection_name}_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Limit:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key=f"{collection_name}_limit"
        )
    
    # Build query
    if search_term and config.get('searchable_fields'):
        filter_conditions = " OR ".join([
            f"LOWER(doc.{field}) LIKE LOWER(@search)"
            for field in config['searchable_fields']
        ])
        aql_query = f"""
        FOR doc IN {collection_name}
            FILTER {filter_conditions}
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = f"""
        FOR doc IN {collection_name}
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading data..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    if not results:
        st.warning("No data found")
        return
    
    st.success(f"✅ Loaded {len(results)} documents")
    
    # Display data
    df = pd.DataFrame(results)
    cols_to_show = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
    
    if cols_to_show:
        df_display = df[cols_to_show].copy()
        
        # Rename columns
        df_display = rename_dataframe_columns(df_display, config['field_mappings'])
        
        # Format values
        df_display = format_dataframe_values(df_display, config)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Download
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_awards_collection(db):
    """Render enhanced Government Contract Awards collection"""
    config = get_collection_config('Award')
    
    # Search and limit
    col_search, col_limit = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            "🔍 Search Government Contract Awards:",
            placeholder="Enter search term",
            key="awards_search"
        )
    
    with col_limit:
        limit = st.number_input(
            "Limit:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="awards_limit"
        )
    
    # Build query
    if search_term:
        filter_conditions = " OR ".join([
            f"LOWER(doc.{field}) LIKE LOWER(@search)"
            for field in config['searchable_fields']
        ])
        aql_query = f"""
        FOR doc IN Award
            FILTER {filter_conditions}
            SORT doc.start_date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"search": f"%{search_term}%", "limit": limit}
    else:
        aql_query = """
        FOR doc IN Award
            SORT doc.start_date DESC
            LIMIT @limit
            RETURN doc
        """
        bind_vars = {"limit": limit}
    
    # Execute query
    with st.spinner("Loading..."):
        try:
            cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if not results:
        st.warning("No awards found")
        return
    
    st.success(f"✅ Loaded {len(results)} documents")
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Select columns to display (in order)
    cols_order = ['award_id', 'internal_id', 'recipient_name', 'start_date', 'end_date', 
                  'award_amount', 'awarding_agency', 'description']
    cols_to_show = [c for c in cols_order if c in df.columns]
    
    # Add any remaining columns not in the order list
    remaining_cols = [c for c in df.columns if c not in cols_to_show and not c.startswith('_') and c != 'description_embedding']
    cols_to_show.extend(remaining_cols)
    
    df_display = df[cols_to_show].copy()
    
    # Rename columns using field mappings
    df_display = rename_dataframe_columns(df_display, config['field_mappings'])
    
    # Format award amounts with currency and commas
    if 'Award Amount' in df_display.columns:
        df_display['Award Amount'] = pd.to_numeric(df_display['Award Amount'], errors='coerce')
        df_display['Award Amount'] = df_display['Award Amount'].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "N/A"
        )
    
    # Truncate long descriptions
    if 'Description' in df_display.columns:
        df_display['Description'] = df_display['Description'].apply(
            lambda x: str(x)[:100] + "..." if pd.notna(x) and len(str(x)) > 100 else str(x) if pd.notna(x) else ""
        )
    
    # Display table
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"government_awards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ============================================================================
# STOCK OVERVIEW (unchanged)
# ============================================================================

def get_stock_overview(db, ticker):
    """Get comprehensive overview for a stock ticker"""
    overview = {}
    
    aql_latest = """
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT 1
        RETURN doc
    """
    cursor = db.aql.execute(aql_latest, bind_vars={"ticker": ticker})
    latest = list(cursor)
    if latest:
        overview['latest'] = latest[0]
    
    aql_history = """
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT 90
        RETURN doc
    """
    cursor = db.aql.execute(aql_history, bind_vars={"ticker": ticker})
    overview['history'] = list(cursor)
    
    aql_awards = """
    FOR doc IN Award
        FILTER doc.ticker == @ticker
        SORT doc.start_date DESC
        LIMIT 10
        RETURN doc
    """
    try:
        cursor = db.aql.execute(aql_awards, bind_vars={"ticker": ticker})
        overview['awards'] = list(cursor)
    except:
        overview['awards'] = []
    
    return overview
def render_stock_overview(db, ticker):
    """Render COMPACT stock overview dashboard"""
    import altair as alt
    
    # Compact header with back button inline
    col_title, col_period, col_back = st.columns([1, 2, 1])
    
    with col_title:
        st.markdown(f"## 📈 {ticker}")
    
    with col_period:
        time_period = st.radio(
            "Period:",
            options=["30D", "90D", "6M", "1Y", "5Y"],
            index=1,
            horizontal=True,
            key=f"time_period_{ticker}",
            label_visibility="collapsed"
        )
    
    with col_back:
        st.markdown("")  # Spacing
        if st.button("← Back", key="back_from_stock", use_container_width=True):
            st.session_state.show_stock_overview = False
            st.session_state.selected_ticker = None
            st.rerun()
    
    # Map time period to days
    period_map = {"30D": 30, "90D": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    days = period_map[time_period]
    
    # Get data
    aql_latest = """
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT 1
        RETURN doc
    """
    cursor = db.aql.execute(aql_latest, bind_vars={"ticker": ticker})
    latest = list(cursor)
    
    if not latest:
        st.warning(f"No market data found for {ticker}")
        return
    
    latest = latest[0]
    
    aql_history = f"""
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        SORT doc.date DESC
        LIMIT @days
        RETURN doc
    """
    cursor = db.aql.execute(aql_history, bind_vars={"ticker": ticker, "days": days})
    history = list(cursor)
    
    aql_awards = """
    FOR doc IN Award
        FILTER doc.ticker == @ticker
        SORT doc.start_date DESC
        LIMIT 10
        RETURN doc
    """
    try:
        cursor = db.aql.execute(aql_awards, bind_vars={"ticker": ticker})
        awards = list(cursor)
    except:
        awards = []
    
    # Calculate statistics
    if history and len(history) > 1:
        history_df = pd.DataFrame(history).sort_values('date')
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        period_start = history_df.iloc[-1]['close']
        period_end = history_df.iloc[0]['close']
        period_change = ((period_end - period_start) / period_start * 100)
        period_high = history_df['high'].max()
        period_low = history_df['low'].min()
        volatility = history_df['close'].std()
    else:
        history_df = None
        period_change = 0
        period_high = latest.get('high', 0)
        period_low = latest.get('low', 0)
        volatility = 0
    
    # SINGLE ROW of all metrics - more compact
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        close_price = latest.get('close', 0)
        open_price = latest.get('open', 0)
        change_pct = ((close_price - open_price) / open_price * 100) if open_price else 0
        st.metric("Price", f"${close_price:,.2f}", delta=f"{change_pct:+.2f}%")
    
    with col2:
        st.metric(f"{time_period} Return", f"{period_change:+.2f}%")
    
    with col3:
        st.metric("High", f"${period_high:,.2f}")
    
    with col4:
        st.metric("Low", f"${period_low:,.2f}")
    
    with col5:
        market_cap = latest.get('marketCap')
        st.metric("Market Cap", format_currency(market_cap) if market_cap else "N/A")
    
    with col6:
        st.metric("Volume", format_number(latest.get('volume', 0)))
    
    with col7:
        st.metric("Volatility", f"${volatility:.2f}")
    
    # CHART AND AWARDS SIDE BY SIDE
    col_chart, col_awards = st.columns([2, 1])
    
    with col_chart:
        if history_df is not None and len(history_df) > 1:
            date_format = '%b %d' if time_period in ["30D", "90D"] else '%b %Y' if time_period == "6M" else '%Y-%m'
            
            base = alt.Chart(history_df).encode(
                x=alt.X('date:T', 
                       axis=alt.Axis(
                           title=None,
                           format=date_format,
                           labelAngle=-45,
                           grid=False,
                           labelColor='#e5e7eb'
                       )
                )
            )
            
            area = base.mark_area(opacity=0.15, color='#6366f1').encode(
                y=alt.Y('high:Q', title=None),
                y2='low:Q'
            )
            
            price_line = base.mark_line(color='#10b981', strokeWidth=2.5).encode(
                y=alt.Y('close:Q',
                       axis=alt.Axis(
                           title='Price ($)',
                           format='$.2f',
                           labelColor='#e5e7eb',
                           titleColor='#e5e7eb',
                           grid=True,
                           gridColor='rgba(255,255,255,0.1)'
                       ),
                       scale=alt.Scale(zero=False)
                ),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('close:Q', title='Close', format='$.2f'),
                    alt.Tooltip('high:Q', title='High', format='$.2f'),
                    alt.Tooltip('low:Q', title='Low', format='$.2f'),
                    alt.Tooltip('volume:Q', title='Volume', format=',')
                ]
            )
            
            chart = (area + price_line).properties(height=350).configure_view(
                strokeWidth=0,
                fill='rgba(0,0,0,0)'
            )
            
            st.altair_chart(chart, use_container_width=True, theme=None)
    
    with col_awards:
        if awards and len(awards) > 0:
            st.markdown("### 🏛️ Gov Contracts")
            
            awards_df = pd.DataFrame(awards)
            total_awards = len(awards)
            total_amount = pd.to_numeric(awards_df['award_amount'], errors='coerce').sum()
            
            st.metric("Count", f"{total_awards}")
            st.metric("Total", format_currency(total_amount))
            
            # Compact award list
            for idx, row in awards_df.head(3).iterrows():
                amount = pd.to_numeric(row.get('award_amount', 0), errors='coerce')
                st.markdown(f"**{format_currency(amount)}**  \n{row.get('awarding_agency', 'N/A')[:30]}...")
        else:
            st.info("No government contracts")


# ============================================================================
# MAIN BROWSER
# ============================================================================
def render_database_browser_tab():
    """Render the database browser interface"""
    inject_custom_css()
    
    st.header("🗄️ Database Browser")
    
    db = arango_db.get_arango_connection()
    if not db:
        st.error("Cannot connect to database")
        return
    
    collections_info = arango_db.get_collections_info(db)
    
    # Check if we should show stock overview
    if st.session_state.get('show_stock_overview') and st.session_state.get('selected_ticker'):
        if st.button("← Back to Companies", key="back_to_companies"):
            st.session_state.show_stock_overview = False
            st.session_state.selected_ticker = None
            st.rerun()
        
        ticker = st.session_state.selected_ticker
        render_stock_overview(db, ticker)
        return
    
    # Create tabs
    collection_tabs = []
    for col_info in collections_info:
        config = get_collection_config(col_info['name'])
        collection_tabs.append(f"{config['icon']} {config['display_name']}")
    
    collection_tabs.extend(["📈 Stock Overview", "⚡ Custom AQL"])
    
    selected_tab = st.tabs(collection_tabs)
    
    # Render appropriate view
    for idx, col_info in enumerate(collections_info):
        with selected_tab[idx]:
            collection_name = col_info['name']
            
            # Use specialized renderers
            if collection_name == 'Company':
                render_company_collection(db)
            elif collection_name == 'MarketData':
                render_market_data_collection(db)
            else:
                render_generic_collection(db, collection_name)
    
    # Stock Overview tab
    with selected_tab[-2]:
        st.subheader("📈 Stock Overview")
        
        ticker_input = st.text_input(
            "Enter Ticker Symbol:",
            value="AAPL",
            key="stock_ticker_input"
        ).upper()
        
        if st.button("Load Stock Data", type="primary", key="btn_load_stock"):
            render_stock_overview(db, ticker_input)
    
    # Custom AQL tab
    with selected_tab[-1]:
        st.subheader("⚡ Custom AQL Query")
        
        aql_query = st.text_area(
            "Enter AQL Query:",
            value="FOR doc IN MarketData\n  FILTER doc.ticker == 'AAPL'\n  LIMIT 10\n  RETURN doc",
            height=150,
            key="custom_aql_input"
        )
        
        if st.button("Execute", type="primary", key="btn_execute_aql"):
            results, error = arango_db.execute_custom_aql(db, aql_query)
            if error:
                st.error(f"Query error: {error}")
            else:
                st.success(f"✅ Retrieved {len(results)} documents")
                if results:
                    df = pd.DataFrame(results)
                    cols = [c for c in df.columns if not c.startswith('_') and c != 'description_embedding']
                    if cols:
                        df_display = arango_db.format_dataframe(df[cols])
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
