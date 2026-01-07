import streamlit as st
import pandas as pd 
import config as cfg 
import database as arango_db
import llm as llm 
import ui as ui 
import time
from query_logger import get_logger
from datetime import datetime
import os 
import torch 
import base64


import streamlit.web.server.server as server
server.Server._max_message_size_bytes = 200 * 1024 * 1024  # 200MB for large responses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")
use_local_model = False  # Default to OpenAI GPT-4

st.set_page_config(
    page_title="GraphRAG LLM", 
    page_icon="src/fga-v3.png",
    layout="wide"
)
# Initialize ALL session state variables
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = None
if 'show_custom_aql' not in st.session_state:
    st.session_state.show_custom_aql = False
if 'show_stock_overview' not in st.session_state:
    st.session_state.show_stock_overview = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'saved_queries' not in st.session_state:
    st.session_state.saved_queries = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 1.15rem;
        color: #a8a8a8;
        font-style: italic;
        margin-top: 5px;
    }
    
    /* Follow-up button styling */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    /* Insight cards */
    .insight-card {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    /* Discovery banner */
    .discovery-banner {
        background: linear-gradient(90deg, #ff6b6b22 0%, #feca5722 100%);
        border-left: 4px solid #ff6b6b;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Convert image to base64 for inline embedding
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

icon_base64 = get_base64_image("src/fga-v3.png")

# Header with icon + title
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/png;base64,{icon_base64}" width="70" height="70">
        <div>
            <h1 class="header-title">GraphRAG LLM</h1>
        </div>
    </div>
    <p class="header-subtitle">
        Ask anything. Get answers. Powered by AI & knowledge graphs.
    </p>
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #333;">
    """,
    unsafe_allow_html=True
)

# ==================== HELPER FUNCTIONS ====================

def generate_follow_up_questions(user_question, results, query_plan):
    """Generate contextual follow-up questions based on results"""
    
    intent = query_plan.get('intent', '')
    collections = query_plan.get('collections', [])
    
    follow_ups = []
    
    # Pattern 1: Temporal expansion
    if 'date' in str(results).lower() or any(c in collections for c in ['MarketData', 'EconomicData']):
        follow_ups.append(" How has this changed over the past year?")
        follow_ups.append(" Show me the trend for the last 5 years")
    
    # Pattern 2: Entity expansion (if tickers found)
    if results and 'ticker' in str(results[0]):
        tickers = [r.get('ticker') for r in results[:3] if r.get('ticker')]
        if tickers:
            follow_ups.append(f"¼ Compare {', '.join(tickers[:3])} financial metrics")
            follow_ups.append(f" What are the biggest risks for {tickers[0]}?")
    
    # Pattern 3: Cross-collection expansion
    if 'Award' in collections:
        follow_ups.append(" What's the stock performance for these companies?")
        follow_ups.append(" Do any of these companies have negative SEC sentiment?")
    
    if 'sec_filings' in collections or 'sec_sentences' in collections:
        follow_ups.append("° Show me government contracts for these companies")
        follow_ups.append(" What are their financial metrics?")
    
    if 'MarketData' in collections:
        follow_ups.append(" Have these companies received government contracts?")
        follow_ups.append(" What risks do they mention in SEC filings?")
    
    # Pattern 4: Depth expansion
    if len(results) > 5:
        follow_ups.append(" Show me only the top 3 results with more detail")
    
    # Pattern 5: Comparative analysis
    if len(results) >= 2:
        follow_ups.append(" Compare the top 3 companies side-by-side")
    
    # Pattern 6: "Rabbit in the hat" - unexpected connections
    follow_ups.append(" Find surprising correlations in this data")
    
    return follow_ups[:4]  # Return top 4


def generate_insights(results, query_plan):
    """Auto-generate 2-3 surprising insights from results"""
    
    insights = []
    
    if not results or len(results) == 0:
        return insights
    
    # Insight 1: Outliers
    if len(results) >= 5 and 'award_amount_float' in str(results[0]):
        amounts = [r.get('award_amount_float', 0) for r in results if r.get('award_amount_float')]
        if amounts:
            max_amount = max(amounts)
            avg_amount = sum(amounts) / len(amounts)
            
            if max_amount > avg_amount * 3:
                top_company = next((r for r in results if r.get('award_amount_float') == max_amount), None)
                if top_company:
                    insights.append(f"¡ **Outlier Alert:** {top_company.get('ticker', 'Unknown')} received ${max_amount/1e9:.1f}B â€” **3x more** than the average!")
    
    # Insight 2: Sentiment patterns
    if len(results) >= 3 and any('sentiment' in str(r) or 'avg_finbert' in str(r) for r in results):
        negative_count = sum(1 for r in results if (r.get('avg_sentiment', 0) < -0.5 or r.get('avg_finbert', 0) < -0.5))
        if negative_count > len(results) * 0.6:
            insights.append(f"âš ï¸ **Sentiment Warning:** {negative_count}/{len(results)} companies show **strongly negative** sentiment")
    
    # Insight 3: Temporal patterns
    if len(results) >= 3:
        dates = []
        for r in results:
            date = r.get('date') or r.get('filing_date') or r.get('start_date')
            if date:
                dates.append(date)
        
        if dates:
            recent_dates = [d for d in dates if d and (d.startswith('2024') or d.startswith('2025'))]
            if len(recent_dates) > len(dates) * 0.7:
                insights.append(f"¥ **Recent Trend:** {len(recent_dates)}/{len(dates)} results are from the **last 2 years**")
    
    # Insight 4: Cross-reference
    if len(results) >= 3:
        tickers = [r.get('ticker') for r in results if r.get('ticker')]
        if tickers:
            unique_tickers = len(set(tickers))
            insights.append(f" **Market Coverage:** Found {unique_tickers} unique {'company' if unique_tickers == 1 else 'companies'} in results")
    
    return insights


def auto_cross_reference(results, query_plan):
    """Automatically find related data user didn't ask for"""
    
    discoveries = []
    
    if not results or len(results) == 0:
        return discoveries
    
    collections = query_plan.get('collections', [])
    
    # Extract tickers from results
    tickers = list(set([r.get('ticker') for r in results[:5] if r.get('ticker')]))
    
    # If they queried awards, suggest checking SEC sentiment
    if 'Award' in collections and tickers:
        discoveries.append({
            'title': 'Risk Alert Available',
            'message': f"Want to see SEC sentiment analysis for these {len(tickers)} companies?",
            'action': f"Show me SEC sentiment for {', '.join(tickers[:3])}"
        })
    
    # If they queried sentiment, suggest checking stock performance
    if any(c in collections for c in ['sec_filings', 'sec_sentences', 'sec_sections']) and tickers:
        discoveries.append({
            'title': ' Market Impact Check',
            'message': f"Curious how sentiment correlates with stock performance?",
            'action': f"Show me stock performance for {', '.join(tickers[:3])}"
        })
    
    # If they queried market data, suggest checking contracts
    if 'MarketData' in collections and tickers:
        discoveries.append({
            'title': 'ðŸ›ï¸ Government Contract Discovery',
            'message': f"These companies might have government contracts worth exploring",
            'action': f"Show me government contracts for {', '.join(tickers[:3])}"
        })
    
    return discoveries[:2]  # Max 2 discoveries


def show_summary_metrics(results, query_plan):
    """Show summary metrics for results"""
    
    if not results or len(results) == 0:
        return
    
    st.markdown("###  Quick Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tickers = list(set([r.get('ticker') for r in results if r.get('ticker')]))
        st.metric("Companies", len(tickers))
    
    with col2:
        if 'award_amount_float' in str(results[0]):
            total = sum(r.get('award_amount_float', 0) for r in results)
            st.metric("Total Value", f"${total/1e9:.1f}B" if total > 0 else "N/A")
        elif 'close' in str(results[0]):
            avg_price = sum(r.get('close', 0) for r in results) / len(results)
            st.metric("Avg Price", f"${avg_price:.2f}")
        else:
            st.metric("Results", len(results))
    
    with col3:
        if 'avg_sentiment' in str(results[0]) or 'avg_finbert' in str(results[0]):
            sentiments = [r.get('avg_sentiment', r.get('avg_finbert', 0)) for r in results]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", 
                     delta="Negative" if avg_sentiment < 0 else "Positive",
                     delta_color="inverse" if avg_sentiment < 0 else "normal")
        else:
            st.metric("Data Points", len(results))
    
    with col4:
        dates = [r.get('date') or r.get('filing_date') or r.get('start_date') for r in results if r.get('date') or r.get('filing_date') or r.get('start_date')]
        if dates:
            dates_sorted = sorted([d for d in dates if d])
            if dates_sorted:
                date_range = f"{dates_sorted[0][:4]}-{dates_sorted[-1][:4]}"
                st.metric("Time Range", date_range)
        else:
            st.metric("Status", " Complete")


# Sidebar
with st.sidebar:
    st.header(" Settings")
    
    # # Model toggle
    # use_local_model = st.checkbox(
    #     " Use Local Fine-Tuned Model",
    #     value=False,
    #     help="Toggle between your fine-tuned Llama model (local) and OpenAI GPT-4"
    # )
    
    # if use_local_model:
    #     st.success(" Using Local Model")
    # else:
    #     st.info("Using OpenAI GPT-4")
    
    # st.divider()
    st.info("Using OpenAI GPT-4")
    # Saved Queries
    st.header("Saved Queries")
    
    if st.session_state.saved_queries:
        for i, saved in enumerate(st.session_state.saved_queries[-5:]):
            if st.button(f"– {saved['question'][:35]}...", key=f"saved_{i}", use_container_width=True):
                st.session_state.current_question = saved['question']
                st.rerun()
    else:
        st.caption("No saved queries yet")
    
    st.divider()
    
    # Recent Queries
    st.header("Recent Queries")
    
    if st.session_state.query_history:
        for i, query in enumerate(st.session_state.query_history[-5:]):
            with st.expander(f" {query['question'][:25]}..."):
                st.caption(f" Results: {query['result_count']}")
                st.caption(f" Time: {query['execution_time']:.1f}s")
                if st.button("Re-run", key=f"rerun_{i}", use_container_width=True):
                    st.session_state.current_question = query['question']
                    st.rerun()
    else:
        st.caption("No query history yet")
    
    st.divider()
    
    st.header(" About")
    st.markdown("""
    This platform queries:
    -  Market data
    - Government contracts
    - Macro indicators
    - Commodity positions
    - SEC filings
    """)
    
    st.divider()
    
    # st.header("Quick Examples")
    # examples = [
    #     "Tesla closing price 2020-06-15",
    #     "AAPL EBITDA March 2017",
    #     "Cybersecurity risks in SEC",
    #     "Top defense contracts 2024"
    # ]
    
    # for example in examples:
    #     if st.button(example, key=f"example_{example[:10]}", use_container_width=True):
    #         st.session_state.current_question = example
    #         st.rerun()
    
    st.divider()
    
    if st.button(" Clear Conversation", use_container_width=True, key="clear_conv"):
        st.session_state.conversation_history = []
        st.session_state.query_history = []
        st.rerun()
    
    st.divider()
    # st.caption(f" {cfg.DB_NAME}")
    # st.caption(f" {cfg.ARANGO_URL}")

# Create tabs
tab1, tab2 = st.tabs([" AI Query Interface", "Database Browser"])

# ==================== TAB 1: AI QUERY ====================
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        sample_questions = [
            "What was Tesla's closing price on 2020-06-15?",
            "What was AAPL's closing price on January 6th, 2020?",
            "What was RTX's EBITDA value on March 9th, 2017?",
            "During the month of April 2018 how did AAPL's stock perform?",
            "Which tech companies have the most negative risk sentiment?",
            "Show me cybersecurity risks mentioned in SEC filings",
            "Show top 5 awards for companies with positive SEC sentiment"
        ]
        
        # Check if we have a programmatic question (from follow-up, saved query, etc.)
        if st.session_state.current_question:
            # Use the programmatic question directly
            user_question = st.session_state.current_question
            st.info(f" **Searching:** {user_question}")
            # Don't clear it yet - wait until after search executes
        else:
            # Normal user input
            user_question = st.selectbox(
                "Ask a question about financial data:",
                options=[""] + sample_questions,
                index=0,
                format_func=lambda x: "Type or select a question..." if x == "" else x,
                key="question_input"
            )
            
            if user_question == "":
                user_question = st.text_input(
                    "Or type your custom question:",
                    placeholder="Ask anything about stocks, SEC filings, or contracts...",
                    label_visibility="collapsed",
                    key="custom_question_input"
                )

    with col2:
        # Auto-trigger search if current_question is set
        search_button = st.button(
            "Search", 
            type="primary", 
            use_container_width=True, 
            key="search_btn",
            disabled=not user_question
        ) or bool(st.session_state.current_question)  # â† Auto-trigger on programmatic questions

    # Conversation history
    if st.session_state.conversation_history:
        with st.expander("¬ Conversation History", expanded=False):
            for i, msg in enumerate(st.session_state.conversation_history[-6:]):
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content'][:150]}...")
                elif msg["role"] == "assistant":
                    st.markdown(f"**Assistant:** {msg['content'][:150]}...")

# Query Execution
if (search_button or user_question) and user_question:
    
    logger = get_logger()
    start_time = time.time()
    
    query_plan = None
    results = None
    llm_response = None
    error = None
    intent = None
    
    try:
        # Step 1: Quick intent check
        with st.spinner("Understanding query type..."):
            intent = llm.quick_intent_check(user_question, use_local=use_local_model)
            st.info(f"Detected: {intent.get('type', 'unknown').upper()} query")
        
        # Step 2: Generate query
        with st.spinner("Planning query..."):
            query_plan = llm.plan_query_with_llm(
                user_question, 
                intent_hint=intent,
                use_local=use_local_model
            )
            
            if not query_plan:
                error = "Could not generate query plan"
                st.error(f"{error}")
                st.stop()
        
        # Step 3: Show plan
        with st.expander(" Query Plan & Strategy", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Intent", query_plan.get("intent", "Unknown"))
                st.metric("Collections", ", ".join(query_plan.get("collections", [])))
                st.metric("Model", "Local (Fine-Tuned)" if use_local_model else "OpenAI GPT-4")
            with col_b:
                st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
                st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
            
            st.code(query_plan.get("aql_query", "No query"), language="sql")
            if query_plan.get("bind_vars"):
                st.json(query_plan.get("bind_vars"))
        
        # Step 4: Execute
        with st.spinner("Executing query..."):
            results = llm.execute_planned_query(query_plan)

        if results:
            st.success(f"Retrieved {len(results)} results")
        else:
            st.warning("No results found")
            llm_response = "No results found for your query."
        
        # Step 5: Show Summary Metrics
        if results and len(results) > 0:
            show_summary_metrics(results, query_plan)
        
        # Step 6: Generate Insights
        if results and len(results) > 2:
            insights = generate_insights(results, query_plan)
            if insights:
                st.markdown("### Key Insights")
                for insight in insights:
                    st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        # Step 7: Analysis
        if results:
            with st.spinner(" Analyzing results..."):
                formatted_context = llm.format_results_for_llm(results, query_plan)
                analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
                llm_response = llm.get_llm_analysis(
                    analysis_prompt, 
                    use_local=use_local_model
                )
            
            st.markdown("###  Analysis")
            st.markdown(llm_response)
            
            # Step 8: Auto Cross-Reference (Rabbit in the Hat)
            discoveries = auto_cross_reference(results, query_plan)
            if discoveries:
                st.markdown("### Related Discoveries")
                for disc in discoveries:
                    st.markdown(f'<div class="discovery-banner"><strong>{disc["title"]}</strong>: {disc["message"]}</div>', unsafe_allow_html=True)
                    if st.button(f" {disc['action']}", key=f"disc_{disc['title']}", use_container_width=True):
                        st.session_state.current_question = disc['action']
                        st.rerun()
            
            # Step 9: Follow-up Questions
            st.markdown("---")
            st.markdown("### Explore Further")
            
            follow_ups = generate_follow_up_questions(user_question, results, query_plan)
            
            cols = st.columns(2)
            for i, question in enumerate(follow_ups):
                with cols[i % 2]:
                    clean_question = question
                    for emoji in ["", "¼", "", "", "", "°", "", "ðŸ›ï¸", "âš ï¸", "ðŸŽ¯", "ðŸ“‹", "âš–ï¸", "ðŸŽ©"]:
                        clean_question = clean_question.replace(f"{emoji} ", "")
                    
                    if st.button(question, key=f"followup_{i}", use_container_width=True):
                        st.session_state.current_question = clean_question
                        st.rerun()
            
            # Step 10: Save Query Option
            col_save1, col_save2 = st.columns([1, 3])
            with col_save1:
                if st.button("Save Query", use_container_width=True):
                    st.session_state.saved_queries.append({
                        'question': user_question,
                        'timestamp': datetime.now().isoformat(),
                        'result_count': len(results)
                    })
                    st.success("Query saved!")
            
            # Raw data
            with st.expander("View Raw Data", expanded=False):
                try:
                    df = pd.DataFrame(results)
                    cols_to_show = [col for col in df.columns if not col.startswith('_') and col != 'description_embedding']
                    if cols_to_show:
                        st.dataframe(df[cols_to_show], use_container_width=True)
                    else:
                        st.json(results[:10])
                except Exception as e:
                    st.json(results[:10])
                    st.caption(f"Could not format as table: {str(e)}")
            
            # Debug
            with st.expander("Debug: LLM Context", expanded=False):
                st.text(formatted_context[:3000])
                if len(formatted_context) > 3000:
                    st.caption("(Truncated for display)")
        
        else:
            st.info("¡ Try rephrasing your question or check if the data exists in the database.")
            st.markdown("**Suggestions:**")
            st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
            st.markdown("- Check date formats (YYYY-MM-DD)")
            st.markdown("- Ensure the collection contains relevant data")
    
    except Exception as e:
        error = str(e)
        st.error(f"âŒ Error: {error}")
        llm_response = f"Query failed with error: {error}"
    
    finally:
        execution_time = time.time() - start_time
        
        # Add to query history
        st.session_state.query_history.append({
            'question': user_question,
            'result_count': len(results) if results else 0,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log everything
        query_id = logger.log_query(
            user_question=user_question,
            query_plan=query_plan,
            results=results,
            llm_response=llm_response,
            execution_time=execution_time,
            error=error,
            metadata={
                "model": "local_finetuned" if use_local_model else "gpt-4o-mini",
                "intent_type": intent.get('type') if intent else None,
                "intent_value": intent.get('value') if intent else None,
                "result_count": len(results) if results else 0,
                "user_agent": "streamlit_app"
            }
        )
        
        # Show query ID and feedback
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.caption(f" Query ID: `{query_id}`")
        
        with col2:
            if st.button("Good", key=f"thumbs_up_{query_id}"):
                logger.log_feedback(query_id, "positive", "User clicked thumbs up")
                st.success("Thanks!")
        
        with col3:
            if st.button("Bad", key=f"thumbs_down_{query_id}"):
                logger.log_feedback(query_id, "negative", "User clicked thumbs down")
                st.warning("We'll review this!")
        
        with col4:
            st.caption(f" {execution_time:.2f}s")
if 'logger' not in st.session_state:
    st.session_state.logger = get_logger()

logger = st.session_state.logger

# Session stats in sidebar
st.sidebar.markdown("---")
st.sidebar.header(" Session Stats")

try:
    summary = logger.get_session_summary()
    
    if "message" not in summary:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Queries", summary['total_queries'])
            st.metric("Success", summary['successful'])
        with col2:
            st.metric("Failed", summary['failed'])
            st.metric("Avg Time", summary['avg_execution_time'])
        
        if summary['collections_used']:
            with st.sidebar.expander("Collections Used"):
                for coll, count in sorted(summary['collections_used'].items(), key=lambda x: x[1], reverse=True):
                    st.caption(f"{coll}: {count}x")
        
        if st.sidebar.button("Download Logs"):
            csv_file = logger.export_to_csv()
            if csv_file:
                with open(csv_file, 'r') as f:
                    st.sidebar.download_button(
                        label="¾ Download CSV",
                        data=f.read(),
                        file_name=f"session_{logger.session_id}.csv",
                        mime="text/csv"
                    )
    else:
        st.sidebar.info("No queries yet")

except Exception as e:
    st.sidebar.warning(" Stats temporarily unavailable")

# ==================== TAB 2: DATABASE BROWSER ====================
with tab2:
    ui.render_database_browser_tab()

# Footer
st.divider()
st.caption("Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")


# import streamlit as st
# import pandas as pd 
# import config as cfg 
# import database as arango_db
# import llm as llm 
# import ui as ui 

# import streamlit as st
# import os 
# import torch 
# import base64
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"  Using device: {device}")

# st.set_page_config(
#     page_title="GraphRAG LLM", 
#     page_icon="src/fga-v3.png",
#     layout="wide"
# )

# # Initialize ALL session state variables
# if 'selected_collection' not in st.session_state:
#     st.session_state.selected_collection = None
# if 'show_custom_aql' not in st.session_state:
#     st.session_state.show_custom_aql = False
# if 'show_stock_overview' not in st.session_state:
#     st.session_state.show_stock_overview = False
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     /* Reduce top padding */
#     .block-container {
#         padding-top: 2rem;
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
    
#     /* Custom header styling */
#     .header-container {
#         display: flex;
#         align-items: center;
#         gap: 20px;
#         margin-bottom: 10px;
#     }
    
#     .header-title {
#         font-size: 3.5rem;
#         font-weight: 700;
#         margin: 0;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         line-height: 1.2;
#     }
    
#     .header-subtitle {
#         font-size: 1.15rem;
#         color: #a8a8a8;
#         font-style: italic;
#         margin-top: 5px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Convert image to base64 for inline embedding
# def get_base64_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# icon_base64 = get_base64_image("src/fga-v3.png")

# # Header with icon + title
# st.markdown(
#     f"""
#     <div class="header-container">
#         <img src="data:image/png;base64,{icon_base64}" width="70" height="70">
#         <div>
#             <h1 class="header-title">GraphRAG LLM</h1>
#         </div>
#     </div>
#     <p class="header-subtitle">
#         Ask anything. Get answers. Powered by AI & knowledge graphs.
#     </p>
#     <hr style="margin: 20px 0; border: none; border-top: 1px solid #333;">
#     """,
#     unsafe_allow_html=True
# )


# # Sidebar
# with st.sidebar:
#     st.header("âš™ï¸ Settings")
    
#     # Add model toggle HERE
#     use_local_model = st.checkbox(
#         " Use Local Fine-Tuned Model",
#         value=False,  # Start with OpenAI by default
#         help="Toggle between your fine-tuned Llama model (local) and OpenAI GPT-4"
#     )
    
#     if use_local_model:
#         st.success("âœ… Using Local Model")
#     else:
#         st.info("â˜ï¸ Using OpenAI GPT-4")
    
#     st.divider()
    
#     st.header(" About")
#     st.markdown("""
#     This platform queries:
#     -  Market data
#     - ðŸ›ï¸ Government contracts
#     -  Macro indicators
#     - ðŸŒ¾ Commodity positions
#     - ðŸ“„ SEC filings
#     """)
    
#     st.divider()
    
#     st.header("¡ Examples")
#     st.markdown("""
#     - "What was Tesla's closing price on 2020-06-15?"
#     - "What was AAPLâ€™s closing price on January 6th, 2020"
#     - "What was RTXs EBITDA value on March 9th, 2017"
#     - "During the month of April 2018 how did AAPLs stock perform?"
#     """)
    
#     st.divider()
    
#     if st.button(" Clear Conversation", use_container_width=True, key="clear_conv"):
#         st.session_state.conversation_history = []
#         st.rerun()
    
#     st.divider()
#     st.caption(f" {cfg.DB_NAME}")
#     st.caption(f" {cfg.ARANGO_URL}")


# # Create tabs
# tab1, tab2 = st.tabs([" AI Query Interface", "ðŸ—„ï¸ Database Browser"])

# # ==================== TAB 1: AI QUERY ====================
# with tab1:
#     col1, col2 = st.columns([3, 1])

#     with col1:
#         sample_questions = [
#             "What was Tesla's closing price on 2020-06-15?",
#             "What was AAPL's closing price on January 6th, 2020?",
#             "What was RTX's EBITDA value on March 9th, 2017?",
#             "During the month of April 2018 how did AAPL's stock perform?",
#             "Which tech companies have the most negative risk sentiment?",
#             "Show me cybersecurity risks mentioned in SEC filings",
#             "Compare risk sentiment between AAPL and MSFT"
#         ]
        
#         # Single selectbox that allows custom input
#         user_question = st.selectbox(
#             "Ask a question about financial data:",
#             options=[""] + sample_questions,
#             index=0,
#             format_func=lambda x: "Type or select a question..." if x == "" else x,
#             key="question_input"
#         )
        
#         # If empty option selected, show text input instead
#         if user_question == "":
#             user_question = st.text_input(
#                 "Or type your custom question:",
#                 placeholder="Ask anything about stocks, SEC filings, or contracts...",
#                 label_visibility="collapsed",
#                 key="custom_question_input"
#             )

#     with col2:
#         search_button = st.button(
#             "Ž Search", 
#             type="primary", 
#             use_container_width=True, 
#             key="search_btn",
#             disabled=not user_question  # Disable if empty
#         )

#     # Conversation history
#     if st.session_state.conversation_history:
#         with st.expander("¬ Conversation History", expanded=False):
#             for i, msg in enumerate(st.session_state.conversation_history[-6:]):
#                 if msg["role"] == "user":
#                     st.markdown(f"**You:** {msg['content'][:150]}...")
#                 elif msg["role"] == "assistant":
#                     st.markdown(f"**Assistant:** {msg['content'][:150]}...")


# # Query Execution
# import time
# from logger import get_logger

# if (search_button or user_question) and user_question:
    
#     # Initialize logger and tracking
#     logger = get_logger()
#     start_time = time.time()
    
#     query_plan = None
#     results = None
#     llm_response = None
#     error = None
#     intent = None
    
#     try:
#         # Step 1: Quick intent check
#         with st.spinner("ðŸ§  Understanding query type..."):
#             intent = llm.quick_intent_check(user_question, use_local=use_local_model)
#             st.info(f"ðŸŽ¯ Detected: {intent.get('type', 'unknown').upper()} query")
        
#         # Step 2: Generate query with intent hint
#         with st.spinner("âš™ï¸ Planning query..."):
#             query_plan = llm.plan_query_with_llm(
#                 user_question, 
#                 intent_hint=intent,
#                 use_local=use_local_model
#             )
            
#             if not query_plan:
#                 error = "Could not generate query plan"
#                 st.error(f"âŒ {error}")
#                 st.stop()
        
#         # Step 3: Show plan
#         with st.expander(" Query Plan & Strategy", expanded=False):
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.metric("Intent", query_plan.get("intent", "Unknown"))
#                 st.metric("Collections", ", ".join(query_plan.get("collections", [])))
#                 st.metric("Model", "Local (Fine-Tuned)" if use_local_model else "OpenAI GPT-4")
#             with col_b:
#                 st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
#                 st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
            
#             st.code(query_plan.get("aql_query", "No query"), language="sql")
#             if query_plan.get("bind_vars"):
#                 st.json(query_plan.get("bind_vars"))
        
#         # Step 4: Execute
#         with st.spinner("âš¡ Executing query..."):
#             results = llm.execute_planned_query(query_plan)

#         if results:
#             st.success(f"âœ… Retrieved {len(results)} results")
#         else:
#             st.warning("âš ï¸ No results found")
#             llm_response = "No results found for your query."
        
#         # Step 5: Analysis
#         if results:
#             with st.spinner(" Analyzing results..."):
#                 formatted_context = llm.format_results_for_llm(results, query_plan)
#                 analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
#                 llm_response = llm.get_llm_analysis(
#                     analysis_prompt, 
#                     use_local=use_local_model
#                 )
            
#             st.markdown("###  Analysis")
#             st.markdown(llm_response)

#              # Add to your UI after showing results:
#             st.markdown("---")
#             st.markdown("### ¡ Explore Further")

#             # follow_ups = llm.generate_follow_up_questions(user_question, results, query_plan)

#             # cols = st.columns(2)
#             # for i, question in enumerate(follow_ups):
#             #     with cols[i % 2]:
#             #         if st.button(question, key=f"followup_{i}", use_container_width=True):
#             #             st.session_state.user_question = question.replace(" ", "").replace("¼ ", "").replace(" ", "").replace(" ", "").replace(" ", "").replace("° ", "").replace(" ", "").replace("ðŸ›ï¸ ", "").replace("âš ï¸ ", "").replace("ðŸŽ¯ ", "").replace("ðŸ“‹ ", "").replace("âš–ï¸ ", "").replace("ðŸŽ© ", "")
#             #             st.rerun()
            
#             # Raw data
#             with st.expander("ðŸ“‹ View Raw Data", expanded=False):
#                 try:
#                     df = pd.DataFrame(results)
#                     cols_to_show = [col for col in df.columns if not col.startswith('_') and col != 'description_embedding']
#                     if cols_to_show:
#                         st.dataframe(df[cols_to_show], use_container_width=True)
#                     else:
#                         st.json(results[:10])
#                 except Exception as e:
#                     st.json(results[:10])
#                     st.caption(f"Could not format as table: {str(e)}")
            
#             # Debug
#             with st.expander("§ Debug: LLM Context", expanded=False):
#                 st.text(formatted_context[:3000])
#                 if len(formatted_context) > 3000:
#                     st.caption("(Truncated for display)")
        
#         else:
#             st.info("¡ Try rephrasing your question or check if the data exists in the database.")
#             st.markdown("**Suggestions:**")
#             st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
#             st.markdown("- Check date formats (YYYY-MM-DD)")
#             st.markdown("- Ensure the collection contains relevant data")
    
#     except Exception as e:
#         error = str(e)
#         st.error(f"âŒ Error: {error}")
#         llm_response = f"Query failed with error: {error}"
    
#     finally:
#         # Calculate execution time
#         execution_time = time.time() - start_time
        
#         # Log everything
#         query_id = logger.log_query(
#             user_question=user_question,
#             query_plan=query_plan,
#             results=results,
#             llm_response=llm_response,
#             execution_time=execution_time,
#             error=error,
#             metadata={
#                 "model": "local_finetuned" if use_local_model else "gpt-4o-mini",
#                 "intent_type": intent.get('type') if intent else None,
#                 "intent_value": intent.get('value') if intent else None,
#                 "result_count": len(results) if results else 0,
#                 "user_agent": "streamlit_app"
#             }
#         )
        
#         # Show query ID and feedback options
#         st.markdown("---")
#         col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
#         with col1:
#             st.caption(f" Query ID: `{query_id}`")
        
#         with col2:
#             if st.button("ðŸ‘ Good", key=f"thumbs_up_{query_id}"):
#                 logger.log_feedback(query_id, "positive", "User clicked thumbs up")
#                 st.success("Thanks for the feedback!")
        
#         with col3:
#             if st.button("ðŸ‘Ž Bad", key=f"thumbs_down_{query_id}"):
#                 logger.log_feedback(query_id, "negative", "User clicked thumbs down")
#                 st.warning("Thanks! We'll review this query.")
        
#         with col4:
#             st.caption(f" {execution_time:.2f}s")


# # Add session stats to sidebar
# st.sidebar.markdown("---")
# st.sidebar.header(" Session Stats")

# logger = get_logger()
# summary = logger.get_session_summary()

# if "message" not in summary:
#     col1, col2 = st.sidebar.columns(2)
#     with col1:
#         st.metric("Queries", summary['total_queries'])
#         st.metric("Success", summary['successful'])
#     with col2:
#         st.metric("Failed", summary['failed'])
#         st.metric("Avg Time", summary['avg_execution_time'])
    
#     # Show collections used
#     if summary['collections_used']:
#         with st.sidebar.expander("ðŸ“š Collections Used"):
#             for coll, count in sorted(summary['collections_used'].items(), key=lambda x: x[1], reverse=True):
#                 st.caption(f"{coll}: {count}x")
    
#     # Download session logs
#     if st.sidebar.button("ðŸ“¥ Download Logs"):
#         csv_file = logger.export_to_csv()
#         if csv_file:
#             with open(csv_file, 'r') as f:
#                 st.sidebar.download_button(
#                     label="¾ Download CSV",
#                     data=f.read(),
#                     file_name=f"session_{logger.session_id}.csv",
#                     mime="text/csv"
#                 )
# else:
#     st.sidebar.info("No queries yet")





        

# # ==================== TAB 2: DATABASE BROWSER ====================
# with tab2:
#     ui.render_database_browser_tab()

# # Footer
# st.divider()
# st.caption("ðŸš€ Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")

# import streamlit as st
# import pandas as pd
# import config as cfg
# import database as arango_db
# import llm as llm
# import time
# from datetime import datetime
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# st.set_page_config(
#     page_title="Quant Intelligence",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'current_results' not in st.session_state:
#     st.session_state.current_results = None
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = ""
# if 'current_analysis' not in st.session_state:
#     st.session_state.current_analysis = None
# if 'current_time' not in st.session_state:
#     st.session_state.current_time = 0
# if 'page' not in st.session_state:
#     st.session_state.page = "search"

# # Dark Forest Green + LSU Purple Theme CSS
# st.markdown("""
# <style>
#     /* Main dark forest background */
#     .stApp {
#         background: linear-gradient(135deg, #0a1612 0%, #0d1f1a 50%, #0f1419 100%);
#     }

#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0a1612 0%, #0d1f1a 100%);
#         border-right: 1px solid rgba(70, 45, 85, 0.3);
#         width: 220px !important;
#     }

#     [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
#         color: #b8c5c0;
#         font-size: 13px;
#     }

#     /* Sidebar buttons */
#     [data-testid="stSidebar"] .stButton button {
#         background: linear-gradient(135deg, #1a3a2e 0%, #16302a 100%);
#         color: #e0ebe7;
#         border: 1px solid rgba(90, 70, 100, 0.4);
#         border-radius: 8px;
#         padding: 10px 16px;
#         font-weight: 500;
#         width: 100%;
#         transition: all 0.3s ease;
#     }

#     [data-testid="stSidebar"] .stButton button:hover {
#         background: linear-gradient(135deg, #234d3d 0%, #1f4237 100%);
#         border-color: rgba(130, 70, 150, 0.6);
#         box-shadow: 0 4px 12px rgba(70, 45, 85, 0.4);
#         transform: translateY(-1px);
#     }

#     /* Center container */
#     .center-box {
#         max-width: 900px;
#         margin: 0 auto;
#         padding: 60px 20px 40px 20px;
#     }

#     /* Title styling - NO BOX, centered, larger */
#     .main-title {
#         text-align: center;
#         margin-bottom: 15px;
#         padding: 0;
#         background: none;
#         border: none;
#     }

#     .main-title h1 {
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-size: 4rem;
#         font-weight: 700;
#         margin: 0;
#         letter-spacing: -0.02em;
#         line-height: 1.2;
#     }

#     .subtitle {
#         text-align: center;
#         color: #8fa9a0;
#         font-size: 16px;
#         margin-bottom: 50px;
#         font-weight: 400;
#     }

#     /* Input field styling */
#     .stTextInput input {
#         background: rgba(30, 50, 40, 0.5) !important;
#         border: 1.5px solid rgba(100, 70, 110, 0.4) !important;
#         border-radius: 10px !important;
#         color: #e8f3ef !important;
#         font-size: 15px !important;
#         padding: 14px 18px !important;
#         transition: all 0.3s ease !important;
#     }

#     .stTextInput input:focus {
#         border-color: rgba(120, 81, 169, 0.7) !important;
#         box-shadow: 0 0 0 2px rgba(120, 81, 169, 0.15) !important;
#         background: rgba(35, 55, 45, 0.6) !important;
#     }

#     .stTextInput input::placeholder {
#         color: #6b8078 !important;
#     }

#     .stTextInput label {
#         color: #8fa9a0;
#         font-size: 14px;
#         font-weight: 500;
#         margin-bottom: 8px;
#     }

#     /* Search button - exclude sidebar buttons */
#     .center-box .stButton button {
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%) !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 10px !important;
#         padding: 12px 32px !important;
#         font-size: 15px !important;
#         font-weight: 600 !important;
#         transition: all 0.3s ease !important;
#         box-shadow: 0 4px 12px rgba(120, 81, 169, 0.3) !important;
#         margin: 20px auto 0 auto !important;
#         display: block !important;
#     }

#     .center-box .stButton button:hover {
#         transform: translateY(-2px) !important;
#         box-shadow: 0 6px 20px rgba(120, 81, 169, 0.5) !important;
#     }

#     /* Tables */
#     .dataframe {
#         background: rgba(20, 35, 30, 0.4) !important;
#         border: 1px solid rgba(80, 60, 90, 0.3) !important;
#         border-radius: 8px !important;
#         color: #d5e5df !important;
#     }

#     .dataframe th {
#         background: rgba(40, 65, 55, 0.6) !important;
#         color: #b8ddd0 !important;
#         font-weight: 600 !important;
#         padding: 12px !important;
#         border-bottom: 2px solid rgba(120, 81, 169, 0.4) !important;
#     }

#     .dataframe td {
#         padding: 10px !important;
#         border-bottom: 1px solid rgba(60, 50, 70, 0.2) !important;
#     }

#     /* Metrics */
#     [data-testid="stMetricValue"] {
#         color: #c5e8d8 !important;
#         font-size: 24px !important;
#         font-weight: 600 !important;
#     }

#     [data-testid="stMetricLabel"] {
#         color: #8fa9a0 !important;
#         font-size: 13px !important;
#     }

#     /* Expander */
#     .streamlit-expanderHeader {
#         background: rgba(30, 50, 40, 0.4) !important;
#         border: 1px solid rgba(80, 60, 90, 0.3) !important;
#         border-radius: 8px !important;
#         color: #b8ddd0 !important;
#     }

#     .streamlit-expanderContent {
#         background: rgba(20, 35, 30, 0.3) !important;
#         border: 1px solid rgba(70, 50, 80, 0.2) !important;
#         border-top: none !important;
#     }

#     /* Loading animation */
#     .loading-dots {
#         display: flex;
#         justify-content: center;
#         gap: 8px;
#     }

#     .loading-dot {
#         width: 12px;
#         height: 12px;
#         border-radius: 50%;
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%);
#         animation: bounce 1.4s infinite ease-in-out both;
#     }

#     .loading-dot:nth-child(1) { animation-delay: -0.32s; }
#     .loading-dot:nth-child(2) { animation-delay: -0.16s; }

#     @keyframes bounce {
#         0%, 80%, 100% { transform: scale(0); }
#         40% { transform: scale(1); }
#     }

#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#         height: 10px;
#     }

#     ::-webkit-scrollbar-track {
#         background: rgba(15, 25, 20, 0.3);
#     }

#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(180deg, #3d8b6f 0%, #7851a9 100%);
#         border-radius: 5px;
#     }

#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(180deg, #4a9d7e 0%, #8a5ec2 100%);
#     }

#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}

#     /* Spinner */
#     .stSpinner > div {
#         border-top-color: #7851a9 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# def render_search_interface():
#     """Render clean search interface"""

#     # Title section - NO BOX, just centered text
#     st.markdown("""
#         <div class="main-title">
#             <h1>Quant Intelligence</h1>
#         </div>
#         <div class="subtitle">
#             Natural language queries across financial data, SEC filings, and market intelligence
#         </div>
#     """, unsafe_allow_html=True)

#     # Search container
#     st.markdown('<div class="search-container">', unsafe_allow_html=True)

#     # Search input
#     user_question = st.text_input(
#         "Ask a question",
#         value=st.session_state.current_question,
#         placeholder="Ask about companies, SEC filings, contracts, or market data...",
#         key="search_input",
#         label_visibility="collapsed"
#     )

#     # Search button - centered under search bar
#     search_clicked = st.button("🔍 Search", use_container_width=False)

#     st.markdown('</div>', unsafe_allow_html=True)

#     return user_question, search_clicked

# def render_results(results, analysis, question, query_time):
#     """Render results with analysis"""
#     st.markdown('<div class="results-box">', unsafe_allow_html=True)

#     # Display analysis text
#     if analysis:
#         st.markdown(f"### Answer")
#         st.markdown(analysis)
#         st.markdown("---")

#     # Metrics
#     if results and len(results) > 0:
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Results", len(results))
#         with col2:
#             st.metric("Execution Time", f"{query_time:.2f}s")
#         with col3:
#             st.metric("Fields", len(results[0].keys()) if results else 0)

#         st.markdown("---")

#         # Display results as DataFrame
#         df = pd.DataFrame(results)

#         # Format numeric columns
#         for col in df.columns:
#             if df[col].dtype in ['float64', 'float32']:
#                 df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

#         st.dataframe(df, use_container_width=True, height=400)
#     else:
#         st.info("No results found. Try rephrasing your query.")

#     st.markdown('</div>', unsafe_allow_html=True)

# def main():
#     """Main application"""

#     # Sidebar
#     with st.sidebar:
#         st.markdown("### Settings")

#         if st.button("⚡ New Search"):
#             st.session_state.current_results = None
#             st.session_state.current_question = ""
#             st.session_state.current_analysis = None
#             st.rerun()

#         st.markdown("---")

#         if st.button("🗄️ Database Explorer"):
#             st.session_state.page = "database"
#             st.rerun()

#         if st.button("📊 Query History"):
#             st.session_state.page = "history"
#             st.rerun()

#         st.markdown("---")
#         st.caption("Powered by ArangoDB + LLM")

#     # Main content area
#     st.markdown('<div class="center-box">', unsafe_allow_html=True)

#     if not st.session_state.get('current_results'):
#         # Search interface
#         question, search_clicked = render_search_interface()

#         # Handle search
#         if search_clicked and question:
#             with st.spinner(""):
#                 st.markdown("""
#                     <div style='text-align: center; padding: 2rem;'>
#                         <div class='loading-dots'>
#                             <div class='loading-dot'></div>
#                             <div class='loading-dot'></div>
#                             <div class='loading-dot'></div>
#                         </div>
#                         <div style='margin-top: 1rem; color: #9aa0a6;'>Analyzing your query...</div>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 start_time = time.time()

#                 try:
#                     # Step 1: Quick intent check (optional but recommended)
#                     intent = llm.quick_intent_check(question, use_local=False)

#                     # Step 2: Plan query with LLM
#                     plan = llm.plan_query_with_llm(
#                         question,
#                         intent_hint=intent,
#                         use_local=False
#                     )

#                     # Step 3: Execute the planned query
#                     results = llm.execute_planned_query(plan)

#                     # Step 4: Generate analysis
#                     if results and len(results) > 0:
#                         # Format results for LLM context
#                         formatted_context = llm.format_results_for_llm(results, plan)

#                         # Create analysis prompt
#                         analysis_prompt = llm.create_analysis_prompt(
#                             question, 
#                             formatted_context, 
#                             plan
#                         )

#                         # Get LLM analysis
#                         analysis = llm.get_llm_analysis(
#                             analysis_prompt,
#                             use_local=False
#                         )
#                     else:
#                         analysis = "No matching data found for your query."

#                     query_time = time.time() - start_time

#                     # Store in session
#                     st.session_state.current_results = results
#                     st.session_state.current_analysis = analysis
#                     st.session_state.current_question = question
#                     st.session_state.current_time = query_time

#                     st.rerun()

#                 except Exception as e:
#                     st.error(f"Query failed: {str(e)}")
#                     import traceback
#                     st.code(traceback.format_exc())

#     else:
#         # Show results
#         render_results(
#             st.session_state.current_results,
#             st.session_state.current_analysis,
#             st.session_state.current_question,
#             st.session_state.current_time
#         )

#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import config as cfg
# import database as arango_db
# import llm as llm
# import time
# from datetime import datetime
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# st.set_page_config(
#     page_title="Quant Intelligence",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'current_results' not in st.session_state:
#     st.session_state.current_results = None
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = ""
# if 'current_analysis' not in st.session_state:
#     st.session_state.current_analysis = None
# if 'current_time' not in st.session_state:
#     st.session_state.current_time = 0
# if 'page' not in st.session_state:
#     st.session_state.page = "search"

# # Dark Forest Green + LSU Purple Theme CSS
# st.markdown("""
# <style>
#     /* Main dark forest background */
#     .stApp {
#         background: linear-gradient(135deg, #0a1612 0%, #0d1f1a 50%, #0f1419 100%);
#     }

#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0a1612 0%, #0d1f1a 100%);
#         border-right: 1px solid rgba(70, 45, 85, 0.3);
#         width: 220px !important;
#     }

#     [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
#         color: #b8c5c0;
#         font-size: 13px;
#     }

#     /* Sidebar buttons */
#     [data-testid="stSidebar"] .stButton button {
#         background: linear-gradient(135deg, #1a3a2e 0%, #16302a 100%);
#         color: #e0ebe7;
#         border: 1px solid rgba(90, 70, 100, 0.4);
#         border-radius: 8px;
#         padding: 10px 16px;
#         font-weight: 500;
#         width: 100%;
#         transition: all 0.3s ease;
#     }

#     [data-testid="stSidebar"] .stButton button:hover {
#         background: linear-gradient(135deg, #234d3d 0%, #1f4237 100%);
#         border-color: rgba(130, 70, 150, 0.6);
#         box-shadow: 0 4px 12px rgba(70, 45, 85, 0.4);
#         transform: translateY(-1px);
#     }

#     /* Center container */
#     .center-box {
#         max-width: 900px;
#         margin: 0 auto;
#         padding: 60px 20px 40px 20px;
#     }

#     /* Title styling - NO BOX, centered, larger */
#     .main-title {
#         text-align: center;
#         margin-bottom: 15px;
#         padding: 0;
#         background: none;
#         border: none;
#     }

#     .main-title h1 {
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-size: 4rem;
#         font-weight: 700;
#         margin: 0;
#         letter-spacing: -0.02em;
#         line-height: 1.2;
#     }

#     .subtitle {
#         text-align: center;
#         color: #8fa9a0;
#         font-size: 16px;
#         margin-bottom: 50px;
#         font-weight: 400;
#     }

#     /* Search container - positioned above results */

#     # /* Results container */
#     # .results-box {
#     #     background: rgba(15, 28, 23, 0.85);
#     #     border: 1px solid rgba(70, 50, 80, 0.4);
#     #     border-radius: 12px;
#     #     padding: 28px;
#     #     backdrop-filter: blur(10px);
#     #     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
#     # }

#     /* Input field styling */
#     .stTextInput input {
#         background: rgba(30, 50, 40, 0.5) !important;
#         border: 1.5px solid rgba(100, 70, 110, 0.4) !important;
#         border-radius: 10px !important;
#         color: #e8f3ef !important;
#         font-size: 15px !important;
#         padding: 14px 18px !important;
#         transition: all 0.3s ease !important;
#     }

#     .stTextInput input:focus {
#         border-color: rgba(120, 81, 169, 0.7) !important;
#         box-shadow: 0 0 0 2px rgba(120, 81, 169, 0.15) !important;
#         background: rgba(35, 55, 45, 0.6) !important;
#     }

#     .stTextInput input::placeholder {
#         color: #6b8078 !important;
#     }

#     .stTextInput label {
#         color: #8fa9a0;
#         font-size: 14px;
#         font-weight: 500;
#         margin-bottom: 8px;
#     }

#     /* Search button */
#     .stButton button {
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%) !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 10px !important;
#         padding: 12px 32px !important;
#         font-size: 15px !important;
#         font-weight: 600 !important;
#         transition: all 0.3s ease !important;
#         box-shadow: 0 4px 12px rgba(120, 81, 169, 0.3) !important;
#     }

#     .stButton button:hover {
#         transform: translateY(-2px) !important;
#         box-shadow: 0 6px 20px rgba(120, 81, 169, 0.5) !important;
#     }

#     /* Center search button container */
#     .search-container .stButton {
#         display: flex !important;
#         justify-content: center !important;
#         margin-top: 20px !important;
#     }

#     /* Tables */
#     .dataframe {
#         background: rgba(20, 35, 30, 0.4) !important;
#         border: 1px solid rgba(80, 60, 90, 0.3) !important;
#         border-radius: 8px !important;
#         color: #d5e5df !important;
#     }

#     .dataframe th {
#         background: rgba(40, 65, 55, 0.6) !important;
#         color: #b8ddd0 !important;
#         font-weight: 600 !important;
#         padding: 12px !important;
#         border-bottom: 2px solid rgba(120, 81, 169, 0.4) !important;
#     }

#     .dataframe td {
#         padding: 10px !important;
#         border-bottom: 1px solid rgba(60, 50, 70, 0.2) !important;
#     }

#     /* Metrics */
#     [data-testid="stMetricValue"] {
#         color: #c5e8d8 !important;
#         font-size: 24px !important;
#         font-weight: 600 !important;
#     }

#     [data-testid="stMetricLabel"] {
#         color: #8fa9a0 !important;
#         font-size: 13px !important;
#     }

#     /* Expander */
#     .streamlit-expanderHeader {
#         background: rgba(30, 50, 40, 0.4) !important;
#         border: 1px solid rgba(80, 60, 90, 0.3) !important;
#         border-radius: 8px !important;
#         color: #b8ddd0 !important;
#     }

#     .streamlit-expanderContent {
#         background: rgba(20, 35, 30, 0.3) !important;
#         border: 1px solid rgba(70, 50, 80, 0.2) !important;
#         border-top: none !important;
#     }

#     /* Loading animation */
#     .loading-dots {
#         display: flex;
#         justify-content: center;
#         gap: 8px;
#     }

#     .loading-dot {
#         width: 12px;
#         height: 12px;
#         border-radius: 50%;
#         background: linear-gradient(135deg, #3d8b6f 0%, #7851a9 100%);
#         animation: bounce 1.4s infinite ease-in-out both;
#     }

#     .loading-dot:nth-child(1) { animation-delay: -0.32s; }
#     .loading-dot:nth-child(2) { animation-delay: -0.16s; }

#     @keyframes bounce {
#         0%, 80%, 100% { transform: scale(0); }
#         40% { transform: scale(1); }
#     }

#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#         height: 10px;
#     }

#     ::-webkit-scrollbar-track {
#         background: rgba(15, 25, 20, 0.3);
#     }

#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(180deg, #3d8b6f 0%, #7851a9 100%);
#         border-radius: 5px;
#     }

#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(180deg, #4a9d7e 0%, #8a5ec2 100%);
#     }

#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}

#     /* Spinner */
#     .stSpinner > div {
#         border-top-color: #7851a9 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# def render_search_interface():
#     """Render clean search interface"""

#     # Title section - NO BOX, just centered text
#     st.markdown("""
#         <div class="main-title">
#             <h1>Quant Intelligence</h1>
#         </div>
#         <div class="subtitle">
#             Natural language queries across financial data, SEC filings, and market intelligence
#         </div>
#     """, unsafe_allow_html=True)

#     # Search container
#     st.markdown('<div class="search-container">', unsafe_allow_html=True)

#     # Search input
#     user_question = st.text_input(
#         "Ask a question",
#         value=st.session_state.current_question,
#         placeholder="Ask about companies, SEC filings, contracts, or market data...",
#         key="search_input",
#         label_visibility="collapsed"
#     )

#     # Search button - centered under search bar
#     search_clicked = st.button("🔍 Search", use_container_width=False)

#     st.markdown('</div>', unsafe_allow_html=True)

#     return user_question, search_clicked

# def render_results(results, analysis, question, query_time):
#     """Render results with analysis"""
#     st.markdown('<div class="results-box">', unsafe_allow_html=True)

#     # Display analysis text
#     if analysis:
#         st.markdown(f"### Answer")
#         st.markdown(analysis)
#         st.markdown("---")

#     # Metrics
#     if results and len(results) > 0:
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Results", len(results))
#         with col2:
#             st.metric("Execution Time", f"{query_time:.2f}s")
#         with col3:
#             st.metric("Fields", len(results[0].keys()) if results else 0)

#         st.markdown("---")

#         # Display results as DataFrame
#         df = pd.DataFrame(results)

#         # Format numeric columns
#         for col in df.columns:
#             if df[col].dtype in ['float64', 'float32']:
#                 df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

#         st.dataframe(df, use_container_width=True, height=400)
#     else:
#         st.info("No results found. Try rephrasing your query.")

#     st.markdown('</div>', unsafe_allow_html=True)

# def main():
#     """Main application"""

#     # Sidebar
#     with st.sidebar:
#         st.markdown("### Settings")

#         if st.button("⚡ New Search"):
#             st.session_state.current_results = None
#             st.session_state.current_question = ""
#             st.session_state.current_analysis = None
#             st.rerun()

#         st.markdown("---")

#         if st.button("🗄️ Database Explorer"):
#             st.session_state.page = "database"
#             st.rerun()

#         if st.button("📊 Query History"):
#             st.session_state.page = "history"
#             st.rerun()

#         st.markdown("---")
#         st.caption("Powered by ArangoDB + LLM")

#     # Main content area
#     st.markdown('<div class="center-box">', unsafe_allow_html=True)

#     if not st.session_state.get('current_results'):
#         # Search interface
#         question, search_clicked = render_search_interface()

#         # Handle search
#         if search_clicked and question:
#             with st.spinner(""):
#                 st.markdown("""
#                     <div style='text-align: center; padding: 2rem;'>
#                         <div class='loading-dots'>
#                             <div class='loading-dot'></div>
#                             <div class='loading-dot'></div>
#                             <div class='loading-dot'></div>
#                         </div>
#                         <div style='margin-top: 1rem; color: #9aa0a6;'>Analyzing your query...</div>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 start_time = time.time()

#                 try:
#                     # Execute query
#                     plan = llm.plan_query_with_llm(question)
#                     results = llm.execute_planned_query(plan)


    

#                     # Generate analysis
#                     if results and len(results) > 0:
#                         analysis = llm.generate_analysis(question, results, plan)
#                     else:
#                         analysis = "No matching data found for your query."

#                     query_time = time.time() - start_time

#                     # Store in session
#                     st.session_state.current_results = results
#                     st.session_state.current_analysis = analysis
#                     st.session_state.current_question = question
#                     st.session_state.current_time = query_time

#                     st.rerun()

#                 except Exception as e:
#                     st.error(f"Query failed: {str(e)}")

#     else:
#         # Show results
#         render_results(
#             st.session_state.current_results,
#             st.session_state.current_analysis,
#             st.session_state.current_question,
#             st.session_state.current_time
#         )

#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()



# ###################################################################

# import streamlit as st
# import pandas as pd 
# import config as cfg 
# import database as arango_db
# import llm as llm 
# import ui as ui 
# import time
# from query_logger import get_logger
# from datetime import datetime
# import os 
# import torch 
# import base64


# import streamlit.web.server.server as server
# server.Server._max_message_size_bytes = 200 * 1024 * 1024  # 200MB for large responses

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"  Using device: {device}")

# st.set_page_config(
#     page_title="GraphRAG LLM", 
#     page_icon="src/fga-v3.png",
#     layout="wide"
# )
# # Initialize ALL session state variables
# if 'selected_collection' not in st.session_state:
#     st.session_state.selected_collection = None
# if 'show_custom_aql' not in st.session_state:
#     st.session_state.show_custom_aql = False
# if 'show_stock_overview' not in st.session_state:
#     st.session_state.show_stock_overview = False
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'saved_queries' not in st.session_state:
#     st.session_state.saved_queries = []
# if 'query_history' not in st.session_state:
#     st.session_state.query_history = []
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = ""

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     /* Reduce top padding */
#     .block-container {
#         padding-top: 2rem;
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
    
#     /* Custom header styling */
#     .header-container {
#         display: flex;
#         align-items: center;
#         gap: 20px;
#         margin-bottom: 10px;
#     }
    
#     .header-title {
#         font-size: 3.5rem;
#         font-weight: 700;
#         margin: 0;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         line-height: 1.2;
#     }
    
#     .header-subtitle {
#         font-size: 1.15rem;
#         color: #a8a8a8;
#         font-style: italic;
#         margin-top: 5px;
#     }
    
#     /* Follow-up button styling */
#     .stButton > button {
#         border-radius: 8px;
#         transition: all 0.3s;
#     }
    
#     /* Insight cards */
#     .insight-card {
#         padding: 15px;
#         border-radius: 10px;
#         background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
#         border-left: 4px solid #667eea;
#         margin: 10px 0;
#     }
    
#     /* Discovery banner */
#     .discovery-banner {
#         background: linear-gradient(90deg, #ff6b6b22 0%, #feca5722 100%);
#         border-left: 4px solid #ff6b6b;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 15px 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Convert image to base64 for inline embedding
# def get_base64_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# icon_base64 = get_base64_image("src/fga-v3.png")

# # Header with icon + title
# st.markdown(
#     f"""
#     <div class="header-container">
#         <img src="data:image/png;base64,{icon_base64}" width="70" height="70">
#         <div>
#             <h1 class="header-title">GraphRAG LLM</h1>
#         </div>
#     </div>
#     <p class="header-subtitle">
#         Ask anything. Get answers. Powered by AI & knowledge graphs.
#     </p>
#     <hr style="margin: 20px 0; border: none; border-top: 1px solid #333;">
#     """,
#     unsafe_allow_html=True
# )

# # ==================== HELPER FUNCTIONS ====================

# def generate_follow_up_questions(user_question, results, query_plan):
#     """Generate contextual follow-up questions based on results"""
    
#     intent = query_plan.get('intent', '')
#     collections = query_plan.get('collections', [])
    
#     follow_ups = []
    
#     # Pattern 1: Temporal expansion
#     if 'date' in str(results).lower() or any(c in collections for c in ['MarketData', 'EconomicData']):
#         follow_ups.append(" How has this changed over the past year?")
#         follow_ups.append(" Show me the trend for the last 5 years")
    
#     # Pattern 2: Entity expansion (if tickers found)
#     if results and 'ticker' in str(results[0]):
#         tickers = [r.get('ticker') for r in results[:3] if r.get('ticker')]
#         if tickers:
#             follow_ups.append(f"¼ Compare {', '.join(tickers[:3])} financial metrics")
#             follow_ups.append(f" What are the biggest risks for {tickers[0]}?")
    
#     # Pattern 3: Cross-collection expansion
#     if 'Award' in collections:
#         follow_ups.append(" What's the stock performance for these companies?")
#         follow_ups.append(" Do any of these companies have negative SEC sentiment?")
    
#     if 'sec_filings' in collections or 'sec_sentences' in collections:
#         follow_ups.append("° Show me government contracts for these companies")
#         follow_ups.append(" What are their financial metrics?")
    
#     if 'MarketData' in collections:
#         follow_ups.append("ðŸ›ï¸ Have these companies received government contracts?")
#         follow_ups.append("âš ï¸ What risks do they mention in SEC filings?")
    
#     # Pattern 4: Depth expansion
#     if len(results) > 5:
#         follow_ups.append("ðŸŽ¯ Show me only the top 3 results with more detail")
    
#     # Pattern 5: Comparative analysis
#     if len(results) >= 2:
#         follow_ups.append("âš–ï¸ Compare the top 3 companies side-by-side")
    
#     # Pattern 6: "Rabbit in the hat" - unexpected connections
#     follow_ups.append("ðŸŽ© Find surprising correlations in this data")
    
#     return follow_ups[:4]  # Return top 4


# def generate_insights(results, query_plan):
#     """Auto-generate 2-3 surprising insights from results"""
    
#     insights = []
    
#     if not results or len(results) == 0:
#         return insights
    
#     # Insight 1: Outliers
#     if len(results) >= 5 and 'award_amount_float' in str(results[0]):
#         amounts = [r.get('award_amount_float', 0) for r in results if r.get('award_amount_float')]
#         if amounts:
#             max_amount = max(amounts)
#             avg_amount = sum(amounts) / len(amounts)
            
#             if max_amount > avg_amount * 3:
#                 top_company = next((r for r in results if r.get('award_amount_float') == max_amount), None)
#                 if top_company:
#                     insights.append(f"¡ **Outlier Alert:** {top_company.get('ticker', 'Unknown')} received ${max_amount/1e9:.1f}B â€” **3x more** than the average!")
    
#     # Insight 2: Sentiment patterns
#     if len(results) >= 3 and any('sentiment' in str(r) or 'avg_finbert' in str(r) for r in results):
#         negative_count = sum(1 for r in results if (r.get('avg_sentiment', 0) < -0.5 or r.get('avg_finbert', 0) < -0.5))
#         if negative_count > len(results) * 0.6:
#             insights.append(f"âš ï¸ **Sentiment Warning:** {negative_count}/{len(results)} companies show **strongly negative** sentiment")
    
#     # Insight 3: Temporal patterns
#     if len(results) >= 3:
#         dates = []
#         for r in results:
#             date = r.get('date') or r.get('filing_date') or r.get('start_date')
#             if date:
#                 dates.append(date)
        
#         if dates:
#             recent_dates = [d for d in dates if d and (d.startswith('2024') or d.startswith('2025'))]
#             if len(recent_dates) > len(dates) * 0.7:
#                 insights.append(f"¥ **Recent Trend:** {len(recent_dates)}/{len(dates)} results are from the **last 2 years**")
    
#     # Insight 4: Cross-reference
#     if len(results) >= 3:
#         tickers = [r.get('ticker') for r in results if r.get('ticker')]
#         if tickers:
#             unique_tickers = len(set(tickers))
#             insights.append(f" **Market Coverage:** Found {unique_tickers} unique {'company' if unique_tickers == 1 else 'companies'} in results")
    
#     return insights


# def auto_cross_reference(results, query_plan):
#     """Automatically find related data user didn't ask for"""
    
#     discoveries = []
    
#     if not results or len(results) == 0:
#         return discoveries
    
#     collections = query_plan.get('collections', [])
    
#     # Extract tickers from results
#     tickers = list(set([r.get('ticker') for r in results[:5] if r.get('ticker')]))
    
#     # If they queried awards, suggest checking SEC sentiment
#     if 'Award' in collections and tickers:
#         discoveries.append({
#             'title': 'âš ï¸ Risk Alert Available',
#             'message': f"Want to see SEC sentiment analysis for these {len(tickers)} companies?",
#             'action': f"Show me SEC sentiment for {', '.join(tickers[:3])}"
#         })
    
#     # If they queried sentiment, suggest checking stock performance
#     if any(c in collections for c in ['sec_filings', 'sec_sentences', 'sec_sections']) and tickers:
#         discoveries.append({
#             'title': ' Market Impact Check',
#             'message': f"Curious how sentiment correlates with stock performance?",
#             'action': f"Show me stock performance for {', '.join(tickers[:3])}"
#         })
    
#     # If they queried market data, suggest checking contracts
#     if 'MarketData' in collections and tickers:
#         discoveries.append({
#             'title': 'ðŸ›ï¸ Government Contract Discovery',
#             'message': f"These companies might have government contracts worth exploring",
#             'action': f"Show me government contracts for {', '.join(tickers[:3])}"
#         })
    
#     return discoveries[:2]  # Max 2 discoveries


# def show_summary_metrics(results, query_plan):
#     """Show summary metrics for results"""
    
#     if not results or len(results) == 0:
#         return
    
#     st.markdown("###  Quick Summary")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         tickers = list(set([r.get('ticker') for r in results if r.get('ticker')]))
#         st.metric("Companies", len(tickers))
    
#     with col2:
#         if 'award_amount_float' in str(results[0]):
#             total = sum(r.get('award_amount_float', 0) for r in results)
#             st.metric("Total Value", f"${total/1e9:.1f}B" if total > 0 else "N/A")
#         elif 'close' in str(results[0]):
#             avg_price = sum(r.get('close', 0) for r in results) / len(results)
#             st.metric("Avg Price", f"${avg_price:.2f}")
#         else:
#             st.metric("Results", len(results))
    
#     with col3:
#         if 'avg_sentiment' in str(results[0]) or 'avg_finbert' in str(results[0]):
#             sentiments = [r.get('avg_sentiment', r.get('avg_finbert', 0)) for r in results]
#             avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
#             st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", 
#                      delta="Negative" if avg_sentiment < 0 else "Positive",
#                      delta_color="inverse" if avg_sentiment < 0 else "normal")
#         else:
#             st.metric("Data Points", len(results))
    
#     with col4:
#         dates = [r.get('date') or r.get('filing_date') or r.get('start_date') for r in results if r.get('date') or r.get('filing_date') or r.get('start_date')]
#         if dates:
#             dates_sorted = sorted([d for d in dates if d])
#             if dates_sorted:
#                 date_range = f"{dates_sorted[0][:4]}-{dates_sorted[-1][:4]}"
#                 st.metric("Time Range", date_range)
#         else:
#             st.metric("Status", "âœ… Complete")


# # Sidebar
# with st.sidebar:
#     st.header("âš™ï¸ Settings")
    
#     # Model toggle
#     use_local_model = st.checkbox(
#         " Use Local Fine-Tuned Model",
#         value=False,
#         help="Toggle between your fine-tuned Llama model (local) and OpenAI GPT-4"
#     )
    
#     if use_local_model:
#         st.success("âœ… Using Local Model")
#     else:
#         st.info("â˜ï¸ Using OpenAI GPT-4")
    
#     st.divider()
    
#     # Saved Queries
#     st.header("â­ Saved Queries")
    
#     if st.session_state.saved_queries:
#         for i, saved in enumerate(st.session_state.saved_queries[-5:]):
#             if st.button(f"– {saved['question'][:35]}...", key=f"saved_{i}", use_container_width=True):
#                 st.session_state.current_question = saved['question']
#                 st.rerun()
#     else:
#         st.caption("No saved queries yet")
    
#     st.divider()
    
#     # Recent Queries
#     st.header("ðŸ“œ Recent Queries")
    
#     if st.session_state.query_history:
#         for i, query in enumerate(st.session_state.query_history[-5:]):
#             with st.expander(f" {query['question'][:25]}..."):
#                 st.caption(f" Results: {query['result_count']}")
#                 st.caption(f" Time: {query['execution_time']:.1f}s")
#                 if st.button("Re-run", key=f"rerun_{i}", use_container_width=True):
#                     st.session_state.current_question = query['question']
#                     st.rerun()
#     else:
#         st.caption("No query history yet")
    
#     st.divider()
    
#     st.header(" About")
#     st.markdown("""
#     This platform queries:
#     -  Market data
#     - ðŸ›ï¸ Government contracts
#     -  Macro indicators
#     - ðŸŒ¾ Commodity positions
#     - ðŸ“„ SEC filings
#     """)
    
#     st.divider()
    
#     st.header("¡ Quick Examples")
#     examples = [
#         "Tesla closing price 2020-06-15",
#         "AAPL EBITDA March 2017",
#         "Cybersecurity risks in SEC",
#         "Top defense contracts 2024"
#     ]
    
#     for example in examples:
#         if st.button(example, key=f"example_{example[:10]}", use_container_width=True):
#             st.session_state.current_question = example
#             st.rerun()
    
#     st.divider()
    
#     if st.button(" Clear Conversation", use_container_width=True, key="clear_conv"):
#         st.session_state.conversation_history = []
#         st.session_state.query_history = []
#         st.rerun()
    
#     st.divider()
#     st.caption(f" {cfg.DB_NAME}")
#     st.caption(f" {cfg.ARANGO_URL}")

# # Create tabs
# tab1, tab2 = st.tabs([" AI Query Interface", "ðŸ—„ï¸ Database Browser"])

# # ==================== TAB 1: AI QUERY ====================
# with tab1:
#     col1, col2 = st.columns([3, 1])

#     with col1:
#         sample_questions = [
#             "What was Tesla's closing price on 2020-06-15?",
#             "What was AAPL's closing price on January 6th, 2020?",
#             "What was RTX's EBITDA value on March 9th, 2017?",
#             "During the month of April 2018 how did AAPL's stock perform?",
#             "Which tech companies have the most negative risk sentiment?",
#             "Show me cybersecurity risks mentioned in SEC filings",
#             "Show top 5 awards for companies with positive SEC sentiment"
#         ]
        
#         # Check if we have a programmatic question (from follow-up, saved query, etc.)
#         if st.session_state.current_question:
#             # Use the programmatic question directly
#             user_question = st.session_state.current_question
#             st.info(f" **Searching:** {user_question}")
#             # Don't clear it yet - wait until after search executes
#         else:
#             # Normal user input
#             user_question = st.selectbox(
#                 "Ask a question about financial data:",
#                 options=[""] + sample_questions,
#                 index=0,
#                 format_func=lambda x: "Type or select a question..." if x == "" else x,
#                 key="question_input"
#             )
            
#             if user_question == "":
#                 user_question = st.text_input(
#                     "Or type your custom question:",
#                     placeholder="Ask anything about stocks, SEC filings, or contracts...",
#                     label_visibility="collapsed",
#                     key="custom_question_input"
#                 )

#     with col2:
#         # Auto-trigger search if current_question is set
#         search_button = st.button(
#             "Ž Search", 
#             type="primary", 
#             use_container_width=True, 
#             key="search_btn",
#             disabled=not user_question
#         ) or bool(st.session_state.current_question)  # â† Auto-trigger on programmatic questions

#     # Conversation history
#     if st.session_state.conversation_history:
#         with st.expander("¬ Conversation History", expanded=False):
#             for i, msg in enumerate(st.session_state.conversation_history[-6:]):
#                 if msg["role"] == "user":
#                     st.markdown(f"**You:** {msg['content'][:150]}...")
#                 elif msg["role"] == "assistant":
#                     st.markdown(f"**Assistant:** {msg['content'][:150]}...")

# # Query Execution
# if (search_button or user_question) and user_question:
    
#     logger = get_logger()
#     start_time = time.time()
    
#     query_plan = None
#     results = None
#     llm_response = None
#     error = None
#     intent = None
    
#     try:
#         # Step 1: Quick intent check
#         with st.spinner("ðŸ§  Understanding query type..."):
#             intent = llm.quick_intent_check(user_question, use_local=use_local_model)
#             st.info(f"ðŸŽ¯ Detected: {intent.get('type', 'unknown').upper()} query")
        
#         # Step 2: Generate query
#         with st.spinner("âš™ï¸ Planning query..."):
#             query_plan = llm.plan_query_with_llm(
#                 user_question, 
#                 intent_hint=intent,
#                 use_local=use_local_model
#             )
            
#             if not query_plan:
#                 error = "Could not generate query plan"
#                 st.error(f"âŒ {error}")
#                 st.stop()
        
#         # Step 3: Show plan
#         with st.expander(" Query Plan & Strategy", expanded=False):
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.metric("Intent", query_plan.get("intent", "Unknown"))
#                 st.metric("Collections", ", ".join(query_plan.get("collections", [])))
#                 st.metric("Model", "Local (Fine-Tuned)" if use_local_model else "OpenAI GPT-4")
#             with col_b:
#                 st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
#                 st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
            
#             st.code(query_plan.get("aql_query", "No query"), language="sql")
#             if query_plan.get("bind_vars"):
#                 st.json(query_plan.get("bind_vars"))
        
#         # Step 4: Execute
#         with st.spinner("âš¡ Executing query..."):
#             results = llm.execute_planned_query(query_plan)

#         if results:
#             st.success(f" Retrieved {len(results)} results")
#         else:
#             st.warning("âš ï¸ No results found")
#             llm_response = "No results found for your query."
        
#         # Step 5: Show Summary Metrics
#         if results and len(results) > 0:
#             show_summary_metrics(results, query_plan)
        
#         # Step 6: Generate Insights
#         if results and len(results) > 2:
#             insights = generate_insights(results, query_plan)
#             if insights:
#                 st.markdown("### ðŸŽ¯ Key Insights")
#                 for insight in insights:
#                     st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
#         # Step 7: Analysis
#         if results:
#             with st.spinner(" Analyzing results..."):
#                 formatted_context = llm.format_results_for_llm(results, query_plan)
#                 analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
#                 llm_response = llm.get_llm_analysis(
#                     analysis_prompt, 
#                     use_local=use_local_model
#                 )
            
#             st.markdown("###  Analysis")
#             st.markdown(llm_response)
            
#             # Step 8: Auto Cross-Reference (Rabbit in the Hat)
#             discoveries = auto_cross_reference(results, query_plan)
#             if discoveries:
#                 st.markdown("### ðŸŽ© Related Discoveries")
#                 for disc in discoveries:
#                     st.markdown(f'<div class="discovery-banner"><strong>{disc["title"]}</strong>: {disc["message"]}</div>', unsafe_allow_html=True)
#                     if st.button(f" {disc['action']}", key=f"disc_{disc['title']}", use_container_width=True):
#                         st.session_state.current_question = disc['action']
#                         st.rerun()
            
#             # Step 9: Follow-up Questions
#             st.markdown("---")
#             st.markdown("### ¡ Explore Further")
            
#             follow_ups = generate_follow_up_questions(user_question, results, query_plan)
            
#             cols = st.columns(2)
#             for i, question in enumerate(follow_ups):
#                 with cols[i % 2]:
#                     clean_question = question
#                     for emoji in ["", "¼", "", "", "", "°", "", "ðŸ›ï¸", "âš ï¸", "ðŸŽ¯", "ðŸ“‹", "âš–ï¸", "ðŸŽ©"]:
#                         clean_question = clean_question.replace(f"{emoji} ", "")
                    
#                     if st.button(question, key=f"followup_{i}", use_container_width=True):
#                         st.session_state.current_question = clean_question
#                         st.rerun()
            
#             # Step 10: Save Query Option
#             col_save1, col_save2 = st.columns([1, 3])
#             with col_save1:
#                 if st.button("¾ Save Query", use_container_width=True):
#                     st.session_state.saved_queries.append({
#                         'question': user_question,
#                         'timestamp': datetime.now().isoformat(),
#                         'result_count': len(results)
#                     })
#                     st.success("Query saved!")
            
#             # Raw data
#             with st.expander("ðŸ“‹ View Raw Data", expanded=False):
#                 try:
#                     df = pd.DataFrame(results)
#                     cols_to_show = [col for col in df.columns if not col.startswith('_') and col != 'description_embedding']
#                     if cols_to_show:
#                         st.dataframe(df[cols_to_show], use_container_width=True)
#                     else:
#                         st.json(results[:10])
#                 except Exception as e:
#                     st.json(results[:10])
#                     st.caption(f"Could not format as table: {str(e)}")
            
#             # Debug
#             with st.expander("§ Debug: LLM Context", expanded=False):
#                 st.text(formatted_context[:3000])
#                 if len(formatted_context) > 3000:
#                     st.caption("(Truncated for display)")
        
#         else:
#             st.info("¡ Try rephrasing your question or check if the data exists in the database.")
#             st.markdown("**Suggestions:**")
#             st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
#             st.markdown("- Check date formats (YYYY-MM-DD)")
#             st.markdown("- Ensure the collection contains relevant data")
    
#     except Exception as e:
#         error = str(e)
#         st.error(f"âŒ Error: {error}")
#         llm_response = f"Query failed with error: {error}"
    
#     finally:
#         execution_time = time.time() - start_time
        
#         # Add to query history
#         st.session_state.query_history.append({
#             'question': user_question,
#             'result_count': len(results) if results else 0,
#             'execution_time': execution_time,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         # Log everything
#         query_id = logger.log_query(
#             user_question=user_question,
#             query_plan=query_plan,
#             results=results,
#             llm_response=llm_response,
#             execution_time=execution_time,
#             error=error,
#             metadata={
#                 "model": "local_finetuned" if use_local_model else "gpt-4o-mini",
#                 "intent_type": intent.get('type') if intent else None,
#                 "intent_value": intent.get('value') if intent else None,
#                 "result_count": len(results) if results else 0,
#                 "user_agent": "streamlit_app"
#             }
#         )
        
#         # Show query ID and feedback
#         st.markdown("---")
#         col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
#         with col1:
#             st.caption(f" Query ID: `{query_id}`")
        
#         with col2:
#             if st.button("ðŸ‘ Good", key=f"thumbs_up_{query_id}"):
#                 logger.log_feedback(query_id, "positive", "User clicked thumbs up")
#                 st.success("Thanks!")
        
#         with col3:
#             if st.button("ðŸ‘Ž Bad", key=f"thumbs_down_{query_id}"):
#                 logger.log_feedback(query_id, "negative", "User clicked thumbs down")
#                 st.warning("We'll review this!")
        
#         with col4:
#             st.caption(f" {execution_time:.2f}s")
# if 'logger' not in st.session_state:
#     st.session_state.logger = get_logger()

# logger = st.session_state.logger

# # Session stats in sidebar
# st.sidebar.markdown("---")
# st.sidebar.header(" Session Stats")

# try:
#     summary = logger.get_session_summary()
    
#     if "message" not in summary:
#         col1, col2 = st.sidebar.columns(2)
#         with col1:
#             st.metric("Queries", summary['total_queries'])
#             st.metric("Success", summary['successful'])
#         with col2:
#             st.metric("Failed", summary['failed'])
#             st.metric("Avg Time", summary['avg_execution_time'])
        
#         if summary['collections_used']:
#             with st.sidebar.expander("ðŸ“š Collections Used"):
#                 for coll, count in sorted(summary['collections_used'].items(), key=lambda x: x[1], reverse=True):
#                     st.caption(f"{coll}: {count}x")
        
#         if st.sidebar.button("ðŸ“¥ Download Logs"):
#             csv_file = logger.export_to_csv()
#             if csv_file:
#                 with open(csv_file, 'r') as f:
#                     st.sidebar.download_button(
#                         label="¾ Download CSV",
#                         data=f.read(),
#                         file_name=f"session_{logger.session_id}.csv",
#                         mime="text/csv"
#                     )
#     else:
#         st.sidebar.info("No queries yet")

# except Exception as e:
#     st.sidebar.warning(" Stats temporarily unavailable")

# # ==================== TAB 2: DATABASE BROWSER ====================
# with tab2:
#     ui.render_database_browser_tab()

# # Footer
# st.divider()
# st.caption("ðŸš€ Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")


# # import streamlit as st
# # import pandas as pd 
# # import config as cfg 
# # import database as arango_db
# # import llm as llm 
# # import ui as ui 

# # import streamlit as st
# # import os 
# # import torch 
# # import base64
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(f"  Using device: {device}")

# # st.set_page_config(
# #     page_title="GraphRAG LLM", 
# #     page_icon="src/fga-v3.png",
# #     layout="wide"
# # )

# # # Initialize ALL session state variables
# # if 'selected_collection' not in st.session_state:
# #     st.session_state.selected_collection = None
# # if 'show_custom_aql' not in st.session_state:
# #     st.session_state.show_custom_aql = False
# # if 'show_stock_overview' not in st.session_state:
# #     st.session_state.show_stock_overview = False
# # if 'conversation_history' not in st.session_state:
# #     st.session_state.conversation_history = []

# # # Custom CSS for better styling
# # st.markdown("""
# #     <style>
# #     /* Reduce top padding */
# #     .block-container {
# #         padding-top: 2rem;
# #     }
    
# #     /* Hide Streamlit branding */
# #     #MainMenu {visibility: hidden;}
# #     footer {visibility: hidden;}
# #     header {visibility: hidden;}
    
# #     /* Custom header styling */
# #     .header-container {
# #         display: flex;
# #         align-items: center;
# #         gap: 20px;
# #         margin-bottom: 10px;
# #     }
    
# #     .header-title {
# #         font-size: 3.5rem;
# #         font-weight: 700;
# #         margin: 0;
# #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #         -webkit-background-clip: text;
# #         -webkit-text-fill-color: transparent;
# #         line-height: 1.2;
# #     }
    
# #     .header-subtitle {
# #         font-size: 1.15rem;
# #         color: #a8a8a8;
# #         font-style: italic;
# #         margin-top: 5px;
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

# # # Convert image to base64 for inline embedding
# # def get_base64_image(path):
# #     with open(path, "rb") as f:
# #         return base64.b64encode(f.read()).decode()

# # icon_base64 = get_base64_image("src/fga-v3.png")

# # # Header with icon + title
# # st.markdown(
# #     f"""
# #     <div class="header-container">
# #         <img src="data:image/png;base64,{icon_base64}" width="70" height="70">
# #         <div>
# #             <h1 class="header-title">GraphRAG LLM</h1>
# #         </div>
# #     </div>
# #     <p class="header-subtitle">
# #         Ask anything. Get answers. Powered by AI & knowledge graphs.
# #     </p>
# #     <hr style="margin: 20px 0; border: none; border-top: 1px solid #333;">
# #     """,
# #     unsafe_allow_html=True
# # )


# # # Sidebar
# # with st.sidebar:
# #     st.header("âš™ï¸ Settings")
    
# #     # Add model toggle HERE
# #     use_local_model = st.checkbox(
# #         " Use Local Fine-Tuned Model",
# #         value=False,  # Start with OpenAI by default
# #         help="Toggle between your fine-tuned Llama model (local) and OpenAI GPT-4"
# #     )
    
# #     if use_local_model:
# #         st.success(" Using Local Model")
# #     else:
# #         st.info("â˜ï¸ Using OpenAI GPT-4")
    
# #     st.divider()
    
# #     st.header(" About")
# #     st.markdown("""
# #     This platform queries:
# #     -  Market data
# #     - ðŸ›ï¸ Government contracts
# #     -  Macro indicators
# #     - ðŸŒ¾ Commodity positions
# #     - ðŸ“„ SEC filings
# #     """)
    
# #     st.divider()
    
# #     st.header("¡ Examples")
# #     st.markdown("""
# #     - "What was Tesla's closing price on 2020-06-15?"
# #     - "What was AAPLâ€™s closing price on January 6th, 2020"
# #     - "What was RTXs EBITDA value on March 9th, 2017"
# #     - "During the month of April 2018 how did AAPLs stock perform?"
# #     """)
    
# #     st.divider()
    
# #     if st.button(" Clear Conversation", use_container_width=True, key="clear_conv"):
# #         st.session_state.conversation_history = []
# #         st.rerun()
    
# #     st.divider()
# #     st.caption(f" {cfg.DB_NAME}")
# #     st.caption(f" {cfg.ARANGO_URL}")


# # # Create tabs
# # tab1, tab2 = st.tabs([" AI Query Interface", "ðŸ—„ï¸ Database Browser"])

# # # ==================== TAB 1: AI QUERY ====================
# # with tab1:
# #     col1, col2 = st.columns([3, 1])

# #     with col1:
# #         sample_questions = [
# #             "What was Tesla's closing price on 2020-06-15?",
# #             "What was AAPL's closing price on January 6th, 2020?",
# #             "What was RTX's EBITDA value on March 9th, 2017?",
# #             "During the month of April 2018 how did AAPL's stock perform?",
# #             "Which tech companies have the most negative risk sentiment?",
# #             "Show me cybersecurity risks mentioned in SEC filings",
# #             "Compare risk sentiment between AAPL and MSFT"
# #         ]
        
# #         # Single selectbox that allows custom input
# #         user_question = st.selectbox(
# #             "Ask a question about financial data:",
# #             options=[""] + sample_questions,
# #             index=0,
# #             format_func=lambda x: "Type or select a question..." if x == "" else x,
# #             key="question_input"
# #         )
        
# #         # If empty option selected, show text input instead
# #         if user_question == "":
# #             user_question = st.text_input(
# #                 "Or type your custom question:",
# #                 placeholder="Ask anything about stocks, SEC filings, or contracts...",
# #                 label_visibility="collapsed",
# #                 key="custom_question_input"
# #             )

# #     with col2:
# #         search_button = st.button(
# #             "Ž Search", 
# #             type="primary", 
# #             use_container_width=True, 
# #             key="search_btn",
# #             disabled=not user_question  # Disable if empty
# #         )

# #     # Conversation history
# #     if st.session_state.conversation_history:
# #         with st.expander("¬ Conversation History", expanded=False):
# #             for i, msg in enumerate(st.session_state.conversation_history[-6:]):
# #                 if msg["role"] == "user":
# #                     st.markdown(f"**You:** {msg['content'][:150]}...")
# #                 elif msg["role"] == "assistant":
# #                     st.markdown(f"**Assistant:** {msg['content'][:150]}...")


# # # Query Execution
# # import time
# # from logger import get_logger

# # if (search_button or user_question) and user_question:
    
# #     # Initialize logger and tracking
# #     logger = get_logger()
# #     start_time = time.time()
    
# #     query_plan = None
# #     results = None
# #     llm_response = None
# #     error = None
# #     intent = None
    
# #     try:
# #         # Step 1: Quick intent check
# #         with st.spinner("ðŸ§  Understanding query type..."):
# #             intent = llm.quick_intent_check(user_question, use_local=use_local_model)
# #             st.info(f"ðŸŽ¯ Detected: {intent.get('type', 'unknown').upper()} query")
        
# #         # Step 2: Generate query with intent hint
# #         with st.spinner("âš™ï¸ Planning query..."):
# #             query_plan = llm.plan_query_with_llm(
# #                 user_question, 
# #                 intent_hint=intent,
# #                 use_local=use_local_model
# #             )
            
# #             if not query_plan:
# #                 error = "Could not generate query plan"
# #                 st.error(f"âŒ {error}")
# #                 st.stop()
        
# #         # Step 3: Show plan
# #         with st.expander(" Query Plan & Strategy", expanded=False):
# #             col_a, col_b = st.columns(2)
# #             with col_a:
# #                 st.metric("Intent", query_plan.get("intent", "Unknown"))
# #                 st.metric("Collections", ", ".join(query_plan.get("collections", [])))
# #                 st.metric("Model", "Local (Fine-Tuned)" if use_local_model else "OpenAI GPT-4")
# #             with col_b:
# #                 st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
# #                 st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
            
# #             st.code(query_plan.get("aql_query", "No query"), language="sql")
# #             if query_plan.get("bind_vars"):
# #                 st.json(query_plan.get("bind_vars"))
        
# #         # Step 4: Execute
# #         with st.spinner("âš¡ Executing query..."):
# #             results = llm.execute_planned_query(query_plan)

# #         if results:
# #             st.success(f" Retrieved {len(results)} results")
# #         else:
# #             st.warning("âš ï¸ No results found")
# #             llm_response = "No results found for your query."
        
# #         # Step 5: Analysis
# #         if results:
# #             with st.spinner(" Analyzing results..."):
# #                 formatted_context = llm.format_results_for_llm(results, query_plan)
# #                 analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
# #                 llm_response = llm.get_llm_analysis(
# #                     analysis_prompt, 
# #                     use_local=use_local_model
# #                 )
            
# #             st.markdown("###  Analysis")
# #             st.markdown(llm_response)

# #              # Add to your UI after showing results:
# #             st.markdown("---")
# #             st.markdown("### ¡ Explore Further")

# #             # follow_ups = llm.generate_follow_up_questions(user_question, results, query_plan)

# #             # cols = st.columns(2)
# #             # for i, question in enumerate(follow_ups):
# #             #     with cols[i % 2]:
# #             #         if st.button(question, key=f"followup_{i}", use_container_width=True):
# #             #             st.session_state.user_question = question.replace(" ", "").replace("¼ ", "").replace(" ", "").replace(" ", "").replace(" ", "").replace("° ", "").replace(" ", "").replace("ðŸ›ï¸ ", "").replace("âš ï¸ ", "").replace("ðŸŽ¯ ", "").replace("ðŸ“‹ ", "").replace("âš–ï¸ ", "").replace("ðŸŽ© ", "")
# #             #             st.rerun()
            
# #             # Raw data
# #             with st.expander("ðŸ“‹ View Raw Data", expanded=False):
# #                 try:
# #                     df = pd.DataFrame(results)
# #                     cols_to_show = [col for col in df.columns if not col.startswith('_') and col != 'description_embedding']
# #                     if cols_to_show:
# #                         st.dataframe(df[cols_to_show], use_container_width=True)
# #                     else:
# #                         st.json(results[:10])
# #                 except Exception as e:
# #                     st.json(results[:10])
# #                     st.caption(f"Could not format as table: {str(e)}")
            
# #             # Debug
# #             with st.expander("§ Debug: LLM Context", expanded=False):
# #                 st.text(formatted_context[:3000])
# #                 if len(formatted_context) > 3000:
# #                     st.caption("(Truncated for display)")
        
# #         else:
# #             st.info("¡ Try rephrasing your question or check if the data exists in the database.")
# #             st.markdown("**Suggestions:**")
# #             st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
# #             st.markdown("- Check date formats (YYYY-MM-DD)")
# #             st.markdown("- Ensure the collection contains relevant data")
    
# #     except Exception as e:
# #         error = str(e)
# #         st.error(f"âŒ Error: {error}")
# #         llm_response = f"Query failed with error: {error}"
    
# #     finally:
# #         # Calculate execution time
# #         execution_time = time.time() - start_time
        
# #         # Log everything
# #         query_id = logger.log_query(
# #             user_question=user_question,
# #             query_plan=query_plan,
# #             results=results,
# #             llm_response=llm_response,
# #             execution_time=execution_time,
# #             error=error,
# #             metadata={
# #                 "model": "local_finetuned" if use_local_model else "gpt-4o-mini",
# #                 "intent_type": intent.get('type') if intent else None,
# #                 "intent_value": intent.get('value') if intent else None,
# #                 "result_count": len(results) if results else 0,
# #                 "user_agent": "streamlit_app"
# #             }
# #         )
        
# #         # Show query ID and feedback options
# #         st.markdown("---")
# #         col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
# #         with col1:
# #             st.caption(f" Query ID: `{query_id}`")
        
# #         with col2:
# #             if st.button("ðŸ‘ Good", key=f"thumbs_up_{query_id}"):
# #                 logger.log_feedback(query_id, "positive", "User clicked thumbs up")
# #                 st.success("Thanks for the feedback!")
        
# #         with col3:
# #             if st.button("ðŸ‘Ž Bad", key=f"thumbs_down_{query_id}"):
# #                 logger.log_feedback(query_id, "negative", "User clicked thumbs down")
# #                 st.warning("Thanks! We'll review this query.")
        
# #         with col4:
# #             st.caption(f" {execution_time:.2f}s")


# # # Add session stats to sidebar
# # st.sidebar.markdown("---")
# # st.sidebar.header(" Session Stats")

# # logger = get_logger()
# # summary = logger.get_session_summary()

# # if "message" not in summary:
# #     col1, col2 = st.sidebar.columns(2)
# #     with col1:
# #         st.metric("Queries", summary['total_queries'])
# #         st.metric("Success", summary['successful'])
# #     with col2:
# #         st.metric("Failed", summary['failed'])
# #         st.metric("Avg Time", summary['avg_execution_time'])
    
# #     # Show collections used
# #     if summary['collections_used']:
# #         with st.sidebar.expander("ðŸ“š Collections Used"):
# #             for coll, count in sorted(summary['collections_used'].items(), key=lambda x: x[1], reverse=True):
# #                 st.caption(f"{coll}: {count}x")
    
# #     # Download session logs
# #     if st.sidebar.button("ðŸ“¥ Download Logs"):
# #         csv_file = logger.export_to_csv()
# #         if csv_file:
# #             with open(csv_file, 'r') as f:
# #                 st.sidebar.download_button(
# #                     label="¾ Download CSV",
# #                     data=f.read(),
# #                     file_name=f"session_{logger.session_id}.csv",
# #                     mime="text/csv"
# #                 )
# # else:
# #     st.sidebar.info("No queries yet")





        

# # # ==================== TAB 2: DATABASE BROWSER ====================
# # with tab2:
# #     ui.render_database_browser_tab()

# # # Footer
# # st.divider()
# # st.caption("ðŸš€ Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")