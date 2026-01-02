import streamlit as st
import pandas as pd 
import config as cfg 
import database as arango_db
import llm as llm 
import ui as ui 

import streamlit as st
import os 
import base64

st.set_page_config(
    page_title="Finna Go Alpha", 
    page_icon="src/fga-v2.png",
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
    </style>
""", unsafe_allow_html=True)

# Convert image to base64 for inline embedding
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

icon_base64 = get_base64_image("src/fga-v2.png")

# Header with icon + title
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/png;base64,{icon_base64}" width="70" height="70">
        <div>
            <h1 class="header-title">Finna Go Alpha</h1>
        </div>
    </div>
    <p class="header-subtitle">
        Ask anything. Get answers. Powered by AI & knowledge graphs.
    </p>
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #333;">
    """,
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Add model toggle HERE
    use_local_model = st.checkbox(
        "🤖 Use Local Fine-Tuned Model",
        value=False,  # Start with OpenAI by default
        help="Toggle between your fine-tuned Llama model (local) and OpenAI GPT-4"
    )
    
    if use_local_model:
        st.success("✅ Using Local Model")
    else:
        st.info("☁️ Using OpenAI GPT-4")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    This platform queries:
    - 📈 Market data
    - 🏛️ Government contracts
    - 📊 Macro indicators
    - 🌾 Commodity positions
    - 📄 SEC filings
    """)
    
    st.divider()
    
    st.header("💡 Examples")
    st.markdown("""
    - "What was Tesla's closing price on 2020-06-15?"
    - "What was AAPL’s closing price on January 6th, 2020"
    - "What was RTXs EBITDA value on March 9th, 2017"
    - "During the month of April 2018 how did AAPLs stock perform?"
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_conv"):
        st.session_state.conversation_history = []
        st.rerun()
    
    st.divider()
    st.caption(f"📍 {cfg.DB_NAME}")
    st.caption(f"🔌 {cfg.ARANGO_URL}")


# Create tabs
tab1, tab2 = st.tabs(["🤖 AI Query Interface", "🗄️ Database Browser"])

# ==================== TAB 1: AI QUERY ====================
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question about financial data:",
            placeholder="e.g., What was Microsoft's closing price on 2024-05-15?",
            key="question_input"
        )

    with col2:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True, key="search_btn")

    # Conversation history
    if st.session_state.conversation_history:
        with st.expander("💬 Conversation History", expanded=False):
            for i, msg in enumerate(st.session_state.conversation_history[-6:]):
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content'][:150]}...")
                elif msg["role"] == "assistant":
                    st.markdown(f"**Assistant:** {msg['content'][:150]}...")

    # Query execution
    # In your Tab 1 query execution section, after planning:

# In Tab 1, replace your entire query execution block with this:
# working below > original open api implementation 
# if (search_button or user_question) and user_question:
    
#     # Step 1: Quick intent check
#     with st.spinner("🧠 Understanding query type..."):
#         intent = llm.quick_intent_check(user_question)
#         st.info(f"🎯 Detected: {intent.get('type', 'unknown').upper()} query")
    
#     # Step 2: Generate query with intent hint
#     with st.spinner("⚙️ Planning query..."):
#         # Add intent to the planning prompt
#         query_plan = llm.plan_query_with_llm(user_question, intent_hint=intent)
        
#         if not query_plan:
#             st.error("❌ Could not generate query plan.")
#             st.stop()
    
#     # Step 3: Show plan
#     with st.expander("🔍 Query Plan & Strategy", expanded=False):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             st.metric("Intent", query_plan.get("intent", "Unknown"))
#             st.metric("Collections", ", ".join(query_plan.get("collections", [])))
#         with col_b:
#             st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
#             st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
        
#         st.code(query_plan.get("aql_query", "No query"), language="sql")
#         if query_plan.get("bind_vars"):
#             st.json(query_plan.get("bind_vars"))
    
#     # Step 4: Execute
#     with st.spinner("⚡ Executing query..."):
#         results = llm.execute_planned_query(query_plan)

#     # Rest of your existing code for displaying results...

#         if results:
#             st.success(f"✅ Retrieved {len(results)} results")
#         else:
#             st.warning("⚠️ No results found")
        
#         # Step 3: Analysis
#         if results:
#             with st.spinner("🤖 Analyzing results..."):
#                 formatted_context = llm.format_results_for_llm(results, query_plan)
#                 analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
#                 answer = llm.get_llm_analysis(analysis_prompt)

if (search_button or user_question) and user_question:
    
        # Step 1: Quick intent check
        with st.spinner("🧠 Understanding query type..."):
            intent = llm.quick_intent_check(user_question, use_local=use_local_model)  # Add flag
            st.info(f"🎯 Detected: {intent.get('type', 'unknown').upper()} query")
        
        # Step 2: Generate query with intent hint
        with st.spinner("⚙️ Planning query..."):
            query_plan = llm.plan_query_with_llm(
                user_question, 
                intent_hint=intent,
                use_local=use_local_model  # Add flag
            )
            
            if not query_plan:
                st.error("❌ Could not generate query plan.")
                st.stop()
        
        # Step 3: Show plan
        with st.expander("🔍 Query Plan & Strategy", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Intent", query_plan.get("intent", "Unknown"))
                st.metric("Collections", ", ".join(query_plan.get("collections", [])))
                st.metric("Model", "Local (Fine-Tuned)" if use_local_model else "OpenAI GPT-4")  # Show which model
            with col_b:
                st.metric("Semantic Search", "Yes" if query_plan.get("requires_embedding") else "No")
                st.caption(f"**Strategy:** {query_plan.get('explanation', 'N/A')}")
            
            st.code(query_plan.get("aql_query", "No query"), language="sql")
            if query_plan.get("bind_vars"):
                st.json(query_plan.get("bind_vars"))
        
        # Step 4: Execute (no changes needed here)
        with st.spinner("⚡ Executing query..."):
            results = llm.execute_planned_query(query_plan)

        if results:
            st.success(f"✅ Retrieved {len(results)} results")
        else:
            st.warning("⚠️ No results found")
        
        # Step 5: Analysis
        if results:
            with st.spinner("🤖 Analyzing results..."):
                formatted_context = llm.format_results_for_llm(results, query_plan)
                analysis_prompt = llm.create_analysis_prompt(user_question, formatted_context, query_plan)
                answer = llm.get_llm_analysis(
                    analysis_prompt, 
                    use_local=use_local_model  # Add flag
                )
        
            
            st.markdown("### 📊 Analysis")
            st.markdown(answer)
            
            # Raw data
            with st.expander("📋 View Raw Data", expanded=False):
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
            with st.expander("🔧 Debug: LLM Context", expanded=False):
                st.text(formatted_context[:3000])
                if len(formatted_context) > 3000:
                    st.caption("(Truncated for display)")
        
        else:
            st.info("💡 Try rephrasing your question or check if the data exists in the database.")
            st.markdown("**Suggestions:**")
            st.markdown("- Verify ticker symbols (e.g., AAPL for Apple)")
            st.markdown("- Check date formats (YYYY-MM-DD)")
            st.markdown("- Ensure the collection contains relevant data")

# ==================== TAB 2: DATABASE BROWSER ====================
with tab2:
    ui.render_database_browser_tab()

# Footer
st.divider()
st.caption("🚀 Powered by ArangoDB, OpenAI GPT-4, and text-embedding-3-small | GraphRAG Architecture")
