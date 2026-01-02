"""LLM query planning, analysis, and validation"""
import openai
import json
from config import *
from prompts import *
import streamlit as st
import re
import openai
from datetime import datetime
import json
import database as arango_db


def get_query_embedding(text):
    """Generate embedding vector for semantic search"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation error: {str(e)}")
        return None

def plan_query_with_llm(question, intent_hint=None, use_local=False):
    """Generate AQL query plan from natural language"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Add intent hint to prompt
    hint_text = ""
    if intent_hint:
        if intent_hint.get("type") == "ticker":
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."
    
    planning_prompt = f"""You are a database query planner for ArangoDB.

{SCHEMA_DESCRIPTION}

{FEW_SHOT_EXAMPLES}

USER QUESTION: "{question}"{hint_text}
Current Date: {current_date}

Generate a JSON response with:
- "intent": classification
- "collections": array
- "requires_embedding": boolean (true for concept queries)
- "embedding_text": text (if semantic)
- "aql_query": AQL query
- "bind_vars": object
- "explanation": strategy explanation

Return ONLY valid JSON.

Response:"""

    # Rest of your existing function...
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=1200,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Query planning error: {str(e)}")
        return None


def quick_intent_check(question, use_local=False):
    """Quick LLM call to determine if ticker or semantic query"""
    
    check_prompt = f"""Question: "{question}"

Is this asking about a TICKER SYMBOL or a CONCEPT?

TICKER: Question mentions a specific stock ticker (2-5 uppercase letters like AAPL, CMI, TSLA)
CONCEPT: Question asks about a topic/theme (AI, cybersecurity, renewable energy, etc.)

Examples:
- "CMI awards" → TICKER (CMI is Cummins stock ticker)
- "awards related to AI" → CONCEPT (AI = artificial intelligence topic)
- "TSLA in 2024" → TICKER
- "renewable energy contracts" → CONCEPT

Return JSON: {{"type": "ticker", "value": "CMI"}} or {{"type": "concept", "value": "artificial intelligence"}}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "unknown"}


def validate_aql_syntax(aql_query):
    """Basic syntax validation before execution"""
    errors = []
    
    # Check for common typos
    if "compan." in aql_query and "company" in aql_query:
        errors.append("Typo detected: 'compan.' should be 'company.'")
        aql_query = aql_query.replace("compan.", "company.")
    
    # Check for undeclared variables
    # Find all variable declarations (FOR var IN ...)
    declared_vars = set(re.findall(r'FOR\s+(\w+)\s+IN', aql_query, re.IGNORECASE))
    
    # Find all variable uses (var.field)
    used_vars = set(re.findall(r'(\w+)\.', aql_query))
    
    # Check if any used vars aren't declared
    undeclared = used_vars - declared_vars - {'doc', 'data'}  # doc and data are common
    if undeclared:
        errors.append(f"Undeclared variables: {undeclared}")
    
    # Check bind variables
    bind_params = set(re.findall(r'@(\w+)', aql_query))
    
    return aql_query, errors, bind_params

# Use in execute_planned_query:
def execute_planned_query(plan):
    if not plan or 'aql_query' not in plan:
        return []
    
    db = arango_db.get_arango_connection()
    if not db:
        return []
    
    try:
        aql_query = plan.get("aql_query", "")
        bind_vars = plan.get("bind_vars", {})
        
        # Validate and fix syntax
        aql_query, syntax_errors, required_bind_vars = validate_aql_syntax(aql_query)
        
        if syntax_errors:
            st.warning(f"⚠️ Query issues detected and auto-fixed:")
            for error in syntax_errors:
                st.caption(f"  - {error}")
        
        # Update the plan with fixed query
        plan["aql_query"] = aql_query
        
        # Check if all required bind variables are provided
        missing_vars = required_bind_vars - set(bind_vars.keys())
        
        # Handle embeddings
        if "@query_vector" in required_bind_vars:
            if plan.get("requires_embedding") and plan.get("embedding_text"):
                embedding = get_query_embedding(plan["embedding_text"])
                if embedding:
                    bind_vars["query_vector"] = embedding
                    missing_vars.discard("query_vector")
                else:
                    st.error("Failed to generate embedding for semantic search")
                    return []
            else:
                st.error("Query requires @query_vector but no embedding_text provided")
                return []
        
        # Remove requires_embedding from bind_vars if it was added (shouldn't happen)
        if "requires_embedding" in bind_vars:
            del bind_vars["requires_embedding"]
        
        if missing_vars:
            st.error(f"❌ Missing bind variables: {missing_vars}")
            with st.expander("🐛 Debug"):
                st.write(f"Required: {required_bind_vars}")
                st.write(f"Provided: {set(bind_vars.keys())}")
            return []
        
        # Execute
        cursor = db.aql.execute(
            aql_query, 
            bind_vars=bind_vars,
            ttl=30,
            batch_size=1000,
            optimizer_rules=["+all"]
        )
        
        results = list(cursor)
        return results
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"Query execution error: {error_msg}")
        
        with st.expander("🐛 Debug Query"):
            st.code(plan.get("aql_query", ""), language="sql")
            st.json(plan.get("bind_vars", {}))
        
        return []




def format_results_for_llm(results, query_plan=None):
    """
    Format query results for LLM consumption
    Handles both dictionary objects and scalar values (counts, aggregates, etc.)
    """
    
    # Handle empty results
    if not results:
        return "No results found."
    
    # Handle None
    if results is None:
        return "Query returned no data."
    
    formatted = []
    
    for doc in results:
        # Case 1: Dictionary (normal document)
        if isinstance(doc, dict):
            # Remove internal fields (_id, _key, _rev)
            clean_doc = {k: v for k, v in doc.items() 
                        if not k.startswith('_')}
            formatted.append(clean_doc)
        
        # Case 2: List or tuple
        elif isinstance(doc, (list, tuple)):
            formatted.append(list(doc))
        
        # Case 3: Simple value (int, float, str, bool, None)
        else:
            formatted.append(doc)
    
    # If single scalar value, format nicely
    if len(formatted) == 1:
        value = formatted[0]
        
        # Single number
        if isinstance(value, (int, float)):
            return f"Result: {value:,}" if isinstance(value, int) else f"Result: {value:.2f}"
        
        # Single dict
        elif isinstance(value, dict):
            return formatted
        
        # Single string/other
        else:
            return f"Result: {value}"
    
    # Multiple results
    return formatted



def create_analysis_prompt(question, formatted_context, plan):
    """
    Create prompt for final LLM analysis with domain expertise
    """
    prompt = f"""You are a quantitative financial analyst providing insights from a multi-source graph database containing market data, government contracts, macroeconomic indicators, and commodity positions.

DATABASE QUERY EXECUTED:
Intent: {plan.get('intent', 'Unknown')}
Strategy: {plan.get('explanation', 'Data retrieved from graph database')}

RETRIEVED DATA:
{formatted_context}

USER QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. Answer the question directly and concisely using ONLY the provided data
2. For financial data: highlight trends, anomalies, patterns, or notable values
3. For government awards: focus on amounts, agencies, recipients, timing
4. For macroeconomic data: provide context and interpretation
5. For commodity data: explain positions and market indicators
6. If multiple results exist, format as a Markdown table with relevant columns
7. Cite specific data points using [1], [2], etc. corresponding to result numbers
8. Provide quantitative summary when applicable (totals, averages, ranges, changes)
9. If data is incomplete or missing, explicitly state what's absent
10. For semantic searches: explain why results are relevant
11. Keep response concise and data-focused (avoid unnecessary preamble)

FORMAT GUIDELINES:
- Single values: Direct answer with citation
- Multiple items: Markdown table with key columns
- Time-series: Show trends and notable changes
- Comparisons: Highlight differences and similarities

ANSWER:"""
    
    return prompt


# def get_llm_analysis(prompt):
#     """Get analysis from OpenAI"""
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=1500,
#             temperature=0.2,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"OpenAI API error: {str(e)}"

#chema context for the model
def get_llm_analysis(prompt, use_local=False):
    """Get analysis from model with explicit logging"""
    
    print(f"🔍 get_llm_analysis called with use_local={use_local}")  # DEBUG
    
    use_local = False
    if use_local:
        print("🟢 Attempting local model...")  # DEBUG
        try:
            from local_llm import get_local_llm
            
            llm = get_local_llm()
            print("✅ Local model loaded")  # DEBUG
            
            result = llm.generate(prompt, max_tokens=512, temperature=0.1)
            
            # Extract response
            if "assistant<|end_header_id|>" in result:
                response = result.split("assistant<|end_header_id|>")[-1].strip()
            else:
                response = result
            
            response = response.replace("<|eot_id|>", "").strip()
            
            print(f"✅ Local model generated {len(response)} chars")  # DEBUG
            return response
            
        except Exception as e:
            print(f"❌ Local model failed: {e}")  # DEBUG
            import traceback
            traceback.print_exc()
            # DON'T fall back - raise error to see what's wrong
            raise Exception(f"Local model failed: {e}")
    
    # OpenAI path
    print("🔵 Using OpenAI...")  # DEBUG
    import openai
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content
