"""
Web search and external context retrieval using Perplexity API
"""
from openai import OpenAI
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

def get_perplexity_client():
    """Get Perplexity API client (uses OpenAI-compatible interface)"""
    if not config.PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY not configured")

    return OpenAI(
        api_key=config.PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )

def classify_query_intent(question: str) -> dict:
    """
    Classify whether question needs: DB only, Web only, or Hybrid

    Returns:
        {
            'intent': 'db_only' | 'web_only' | 'hybrid',
            'reasoning': str,
            'requires_current_events': bool
        }
    """
    from app.llm.planning import get_openai_client

    client = get_openai_client()

    prompt = f"""Classify this user question into one of three categories:

1. **db_only** - Question can be fully answered with historical database data (stock prices, financial metrics, SEC filings, government contracts, prediction market history)
   Examples:
   - "What's AAPL's PE ratio?"
   - "Show me defense contracts over $10M"
   - "What companies have golden crosses?"

2. **web_only** - Question requires external/current information not in database (general knowledge, current events, company news)
   Examples:
   - "What happened at CES 2026?"
   - "Explain quantum computing"
   - "What's the latest on AI regulation?"

3. **hybrid** - Question needs BOTH database data AND current external context (Polymarket movements, stock price changes tied to news, predictions about real-world events)
   Examples:
   - "Why is the Trump prediction market spiking?"
   - "What's driving NVDA's recent surge?"
   - "Why are defense stocks up this week?"

User Question: "{question}"

Respond in JSON format:
{{
    "intent": "db_only" | "web_only" | "hybrid",
    "reasoning": "Brief explanation",
    "requires_current_events": true/false
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    import json
    result = json.loads(response.choices[0].message.content)

    print(f"[INTENT] {result['intent'].upper()} - {result['reasoning']}")

    return result

def search_web_context(question: str, focus_areas: list = None) -> dict:
    """
    Search web for current context using Perplexity Sonar model

    Args:
        question: User's question
        focus_areas: Optional list of topics to focus search on
            e.g. ["political news", "election polls", "Trump"]

    Returns:
        {
            'summary': str,  # Main answer from Perplexity
            'sources': list,  # URLs cited
            'citations': list  # Specific facts with sources
        }
    """
    client = get_perplexity_client()

    # Build search prompt
    search_prompt = question
    if focus_areas:
        search_prompt += f"\n\nFocus on: {', '.join(focus_areas)}"

    print(f"[WEB SEARCH] Querying Perplexity: {question}")

    response = client.chat.completions.create(
        model="sonar",  # Perplexity's web-search model
        messages=[
            {
                "role": "system",
                "content": "You are a financial research assistant. Provide factual, recent information with inline citations [1], [2], etc. Focus on events, news, and developments that could impact financial markets."
            },
            {
                "role": "user",
                "content": search_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    content = response.choices[0].message.content

    # Extract citations from Perplexity response
    # Perplexity returns citations in the 'citations' field (list of URLs)
    citations = []
    if hasattr(response, 'citations') and response.citations:
        citations = response.citations
    elif hasattr(response.choices[0].message, 'citations') and response.choices[0].message.citations:
        citations = response.choices[0].message.citations

    # Try to get from response metadata if not in message
    if not citations and hasattr(response, 'usage') and hasattr(response, 'model'):
        # Check if citations are in the response object itself
        try:
            # Perplexity may include citations in different locations
            if hasattr(response, '__dict__') and 'citations' in response.__dict__:
                citations = response.__dict__['citations']
        except:
            pass

    result = {
        'summary': content,
        'sources': citations if citations else [],
        'citations': []  # Will be populated below with [number] -> URL mapping
    }

    # Create citation mapping [1] -> URL, [2] -> URL, etc.
    if citations:
        # Parse citation numbers from content like [1], [2]
        import re
        citation_numbers = re.findall(r'\[(\d+)\]', content)

        # Create list of dicts with number and URL
        for i, url in enumerate(citations, 1):
            result['citations'].append({
                'number': i,
                'url': url,
                'referenced': str(i) in citation_numbers
            })

    print(f"[WEB SEARCH] Retrieved {len(content)} chars with {len(citations)} sources")

    return result

def extract_requested_fields(question: str) -> list:
    """Extract specific fields user requested (e.g., 'show me market cap, ebitda')"""
    import re

    # Common field patterns in user questions
    field_patterns = {
        'market_cap|marketcap|market cap': 'marketCap',
        'shares outstanding|outstanding shares': 'sharesOutstanding',
        'employees|employee count': 'fullTimeEmployees',
        'ebitda': 'ebitda',
        'revenue': 'revenue',
        'pe ratio|p/e': 'trailingPE',
        'forward pe': 'forwardPE',
        'dividend': 'dividendYield',
        'eps': 'trailingEps',
        'price': 'close',
        'volume': 'volume',
        'sma|moving average': 'sma_50',
        'rsi': 'rsi_14',
    }

    requested = []
    question_lower = question.lower()

    for pattern, field in field_patterns.items():
        if re.search(pattern, question_lower):
            requested.append(field)

    return requested


def synthesize_hybrid_response(question: str, db_results: dict, web_context: dict) -> str:
    """
    Synthesize answer combining database results and web context WITH TABLES

    Args:
        question: Original user question
        db_results: Results from ArangoDB query
        web_context: Results from web search

    Returns:
        Comprehensive answer combining both sources with markdown tables
    """
    from app.llm.planning import get_openai_client
    import json

    client = get_openai_client()

    # Extract specific fields user requested
    requested_fields = extract_requested_fields(question)

    # Format DB results with full data for table generation
    db_summary = "### Database Results:\n"
    if db_results.get('data') and len(db_results['data']) > 0:
        results_sample = db_results['data'][:10]  # Top 10 for table
        db_summary += f"Found {db_results.get('count', len(db_results['data']))} records\n\n"
        db_summary += f"Sample data:\n```json\n{json.dumps(results_sample, indent=2)}\n```"
    else:
        db_summary += "No relevant data found in database"

    # Format web context
    web_summary = "### Current Events & News:\n"
    web_summary += web_context.get('summary', 'No web context available')

    # Build priority fields instruction
    field_instruction = ""
    if requested_fields:
        field_instruction = f"\n**User specifically requested these fields: {', '.join(requested_fields)}**\nMake sure to include these in your table."

    prompt = f"""You are a financial data analyst combining database results with current web context.

User Question: "{question}"

{db_summary}

{web_summary}

Instructions:
**CRITICAL: You MUST present database results in a markdown table format if data exists.**

1. **Start with a markdown table** showing the database results:
   - Include top 10 rows (or all if less than 10)
   - Choose most relevant columns (max 6-7 columns)
   - Exclude internal fields (_id, _key, _rev)
   - Format numbers with proper units ($, %, dates)
   {field_instruction}

2. **After the table**, provide analysis that:
   - Connects the database patterns to current events from web context
   - Explains HOW recent news impacts the data trends
   - For stocks: connect price/fundamentals to recent developments
   - For prediction markets: explain movements in context of real events
   - Be specific with numbers and cite web sources

3. Keep total response under 3 paragraphs after table

Markdown table format:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |

Response:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500  # Increased for table + analysis
    )

    synthesized = response.choices[0].message.content

    # Append formatted sources with numbers matching the content citations
    if web_context.get('citations'):
        synthesized += "\n\n**Sources:**\n"
        for citation in web_context['citations'][:10]:
            synthesized += f"[{citation['number']}] {citation['url']}\n"
    elif web_context.get('sources'):
        synthesized += "\n\n**Sources:**\n"
        for i, source in enumerate(web_context['sources'][:5], 1):
            synthesized += f"[{i}] {source}\n"

    return synthesized
