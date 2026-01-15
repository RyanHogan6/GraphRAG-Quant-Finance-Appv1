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
                "content": "You are a financial research assistant. Provide factual, recent information with citations. Focus on events, news, and developments that could impact financial markets."
            },
            {
                "role": "user",
                "content": search_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    result = {
        'summary': response.choices[0].message.content,
        'sources': [],
        'citations': []
    }

    # Extract citations if available (Perplexity includes them in response)
    # Note: Perplexity's citation format may vary, adjust as needed
    if hasattr(response, 'citations'):
        result['sources'] = response.citations

    print(f"[WEB SEARCH] Retrieved {len(result['summary'])} chars")

    return result

def synthesize_hybrid_response(question: str, db_results: dict, web_context: dict) -> str:
    """
    Synthesize answer combining database results and web context

    Args:
        question: Original user question
        db_results: Results from ArangoDB query
        web_context: Results from web search

    Returns:
        Comprehensive answer combining both sources
    """
    from app.llm.planning import get_openai_client

    client = get_openai_client()

    # Format DB results
    db_summary = "### Database Results:\n"
    if db_results.get('data'):
        db_summary += f"Found {len(db_results['data'])} records\n"
        db_summary += f"```json\n{db_results['data'][:3]}\n```"  # Show sample
    else:
        db_summary += "No relevant data found in database"

    # Format web context
    web_summary = "### Web Context:\n"
    web_summary += web_context.get('summary', 'No web context available')
    if web_context.get('sources'):
        web_summary += f"\n\nSources: {', '.join(web_context['sources'][:3])}"

    prompt = f"""You are analyzing a financial question that requires both historical data and current context.

User Question: "{question}"

{db_summary}

{web_summary}

Instructions:
1. Synthesize a comprehensive answer combining BOTH database insights and current web context
2. Explain HOW the current events (web) relate to the data patterns (database)
3. For Polymarket questions: explain market movements in context of real-world events
4. For stock questions: connect price/fundamental data to recent news
5. Be specific with numbers from database
6. Cite recent events from web context
7. Keep answer concise (2-3 paragraphs max)

Provide a clear, insightful answer:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    synthesized = response.choices[0].message.content

    # Append sources
    if web_context.get('sources'):
        synthesized += "\n\n**Sources:**\n"
        for source in web_context['sources'][:5]:
            synthesized += f"- {source}\n"

    return synthesized
