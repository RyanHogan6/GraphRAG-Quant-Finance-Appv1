"""
Investment Report Generator using Claude API
Transforms graph query results into actionable intelligence
"""
import anthropic
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import config


def generate_investment_report(
    query_description: str,
    query_results: List[Dict],
    metadata: Dict[str, Any],
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive investment report from graph query results

    This is the EDGE over Bloomberg terminals - explaining WHY multi-source
    graph queries create alpha vs. siloed data analysis
    """

    # Format results for Claude (truncate if too large)
    results_str = json.dumps(query_results[:20], indent=2, default=str)
    if len(results_str) > 15000:
        results_str = results_str[:15000] + "\n... (truncated)"

    # Build context about what data sources were used
    data_sources = []
    if metadata.get("has_company_data"):
        data_sources.append("Company fundamentals")
    if metadata.get("has_options"):
        data_sources.append("Options flow (insider signal detection)")
    if metadata.get("has_sec"):
        data_sources.append("SEC filings (sentiment analysis)")
    if metadata.get("has_awards"):
        data_sources.append("Government contracts")
    if metadata.get("has_commodities"):
        data_sources.append("Commodity futures & energy data")
    if metadata.get("has_markets"):
        data_sources.append("Prediction markets")

    prompt = f"""You are an expert financial analyst generating an investment research report from a graph database query.

**Query Description:** {query_description}

**Data Sources Integrated:** {', '.join(data_sources)}

**Query Results:** ({metadata.get('result_count', 0)} records)
```json
{results_str}
```

**Date Range:** {metadata.get('date_range', {}).get('earliest', 'N/A')} to {metadata.get('date_range', {}).get('latest', 'N/A')}

Generate a comprehensive investment report in the following JSON structure:

{{
  "executive_summary": "2-3 sentence summary of key findings",
  "investment_thesis": "Why this analysis matters - what edge does it provide?",
  "key_findings": [
    {{
      "finding": "Specific observation from data",
      "significance": "Why this matters for investors",
      "data_point": "Supporting metric/value"
    }}
  ],
  "graph_advantage": {{
    "title": "Why Graph Queries Beat Traditional Analysis",
    "traditional_approach": "What a Bloomberg terminal user would do (siloed)",
    "graph_approach": "How our multi-source query creates alpha",
    "time_advantage": "How much faster/earlier we spotted this",
    "examples": [
      "Specific example of connection that traditional analysis would miss"
    ]
  }},
  "risk_factors": [
    "Potential risk or caveat to consider"
  ],
  "recommended_actions": [
    "Specific actionable next step"
  ],
  "data_quality_notes": "Any gaps or limitations in the data"
}}

**CRITICAL INSTRUCTIONS:**
1. Focus on CONNECTIONS between data sources - that's the graph advantage
2. Explain WHY multi-source analysis matters (e.g., "Options activity 3 days before contract award suggests insider knowledge")
3. Be specific with numbers and dates from the results
4. If this is company-specific ({ticker if ticker else 'N/A'}), mention the ticker prominently
5. Highlight temporal relationships (before/after patterns)
6. Call out unusual patterns or anomalies
7. Educational tone - explain concepts for mid-level analysts
8. If limited data, acknowledge it honestly

Return ONLY the JSON object, no other text."""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.3,  # Lower temperature for factual analysis
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse Claude's response
        content = response.content[0].text

        # Extract JSON (Claude might wrap it in markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        report = json.loads(content)

        # Add metadata
        report["generated_at"] = datetime.now().isoformat()
        report["query_description"] = query_description
        report["data_sources"] = data_sources
        report["result_count"] = metadata.get("result_count", 0)

        return report

    except json.JSONDecodeError as e:
        print(f"[REPORT] Failed to parse Claude response as JSON: {e}")
        print(f"[REPORT] Raw response: {content[:500]}")

        # Fallback: Return basic report
        return create_fallback_report(query_description, query_results, metadata)

    except Exception as e:
        print(f"[REPORT] Error generating report: {e}")
        return create_fallback_report(query_description, query_results, metadata)


def create_fallback_report(
    query_description: str,
    query_results: List[Dict],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Create basic report when Claude generation fails"""

    key_findings = []

    # Try to extract some basic insights
    if query_results:
        first_result = query_results[0]

        # Look for interesting fields
        for key, value in first_result.items():
            if any(keyword in key.lower() for keyword in ['price', 'volume', 'sentiment', 'amount']):
                key_findings.append({
                    "finding": f"{key}: {value}",
                    "significance": "Key metric from query results",
                    "data_point": str(value)
                })

    return {
        "executive_summary": f"Query returned {metadata.get('result_count', 0)} results from {', '.join(metadata.get('collections', []))}.",
        "investment_thesis": "Analysis of multi-source graph data.",
        "key_findings": key_findings[:5],
        "graph_advantage": {
            "title": "Multi-Source Analysis",
            "traditional_approach": "Traditional analysis requires manual correlation across multiple platforms",
            "graph_approach": "Graph database automatically links related entities and events",
            "time_advantage": "Instant connection discovery vs. hours of manual research",
            "examples": ["Automated detection of cross-source relationships"]
        },
        "risk_factors": ["Limited automated interpretation - manual review recommended"],
        "recommended_actions": ["Review raw data for additional insights"],
        "data_quality_notes": "Automated report generation - verify key findings",
        "generated_at": datetime.now().isoformat(),
        "query_description": query_description,
        "data_sources": metadata.get("collections", []),
        "result_count": metadata.get("result_count", 0)
    }
