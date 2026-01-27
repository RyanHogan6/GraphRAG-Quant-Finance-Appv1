"""
Context-Aware Response Synthesis Engine
Analyzes query results with domain-specific intelligence instead of lazy data dumps
"""
import json
from typing import Dict, List, Any, Optional


# Domain-specific analysis strategies
ANALYSIS_STRATEGIES = {
    "options_screening": {
        "focus": "Unusual activity patterns and potential insider signals",
        "key_metrics": ["unusual_ratio", "put_call_ratio", "iv_rank", "volume"],
        "novelty_checks": [
            "Is unusual_ratio > 3? (Strong signal)",
            "Is put_call_ratio < 0.5? (Bullish) or > 2? (Bearish)",
            "Is IV elevated vs historical? (Anticipation of volatility)",
            "Is this before earnings/events? (Potential insider knowledge)"
        ],
        "synthesis": """
        Connect patterns:
        - Multiple stocks in same sector with unusual calls = sector rotation
        - High volume + low P/C + high IV = bullish positioning
        - Call sweeps near 52-week highs = breakout anticipation
        """
    },

    "insider_trading_detection": {
        "focus": "Correlation between options activity and corporate events",
        "key_metrics": ["days_before_event", "unusual_ratio", "call_premium", "award_amount"],
        "novelty_checks": [
            "Are there multiple instances of activity 30-60 days before awards?",
            "Is the pattern consistent across defense contractors?",
            "Does high premium suggest informed buying?",
            "Are there similar patterns before SEC filings?"
        ],
        "synthesis": """
        Pattern detection:
        - Consistent timing (30-60 days before) = potential insider knowledge
        - Large premiums paid = strong conviction
        - Multiple occurrences = systematic pattern, not coincidence
        - Correlation with event magnitude (larger awards = larger options bets)
        """
    },

    "commodity_price_analysis": {
        "focus": "Price trends, technical signals, and fundamental drivers",
        "key_metrics": ["price_change", "rsi", "volume", "volatility"],
        "novelty_checks": [
            "Is RSI > 70 (overbought) or < 30 (oversold)?",
            "Is price at 52-week high/low?",
            "Is volume significantly above average?",
            "Are there divergences (price up but volume down)?"
        ],
        "synthesis": """
        Market dynamics:
        - Technical signals: RSI extremes, breakouts, support/resistance
        - Volume analysis: Conviction vs weak moves
        - Cross-commodity: Gold up + crude down = flight to safety
        - Seasonality: Natural gas storage cycles, agricultural harvests
        """
    },

    "commodity_fundamental_analysis": {
        "focus": "Inventory/storage impacts on prices",
        "key_metrics": ["inventory_change", "vs_5yr_avg", "price_response"],
        "novelty_checks": [
            "Is storage significantly below 5-year average? (Bullish for prices)",
            "Was there a large build/draw? (>10% weekly change)",
            "Did prices respond as expected? (Build = bearish, draw = bullish)",
            "Are there seasonal anomalies?"
        ],
        "synthesis": """
        Supply/demand dynamics:
        - Low storage + winter weather = price spike risk (natural gas)
        - Large crude build + high production = bearish for oil
        - Inventory vs price divergence = market inefficiency
        - Compare to refinery utilization, Cushing levels
        """
    },

    "award_analysis": {
        "focus": "Contract patterns, agency spending, competitive dynamics",
        "key_metrics": ["award_amount", "awarding_agency", "recipient", "contract_type"],
        "novelty_checks": [
            "Is this an unusually large contract? (>$1B)",
            "Is there a concentration among few recipients?",
            "Are awards trending up/down over time?",
            "Are there new entrants or market share shifts?"
        ],
        "synthesis": """
        Strategic insights:
        - Defense spending trends: Concentration vs diversification
        - Incumbent dominance: Same recipients getting majority
        - Emerging areas: AI, cybersecurity, space contracts growing
        - Timing: Q4 "use it or lose it" spending spikes
        """
    },

    "multi_source_correlation": {
        "focus": "Connections across data sources (options + awards, futures + CFTC)",
        "key_metrics": ["timing", "magnitude", "correlation_strength"],
        "novelty_checks": [
            "Do events correlate across time? (Options before awards)",
            "Is there magnitude correlation? (Bigger awards = bigger options bets)",
            "Are there leading indicators? (CFTC positions predict futures prices)",
            "Are correlations stronger than random?"
        ],
        "synthesis": """
        Cross-signal analysis:
        - Options + Awards: Insider trading detection
        - Futures + CFTC: Speculator positioning predicts moves
        - Stock + Prediction Markets: Sentiment vs reality divergence
        - Options + SEC Filings: Pre-announcement positioning
        """
    },

    "company_overview": {
        "focus": "Comprehensive company intelligence synthesis",
        "key_metrics": ["price_trend", "fundamentals", "recent_filings", "options_activity", "government_contracts"],
        "novelty_checks": [
            "Is there unusual price movement? (>5% single day, >20% monthly)",
            "Are there recent SEC filings with significant sentiment shifts?",
            "Is there unusual options activity? (Volume > 3x average)",
            "Are there major government contracts? (>$100M)",
            "Are fundamentals improving or deteriorating?"
        ],
        "synthesis": """
        Holistic company analysis:
        - Price action: Trends, breakouts, support/resistance levels
        - Fundamental health: P/E, analyst targets, earnings trajectory
        - Corporate signals: SEC filings sentiment, management tone
        - Smart money: Options flow, insider activity patterns
        - Revenue catalysts: Government contracts, new business wins

        Synthesize into 4-sentence AI intelligence summary - NO TABLE NEEDED!
        """
    }
}


def detect_analysis_type(user_question: str, query_plan: dict, results: List[Dict] = None) -> str:
    """Detect which analysis strategy to use based on question and query"""

    question_lower = user_question.lower()
    collections = query_plan.get('collections', [])
    intent = query_plan.get('intent', '')

    # Check for company overview - comprehensive workup structure
    # This structure has nested MarketData, sec_filings, Award, options_flow
    if results and len(results) > 0:
        first_result = results[0]
        has_nested_market_data = isinstance(first_result.get('MarketData'), list)
        has_nested_filings = isinstance(first_result.get('sec_filings'), list)
        has_company_ticker = 'ticker' in first_result and 'company' in first_result

        if has_nested_market_data or (has_nested_filings and has_company_ticker):
            return "company_overview"

    # Also detect company overview by question pattern
    company_patterns = ['show me', 'tell me about', 'overview', 'analysis of', 'info on', 'information about']
    if any(pattern in question_lower for pattern in company_patterns):
        # Check if question is about a single ticker (2-5 uppercase letters)
        import re
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', question_lower.upper())
        if ticker_match and len(question_lower.split()) <= 8:
            return "company_overview"

    # Check for multi-source queries
    if len(collections) >= 3:
        return "multi_source_correlation"

    # Options-specific
    if 'options_flow' in collections or any(keyword in question_lower for keyword in ['unusual', 'call', 'put', 'sweep', 'options']):
        if any(keyword in question_lower for keyword in ['before', 'award', 'filing', 'insider']):
            return "insider_trading_detection"
        return "options_screening"

    # Commodity-specific
    if 'futures_prices' in collections:
        if any(coll in collections for coll in ['eia_crude_inventory', 'eia_natgas_storage', 'commodity_positions']):
            return "commodity_fundamental_analysis"
        return "commodity_price_analysis"

    # EIA-specific
    if any(coll.startswith('eia_') for coll in collections):
        return "commodity_fundamental_analysis"

    # Awards
    if 'Award' in collections:
        return "award_analysis"

    # Default
    return "generic"


def check_data_novelty(results: List[Dict], strategy: Dict) -> Dict[str, Any]:
    """Check if the data contains anything novel or significant"""

    if not results:
        return {"is_novel": False, "reason": "No data found"}

    novelty_findings = []

    # Run strategy-specific novelty checks
    for check in strategy.get("novelty_checks", []):
        # This is a simplified version - in production, implement actual checks
        novelty_findings.append({
            "check": check,
            "finding": "Evaluating..."  # Placeholder
        })

    # Statistical checks (generic)
    if len(results) < 5:
        novelty_findings.append({
            "check": "Data sparsity",
            "finding": f"Only {len(results)} results - limited statistical significance"
        })

    return {
        "is_novel": len(novelty_findings) > 0,
        "findings": novelty_findings
    }


def generate_context_aware_prompt(
    user_question: str,
    results: List[Dict],
    query_plan: dict,
    analysis_type: str
) -> str:
    """Generate analysis prompt with domain-specific intelligence"""

    strategy = ANALYSIS_STRATEGIES.get(analysis_type, {})

    # Note: results are already trimmed by trim_results_for_llm() in planning.py
    # No need to trim again here
    result_count = len(results)

    # Check for novelty
    novelty_analysis = check_data_novelty(results, strategy)

    # Build context-aware prompt
    prompt = f"""You are a specialized financial intelligence analyst.

**User Question:** "{user_question}"

**Analysis Type:** {analysis_type.replace('_', ' ').title()}"""

    # Add NO TABLE warning at the top for company overview
    if analysis_type == "company_overview":
        prompt += """

═══════════════════════════════════════════════════════════════════════════════
🚨 MANDATORY INSTRUCTION - FAILURE TO COMPLY WILL RESULT IN INCORRECT OUTPUT 🚨
═══════════════════════════════════════════════════════════════════════════════

THIS IS A COMPANY OVERVIEW QUERY.

YOU MUST NOT CREATE ANY TABLES WHATSOEVER.

SPECIFICALLY PROHIBITED:
❌ NO markdown tables with | pipes
❌ NO tables with Date/Close Price/Volume columns
❌ NO formatted data grids
❌ NO "Apple Stock Data from Database" sections
❌ NO tabular layouts of ANY kind

WHY: The frontend CompanyWorkup component already displays:
- Interactive price charts with full history
- Fundamental metrics in a structured layout
- All company data in visual format

YOUR JOB: Write ONLY natural language analysis (paragraphs + bullet points)

═══════════════════════════════════════════════════════════════════════════════
"""

    prompt += f"""

**Data Retrieved:** {result_count} results from {', '.join(query_plan.get('collections', []))}

**Data Retrieved:** {result_count} results from {', '.join(query_plan.get('collections', []))}

**Results Sample:**
{json.dumps(results, indent=2)}

**Analysis Focus:** {strategy.get('focus', 'General data analysis')}

**Key Metrics to Highlight:**
{chr(10).join(f'- {metric}' for metric in strategy.get('key_metrics', []))}

**Novelty Checks:**
{chr(10).join(f'- {check}' for check in strategy.get('novelty_checks', []))}

**Domain-Specific Synthesis Strategy:**
{strategy.get('synthesis', 'Provide general insights and patterns')}

**CRITICAL INSTRUCTIONS:**

1. **Start with Executive Summary (2-3 sentences)**
   - What's the key finding? Is it significant?
   - If data shows nothing unusual, say so clearly
"""

    # Add table instruction ONLY for non-company-overview queries
    if analysis_type != "company_overview":
        prompt += """
2. **Present Data in Markdown Table**
   - Top 10 rows (or all if less)
   - 5-7 most relevant columns only
   - Format numbers properly ($, %, dates)

3. **Provide Domain-Specific Analysis (not generic insights!)**
"""
    else:
        prompt += """
2. **Provide Domain-Specific Intelligence Synthesis**
   🚨 CRITICAL: DO NOT CREATE ANY TABLES OR FORMATTED DATA! 🚨
   🚨 NO MARKDOWN TABLES (|---|---) ALLOWED! 🚨
   🚨 The frontend CompanyWorkup component already displays ALL company data with interactive charts! 🚨

   Your ONLY job is to write a natural language paragraph (4-5 sentences) that synthesizes the key insights:
"""

    prompt += """   - Apply the novelty checks listed above
   - Explain WHY patterns matter, not just WHAT they are
   - Connect to broader market context
   - Identify anomalies, outliers, or significant patterns
"""

    if analysis_type != "company_overview":
        prompt += """
4. **Synthesis (if applicable)**
   - If multiple data sources: connect the dots
   - If time series: identify trends, breakouts, reversals
   - If screening: explain what makes these results stand out

5. **Avoid Generic Fluff**
   - DON'T just describe the table
   - DON'T provide obvious observations
   - DO explain significance and implications
   - DO highlight anything unusual or actionable
"""
    else:
        prompt += """
3. **Key Insights (3-4 bullet points)**
   - Price action and technical signals
   - Fundamental strength/weakness
   - Corporate developments (filings, awards, options activity)
   - Smart money signals (unusual options, insider patterns)
"""

    prompt += """
**Example of Good vs Bad Analysis:**

❌ BAD: "The table shows 5 stocks with high call volume. NVDA has 10,000 calls."
✅ GOOD: "NVDA shows 3x normal call volume with P/C ratio of 0.3, suggesting strong bullish positioning. This is significant because it coincides with earnings in 2 weeks - similar patterns preceded the last 3 earnings beats."

**Response Format:**"""

    if analysis_type == "company_overview":
        prompt += """

🚨 COMPANY OVERVIEW FORMAT (NO TABLES!) 🚨

❌ DO NOT DO THIS:
| Date | Close Price | Volume |
|------|-------------|--------|
| 2025-12-31 | $26.06 | 1.2M |

✅ DO THIS INSTEAD:
## AI Intelligence Summary

[Write a flowing paragraph synthesizing the data - NO TABLES!]

Apple has declined approximately 8.5% over the past month, closing at $248.04 as of January 23, 2026, down from its end-of-year price of $271.86. The stock shows structural momentum with technical indicators suggesting... Recent SEC filings indicate bearish sentiment with BIAS scores of -0.244 and -0.274... Options activity remains within normal ranges with no unusual volume spikes... The company maintains strong fundamentals with 34.78% revenue growth and 22.96% profit margins.

## Key Insights
- [Technical/price signal]
- [Fundamental strength/weakness]
- [Corporate development]
- [Smart money signal]
"""
    else:
        prompt += """
## [Executive Summary]

[Table]

## Analysis

[Domain-specific insights]

## Key Takeaways
- [Actionable insight 1]
- [Actionable insight 2]
"""

    return prompt


def analyze_results_with_context(
    user_question: str,
    results: List[Dict],
    query_plan: dict
) -> str:
    """
    Main entry point for context-aware result analysis.
    Replaces generic analyze_results_with_llm function.
    """

    # Detect analysis type (pass results to detect company_overview structure)
    analysis_type = detect_analysis_type(user_question, query_plan, results)

    print(f"\n[SYNTHESIS] Detected analysis type: {analysis_type}")
    print(f"[SYNTHESIS] Result count: {len(results)}")

    # Generate context-aware prompt
    prompt = generate_context_aware_prompt(
        user_question,
        results,
        query_plan,
        analysis_type
    )

    print(f"[SYNTHESIS] Generated context-aware prompt ({len(prompt)} chars)")

    # Return prompt (caller will send to LLM)
    return prompt


# Backward compatibility wrapper
def get_enhanced_analysis_prompt(user_question: str, results: List[Dict], query_plan: dict) -> str:
    """
    Drop-in replacement for the current analysis_prompt generation.
    Can be called from existing analyze_results_with_llm function.
    """
    return analyze_results_with_context(user_question, results, query_plan)
