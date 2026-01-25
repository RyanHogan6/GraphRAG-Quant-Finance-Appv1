"""
API cost estimation for OpenAI and Perplexity
Based on current pricing as of January 2026
"""

# OpenAI Pricing (per 1M tokens)
OPENAI_PRICING = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0
    },
    "text-embedding-3-large": {
        "input": 0.13,
        "output": 0.0
    }
}

# Perplexity Pricing (per 1M tokens)
PERPLEXITY_PRICING = {
    "sonar": {
        "input": 1.00,
        "output": 1.00
    },
    "sonar-pro": {
        "input": 3.00,
        "output": 15.00
    }
}


def estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate OpenAI API cost

    Args:
        model: Model name (e.g., "gpt-4o-mini")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in dollars
    """
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def estimate_perplexity_cost(input_tokens: int, output_tokens: int, is_pro: bool = False) -> float:
    """
    Estimate Perplexity API cost

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        is_pro: Whether using sonar-pro

    Returns:
        Cost in dollars
    """
    model = "sonar-pro" if is_pro else "sonar"
    pricing = PERPLEXITY_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def estimate_query_cost_simple(
    question_length: int,
    result_count: int,
    has_web_search: bool = True,
    model: str = "gpt-4o-mini"
) -> float:
    """
    Simplified cost estimation when token counts aren't available

    Assumptions:
    - Planning: ~2000 input tokens + ~500 output tokens
    - Analysis: ~1500 input tokens + ~800 output tokens
    - Web search: ~500 input tokens + ~1000 output tokens
    - Embedding: ~100 tokens (if semantic search)

    Args:
        question_length: Length of user question (characters)
        result_count: Number of results returned
        has_web_search: Whether web search was used
        model: OpenAI model used

    Returns:
        Estimated cost in dollars
    """
    total_cost = 0.0

    # Query planning cost (always runs)
    # Input: schema + prompt + question + examples ≈ 2000 tokens
    # Output: AQL query + explanation ≈ 500 tokens
    total_cost += estimate_openai_cost(model, 2000, 500)

    # Results analysis cost (if results returned)
    if result_count > 0:
        # Input: question + results (estimate 10 tokens per char, max 1000)
        input_tokens = min(question_length * 10 + result_count * 50, 1500)
        # Output: analysis ≈ 800 tokens
        total_cost += estimate_openai_cost(model, input_tokens, 800)

    # Web search cost (if enabled)
    if has_web_search:
        total_cost += estimate_perplexity_cost(500, 1000, is_pro=False)

    return round(total_cost, 4)
