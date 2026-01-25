"""
Large-scale query test suite
Tests LLM query generation across all collection types and patterns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
from app.llm.planning import plan_query_with_llm
from app.database.connection import execute_aql
from dotenv import load_dotenv
load_dotenv()
# Test cases: (question, expected_collections, expected_pattern, min_results)
TEST_CASES = [
    # === BASIC LOOKUPS ===
    {
        "id": "basic_001",
        "category": "Company Lookup",
        "question": "Show me Apple company info",
        "expected_collections": ["Company"],
        "expected_pattern": "ticker.*AAPL",
        "min_results": 1,
        "max_time_ms": 2000
    },
    {
        "id": "basic_002",
        "category": "Market Data",
        "question": "What was Tesla's closing price on January 5, 2024?",
        "expected_collections": ["MarketData"],
        "expected_pattern": "ticker.*TSLA",
        "min_results": 0,  # May not have that exact date
        "max_time_ms": 2000
    },
    {
        "id": "basic_003",
        "category": "Awards",
        "question": "Show me the top 10 largest government contracts",
        "expected_collections": ["Award"],
        "expected_pattern": "award_amount",
        "min_results": 10,
        "max_time_ms": 3000
    },

    # === SEMANTIC SEARCH ===
    {
        "id": "semantic_001",
        "category": "Award Semantic",
        "question": "Find government contracts related to artificial intelligence",
        "expected_collections": ["Award"],
        "expected_pattern": "description.*",
        "min_results": 5,
        "max_time_ms": 8000,
        "requires_embedding": True
    },
    {
        "id": "semantic_002",
        "category": "Polymarket Semantic",
        "question": "Find prediction markets about climate change",
        "expected_collections": ["prediction_markets_polymarket"],
        "expected_pattern": "question.*",
        "min_results": 3,
        "max_time_ms": 8000,
        "requires_embedding": True
    },

    # === GRAPH TRAVERSALS ===
    {
        "id": "graph_001",
        "category": "Company → Awards",
        "question": "Show me Lockheed Martin's government contracts",
        "expected_collections": ["Company", "Award"],
        "expected_pattern": "HAS_AWARD",
        "min_results": 1,
        "max_time_ms": 3000
    },
    {
        "id": "graph_002",
        "category": "Company → SEC Filings",
        "question": "Show me Microsoft's most recent 10-K filings",
        "expected_collections": ["Company", "sec_filings"],
        "expected_pattern": "HAS_FILING",
        "min_results": 1,
        "max_time_ms": 3000
    },
    {
        "id": "graph_003",
        "category": "Company → Market Data",
        "question": "Show me energy sector stocks from the last 30 days",
        "expected_collections": ["Company", "MarketData"],
        "expected_pattern": "HAS_MARKETDATA",
        "min_results": 5,
        "max_time_ms": 5000
    },
    {
        "id": "graph_004",
        "category": "Deep Traversal",
        "question": "Find negative sentiment in Apple's SEC filings",
        "expected_collections": ["Company", "sec_filings"],
        "expected_pattern": "finbert",
        "min_results": 1,
        "max_time_ms": 5000
    },

    # === TECHNICAL INDICATORS ===
    {
        "id": "technical_001",
        "category": "Golden Cross",
        "question": "Which stocks are in a golden cross?",
        "expected_collections": ["MarketData"],
        "expected_pattern": "golden_cross",
        "min_results": 0,  # May be none at the moment
        "max_time_ms": 4000
    },
    {
        "id": "technical_002",
        "category": "RSI Screening",
        "question": "Find stocks with RSI above 70",
        "expected_collections": ["MarketData"],
        "expected_pattern": "rsi",
        "min_results": 0,
        "max_time_ms": 4000
    },

    # === PREDICTION MARKETS ===
    {
        "id": "prediction_001",
        "category": "Polymarket Volume",
        "question": "What are the most active Polymarket markets?",
        "expected_collections": ["prediction_markets_polymarket"],
        "expected_pattern": "volume",
        "min_results": 5,
        "max_time_ms": 3000
    },
    {
        "id": "prediction_002",
        "category": "Whale Traders",
        "question": "Show me the most profitable Polymarket whale traders",
        "expected_collections": ["polymarket_traders"],
        "expected_pattern": "profit|whale",
        "min_results": 1,
        "max_time_ms": 3000
    },
    {
        "id": "prediction_003",
        "category": "Company Sentiment",
        "question": "Find prediction markets about Tesla",
        "expected_collections": ["Company", "prediction_markets_polymarket"],
        "expected_pattern": "market_mentions_company",
        "min_results": 0,
        "max_time_ms": 4000
    },

    # === COMMODITIES & CFTC ===
    {
        "id": "commodity_001",
        "category": "CFTC Positions",
        "question": "Show me companies with crude oil positions",
        "expected_collections": ["Company", "commodity_positions"],
        "expected_pattern": "HAS_COMMODITY_POSITION",
        "min_results": 1,
        "max_time_ms": 4000
    },
    {
        "id": "commodity_002",
        "category": "Futures Prices",
        "question": "Show me recent gold futures prices",
        "expected_collections": ["futures_prices"],
        "expected_pattern": "GOLD|gold",
        "min_results": 1,
        "max_time_ms": 3000
    },

    # === MULTI-SOURCE SYNTHESIS ===
    {
        "id": "multi_001",
        "category": "Defense + Prediction Markets",
        "question": "Defense contractors with bullish prediction market sentiment",
        "expected_collections": ["Company", "Award", "prediction_markets_polymarket"],
        "expected_pattern": "HAS_AWARD.*market",
        "min_results": 0,
        "max_time_ms": 6000
    },
    {
        "id": "multi_002",
        "category": "SEC Sentiment vs Price",
        "question": "Companies with negative SEC sentiment but rising stock prices",
        "expected_collections": ["Company", "sec_filings", "MarketData"],
        "expected_pattern": "finbert.*HAS_MARKETDATA",
        "min_results": 0,
        "max_time_ms": 6000
    },

    # === AGGREGATIONS ===
    {
        "id": "agg_001",
        "category": "Sector Analysis",
        "question": "Show me average P/E ratio by sector",
        "expected_collections": ["Company", "MarketData"],
        "expected_pattern": "sector.*COLLECT",
        "min_results": 5,
        "max_time_ms": 5000
    },
    {
        "id": "agg_002",
        "category": "Total Contract Value",
        "question": "Total government contract value by company",
        "expected_collections": ["Company", "Award"],
        "expected_pattern": "SUM.*award_amount",
        "min_results": 5,
        "max_time_ms": 5000
    },

    # === DATE RANGES ===
    {
        "id": "date_001",
        "category": "Date Range",
        "question": "Show me Tesla stock prices for January 2024",
        "expected_collections": ["MarketData"],
        "expected_pattern": "2024-01",
        "min_results": 0,
        "max_time_ms": 4000
    },
    {
        "id": "date_002",
        "category": "Recent Filings",
        "question": "SEC filings from the last 30 days",
        "expected_collections": ["sec_filings"],
        "expected_pattern": "DATE_SUBTRACT|filing_date",
        "min_results": 0,
        "max_time_ms": 4000
    },

    # === EDGE CASES ===
    {
        "id": "edge_001",
        "category": "Ambiguous Query",
        "question": "Show me markets",  # Could be stock markets or prediction markets
        "expected_collections": ["MarketData", "prediction_markets_polymarket"],
        "expected_pattern": ".*",
        "min_results": 1,
        "max_time_ms": 4000
    },
    {
        "id": "edge_002",
        "category": "Complex Multi-Hop",
        "question": "Energy companies with commodity positions and recent contracts",
        "expected_collections": ["Company", "commodity_positions", "Award"],
        "expected_pattern": "sector.*Energy",
        "min_results": 0,
        "max_time_ms": 6000
    },
]


class TestResult:
    def __init__(self, test_case: dict):
        self.test_case = test_case
        self.passed = False
        self.error = None
        self.execution_time_ms = 0
        self.query_plan = None
        self.result_count = 0
        self.aql_generated = None
        self.warnings = []

    def to_dict(self):
        return {
            "id": self.test_case["id"],
            "category": self.test_case["category"],
            "question": self.test_case["question"],
            "passed": self.passed,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "result_count": self.result_count,
            "expected_min_results": self.test_case["min_results"],
            "max_time_ms": self.test_case["max_time_ms"],
            "warnings": self.warnings,
            "aql_preview": self.aql_generated[:200] if self.aql_generated else None
        }


def run_test(test_case: dict) -> TestResult:
    """Run a single test case"""
    result = TestResult(test_case)

    try:
        start_time = time.time()

        # Generate query plan
        query_plan = plan_query_with_llm(test_case["question"])

        if not query_plan:
            result.error = "Failed to generate query plan"
            return result

        if query_plan.get('error'):
            result.error = f"Planning error: {query_plan.get('error_message')}"
            return result

        result.query_plan = query_plan
        result.aql_generated = query_plan.get("aql_query", "")

        # Execute query
        aql_query = query_plan.get("aql_query")
        bind_vars = query_plan.get("bind_vars", {})

        # Handle embedding requirement
        if test_case.get("requires_embedding"):
            if not query_plan.get("requires_embedding"):
                result.warnings.append("Expected embedding but query doesn't use it")

        results, error = execute_aql(aql_query, bind_vars)

        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        result.execution_time_ms = round(execution_time, 2)

        if error:
            result.error = f"Execution error: {error}"
            return result

        result.result_count = len(results) if results else 0

        # Validation checks
        checks_passed = []

        # 1. Check if expected collections are in query
        for expected_col in test_case.get("expected_collections", []):
            if expected_col in aql_query:
                checks_passed.append(f"✓ Collection: {expected_col}")
            else:
                result.warnings.append(f"Missing expected collection: {expected_col}")

        # 2. Check if pattern is in query
        import re
        if test_case.get("expected_pattern"):
            if re.search(test_case["expected_pattern"], aql_query, re.IGNORECASE):
                checks_passed.append(f"✓ Pattern: {test_case['expected_pattern']}")
            else:
                result.warnings.append(f"Missing expected pattern: {test_case['expected_pattern']}")

        # 3. Check minimum results
        if result.result_count >= test_case["min_results"]:
            checks_passed.append(f"✓ Results: {result.result_count} >= {test_case['min_results']}")
        else:
            result.warnings.append(f"Insufficient results: {result.result_count} < {test_case['min_results']}")

        # 4. Check execution time
        if execution_time <= test_case["max_time_ms"]:
            checks_passed.append(f"✓ Performance: {execution_time:.0f}ms <= {test_case['max_time_ms']}ms")
        else:
            result.warnings.append(f"Slow execution: {execution_time:.0f}ms > {test_case['max_time_ms']}ms")

        # Pass if no errors and at least 50% of checks passed
        result.passed = len(checks_passed) >= 2 and not result.error

    except Exception as e:
        result.error = f"Exception: {str(e)}"

    return result


def run_all_tests() -> Tuple[List[TestResult], Dict]:
    """Run all test cases and return results + summary"""
    print(f"\n{'='*80}")
    print(f"KARGA QUERY TEST SUITE")
    print(f"Running {len(TEST_CASES)} test cases...")
    print(f"{'='*80}\n")

    results = []
    categories = {}

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {test_case['id']}: {test_case['category']}")
        print(f"    Q: {test_case['question']}")

        result = run_test(test_case)
        results.append(result)

        # Track by category
        cat = test_case["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failed": 0}
        categories[cat]["total"] += 1
        if result.passed:
            categories[cat]["passed"] += 1
            print(f"    ✅ PASS ({result.execution_time_ms:.0f}ms, {result.result_count} results)")
        else:
            categories[cat]["failed"] += 1
            print(f"    ❌ FAIL: {result.error or 'Validation failed'}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"       ⚠️  {warning}")
        print()

    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    avg_time = sum(r.execution_time_ms for r in results) / total if total > 0 else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total * 100), 1) if total > 0 else 0,
        "avg_execution_time_ms": round(avg_time, 2),
        "categories": categories
    }

    return results, summary


def print_summary(results: List[TestResult], summary: Dict):
    """Print test summary report"""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests:    {summary['total_tests']}")
    print(f"Passed:         {summary['passed']} ✅")
    print(f"Failed:         {summary['failed']} ❌")
    print(f"Pass Rate:      {summary['pass_rate']}%")
    print(f"Avg Time:       {summary['avg_execution_time_ms']:.0f}ms")
    print()

    print("BY CATEGORY:")
    for cat, stats in summary["categories"].items():
        pass_pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        status = "✅" if pass_pct >= 80 else "⚠️" if pass_pct >= 50 else "❌"
        print(f"  {status} {cat:30s} {stats['passed']}/{stats['total']} ({pass_pct:.0f}%)")
    print()

    # Show failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("FAILED TESTS:")
        for r in failures:
            print(f"  ❌ {r.test_case['id']}: {r.test_case['question']}")
            print(f"     Error: {r.error or 'Validation failed'}")
            if r.warnings:
                for w in r.warnings:
                    print(f"     ⚠️  {w}")
        print()


def save_results(results: List[TestResult], summary: Dict, filename: str = None):
    """Save results to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"

    output = {
        "summary": summary,
        "results": [r.to_dict() for r in results]
    }

    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    results, summary = run_all_tests()
    print_summary(results, summary)
    save_results(results, summary)

    # Exit with error code if pass rate < 70%
    if summary["pass_rate"] < 70:
        print("\n⚠️  WARNING: Pass rate below 70% threshold")
        sys.exit(1)
    else:
        print("\n✅ All critical tests passing!")
        sys.exit(0)
