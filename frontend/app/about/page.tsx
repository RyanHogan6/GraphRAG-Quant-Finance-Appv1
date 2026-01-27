export default function About() {
  return (
    <div className="min-h-screen bg-dark-900 text-gray-200 py-8 md:py-16 px-4 md:px-6 antialiased">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl md:text-5xl font-bold text-gold mb-4 md:mb-6 font-mono tracking-tight">About KARGA Markets</h1>
        <p className="text-lg md:text-xl text-gray-400 mb-8 md:mb-12 font-light">
          A technical deep-dive into AI-powered financial knowledge graphs
        </p>

        <div className="space-y-16">
          {/* Problem Statement */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">📊</span>
              The Problem
            </h2>
            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-8">
              <p className="text-gray-300 mb-4 text-lg leading-relaxed font-light">
                Traditional financial data platforms store information in isolated silos. Stock prices live in one database,
                government contracts in another, SEC filings in a third, and prediction markets somewhere else entirely.
              </p>
              <p className="text-gray-300 mb-4 text-lg leading-relaxed font-light">
                When you want to answer questions like <em className="text-gold font-normal">"Which S&P 500 companies with significant
                  government contracts are mentioned in prediction markets?"</em> - you'd need to manually query multiple
                systems, export data, and perform complex joins in spreadsheets.
              </p>
              <p className="text-gold font-semibold text-lg">
                There had to be a better way.
              </p>
            </div>
          </section>

          {/* Solution */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">⚡</span>
              The Solution: KARGA
            </h2>
            <div className="space-y-6">
              <p className="text-gray-300 text-base md:text-lg leading-relaxed">
                KARGA Markets combines three powerful technologies:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <h3 className="text-lg md:text-xl font-semibold text-gold mb-3">1. Knowledge Graphs</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    Data stored as interconnected nodes and relationships using ArangoDB, enabling
                    complex multi-hop queries across disparate data sources in milliseconds.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <h3 className="text-lg md:text-xl font-semibold text-gold mb-3">2. Retrieval Augmented Generation</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    AI (GPT-4) generates precise database queries from natural language, then analyzes
                    results with full context - no hallucination, only real data.
                  </p>
                </div>

                <div className="bg-dark-800 border border-gold/30 rounded-lg p-4 md:p-6 hover:border-gold/60 transition-all">
                  <h3 className="text-lg md:text-xl font-semibold text-gold mb-3">3. Semantic Search</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    Vector embeddings enable concept-based search - find "cybersecurity contracts"
                    even when documents use terms like "network security" or "threat detection."
                  </p>
                </div>
              </div>

              <div className="mt-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-blue-300 mb-4">Recent Enhancements</h3>
                <div className="space-y-3 text-sm text-gray-300">
                  <div className="flex items-start gap-3">
                    <span className="text-blue-400 font-bold">•</span>
                    <div>
                      <strong className="text-blue-300">SEC Sentence Embeddings:</strong> 4.36M sentences from 10-K/10-Q filings
                      trained with Doc2Vec (300-dim financial domain embeddings) for semantic search over regulatory disclosures
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-blue-400 font-bold">•</span>
                    <div>
                      <strong className="text-blue-300">Visual Query Builder:</strong> Smart field type detection
                      (date/number/boolean/text) with date pickers, number inputs, and optimized operator selection
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-blue-400 font-bold">•</span>
                    <div>
                      <strong className="text-blue-300">Enriched Results Display:</strong> AI inference engine
                      generates contextual insights from multi-source queries with collapsible drill-down sections
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-blue-400 font-bold">•</span>
                    <div>
                      <strong className="text-blue-300">Options Flow Detection:</strong> Unusual activity detection
                      with 20-day baseline averages for insider trading signal identification
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Architecture Diagram */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">🔧</span>
              System Architecture
            </h2>

            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-8 overflow-x-auto">
              <pre className="text-xs md:text-sm text-green-400 font-mono leading-relaxed">
                {`┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│  Next.js 14 + React + Framer Motion + Tailwind CSS                  │
│  • Natural Language Query Input                                      │
│  • Interactive Graph Visualization (ReactFlow)                       │
│  • Real-time Market Cards                                            │
│  • Data Tables with Filtering/Sorting                                │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTPS / REST API
┌─────────────────────▼───────────────────────────────────────────────┐
│                     FASTAPI BACKEND                                  │
│  Python 3.13 + FastAPI + Pydantic                                    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  Query Pipeline (Parallel Execution)                  │           │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │           │
│  │  │ GPT-4      │  │ Perplexity   │  │ ArangoDB     │ │           │
│  │  │ Intent     │  │ Web Search   │  │ Graph Query  │ │           │
│  │  │ Detection  │  │ (Current     │  │ (Historical  │ │           │
│  │  │            │  │  Events)     │  │  Data)       │ │           │
│  │  └────┬───────┘  └──────┬───────┘  └──────┬───────┘ │           │
│  │       │                 │                   │         │           │
│  │       └─────────────────┴───────────────────┘         │           │
│  │                         │                              │           │
│  │                  ┌──────▼──────────┐                  │           │
│  │                  │  GPT-4 Synthesis │                  │           │
│  │                  │  Combines Results│                  │           │
│  │                  └──────────────────┘                  │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                       │
│  Security: Rate Limiting • Input Validation • HSTS Headers           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ AQL Queries
┌─────────────────────▼───────────────────────────────────────────────┐
│                    ARANGODB CLOUD                                    │
│  Multi-Model Database (Document + Graph + Search)                   │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Document    │  │  Graph       │  │  Vector      │              │
│  │  Collections │  │  Edges       │  │  Embeddings  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
│  • 612 companies (S&P 500)                                           │
│  • 2M+ daily market data points (OHLCV + 40 indicators)             │
│  • 100K+ government contract awards (with embeddings)                │
│  • 7.5K SEC filings + 4.36M sentences (Doc2Vec embeddings)          │
│  • 20K+ prediction markets (Polymarket + Kalshi)                    │
│  • 64K+ futures prices (CME commodities)                             │
│  • CFTC commodity positions + EIA energy data                        │
│  • Daily options flow (612 tickers)                                  │
└───────────────────────────────────────────────────────────────────────┘`}
              </pre>
            </div>
          </section>

          {/* Data Sources */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">📊</span>
              Data Sources & Integration
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
              {/* Stock Market Data */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-green-400 mb-4">📊 Stock Market Data</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> Yahoo Finance API</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Coverage:</strong> All S&P 500 companies</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> OHLCV, volume, market cap, P/E ratios, technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Update Frequency:</strong> Daily via Airflow DAG</span>
                  </li>
                </ul>
              </div>

              {/* Government Contracts */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-blue-400 mb-4">📋 Government Contracts</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> USASpending.gov API</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Coverage:</strong> Federal contract awards to public companies</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> Award amounts, agencies, descriptions, dates</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Special Feature:</strong> Vector embeddings for semantic search (find "AI contracts" without exact keyword match)</span>
                  </li>
                </ul>
              </div>

              {/* Prediction Markets */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-purple-400 mb-4">📈 Prediction Markets</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Sources:</strong> Polymarket API, Kalshi API</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Coverage:</strong> Politics, economics, sports, entertainment</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> Probabilities, volumes, liquidity, traders</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Connection:</strong> Graph edges link markets to mentioned companies (e.g., "Will Tesla reach $300?" → Tesla stock)</span>
                  </li>
                </ul>
              </div>

              {/* SEC Filings */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-orange-400 mb-4">📋 SEC Filings</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> SEC EDGAR API</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Types:</strong> 10-K (annual), 10-Q (quarterly), 8-K (events)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Processing:</strong> 7.5K filings parsed into 4.36M sentences</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Sentiment:</strong> FinBERT scores for each sentence (-1 to +1)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Embeddings:</strong> Doc2Vec 300-dim financial domain embeddings for semantic search</span>
                  </li>
                </ul>
              </div>

              {/* Economic Indicators */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-cyan-400 mb-4">📊 Economic Indicators</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> Federal Reserve Economic Data (FRED)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> S&P 500 index, Fed funds rate, unemployment, GDP, yield curves</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Coverage:</strong> Historical time series data</span>
                  </li>
                </ul>
              </div>

              {/* Commodity Positions */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-yellow-400 mb-4">📊 Commodity Positions</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> CFTC Commitments of Traders Report</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> Long/short positions by trader type (commercial, non-commercial, retail)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Commodities:</strong> Oil, gold, wheat, corn, natural gas, etc.</span>
                  </li>
                </ul>
              </div>

              {/* Options Flow */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-indigo-400 mb-4">📈 Options Flow</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> Options data via yfinance</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Coverage:</strong> All S&P 500 companies (612 tickers)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> Call/put volumes, open interest, implied volatility, put/call ratios</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Detection:</strong> Unusual activity flags for potential insider trading signals</span>
                  </li>
                </ul>
              </div>

              {/* Futures Prices */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-amber-400 mb-4">🌾 Futures Prices</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> CME Group via yfinance</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Records:</strong> 64,000+ historical prices</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Commodities:</strong> Crude oil, natural gas, gold, silver, copper, corn, wheat, soybeans</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Indicators:</strong> OHLCV data plus technical indicators (RSI, MACD, SMA)</span>
                  </li>
                </ul>
              </div>

              {/* EIA Energy Data */}
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-emerald-400 mb-4">⚡ EIA Energy Data</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Source:</strong> U.S. Energy Information Administration API</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Data:</strong> Crude oil inventory, natural gas storage, production, LNG exports</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Frequency:</strong> Weekly and monthly updates</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gold mr-2">•</span>
                    <span><strong>Connection:</strong> Linked to futures prices for supply/demand analysis</span>
                  </li>
                </ul>
              </div>
            </div>
          </section>

          {/* Graph Structure */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">🔗</span>
              Knowledge Graph Structure
            </h2>

            <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-8">
              <p className="text-gray-300 mb-6 leading-relaxed">
                Data isn't just stored—it's <strong className="text-gold">connected</strong>. Here's how relationships enable powerful queries:
              </p>

              <div className="space-y-4">
                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-green-400 mb-2">Company → Market Data</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">HAS_MARKETDATA</code> edges connect
                    companies to their daily stock prices, enabling queries like "Show me tech companies
                    with SMA_50 &gt; SMA_200 (golden cross)"
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-blue-400 mb-2">Company → Government Awards</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">HAS_AWARD</code> edges link companies
                    to contracts, enabling semantic searches: "Defense companies with cybersecurity contracts over $10M"
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-purple-400 mb-2">Market → Company</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">market_mentions_company</code> edges
                    connect prediction markets to mentioned tickers: "Tesla reaches $300" → TSLA
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-orange-400 mb-2">Company → SEC Filings → Sentences</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">HAS_FILING → has_section → has_sentence</code>
                    Multi-hop traversal for sentiment analysis: "Show negative FinBERT sentences from recent Apple 10-Ks"
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-cyan-400 mb-2">Company → Commodity Positions</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">HAS_COMMODITY_POSITION</code> links
                    companies to CFTC data for commodity exposure analysis
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-indigo-400 mb-2">Company → Options Flow</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">COMPANY_HAS_OPTIONS</code> connects companies to daily options activity.
                    <code className="bg-dark-900 px-2 py-1 rounded">OPTIONS_BEFORE_AWARD</code> flags unusual activity before contract awards
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-amber-400 mb-2">CFTC → Futures → EIA Data</h4>
                  <p className="text-sm text-gray-400">
                    <code className="bg-dark-900 px-2 py-1 rounded">POSITION_ON_COMMODITY</code>,
                    <code className="bg-dark-900 px-2 py-1 rounded">INVENTORY_AFFECTS_PRICE</code>,
                    <code className="bg-dark-900 px-2 py-1 rounded">STORAGE_AFFECTS_PRICE</code> enable supply/demand correlation analysis
                  </p>
                </div>

                <div className="bg-dark-700 border border-gold/10 rounded-lg p-6">
                  <h4 className="font-semibold text-emerald-400 mb-2">SEC Sentences (Semantic Search)</h4>
                  <p className="text-sm text-gray-400">
                    Multi-hop traversal <code className="bg-dark-900 px-2 py-1 rounded">HAS_FILING → has_section → has_sentence</code>
                    with Doc2Vec embeddings: "Find sentences discussing supply chain risks in energy sector 10-Ks"
                  </p>
                </div>
              </div>

              <div className="mt-6 md:mt-8 p-3 md:p-4 bg-gold/10 border border-gold/30 rounded-lg">
                <p className="text-xs md:text-sm text-gold font-semibold mb-2">💡 Example Multi-Hop Query</p>
                <p className="text-xs md:text-sm text-gray-300 font-mono">
                  "Find energy companies with government contracts mentioning 'renewable' that are
                  mentioned in prediction markets with volume &gt; $50k"
                </p>
                <p className="text-[10px] md:text-xs text-gray-400 mt-2">
                  → Traverses Company → Awards (semantic search) → Markets (graph join) in milliseconds
                </p>
              </div>
            </div>
          </section>

          {/* Query Pipeline */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">🤖</span>
              AI Query Pipeline
            </h2>

            <div className="space-y-6">
              <p className="text-gray-300 text-base md:text-lg leading-relaxed">
                When you ask a question, here's what happens behind the scenes:
              </p>

              <div className="space-y-3 md:space-y-4">
                {/* Step 1 */}
                <div className="bg-dark-800 border-l-4 border-green-500 p-4 md:p-6 rounded-r-lg">
                  <h4 className="font-semibold text-green-400 mb-2 text-base md:text-lg">Step 1: Intent Detection</h4>
                  <p className="text-gray-400 mb-3 text-sm md:text-base">
                    GPT-4 classifies your query: Is it about a specific ticker (AAPL, MSFT) or a concept
                    (AI, cybersecurity)? This determines whether to use exact matching or semantic search.
                  </p>
                  <div className="bg-dark-900 p-2 md:p-3 rounded font-mono text-[10px] md:text-xs">
                    <span className="text-gray-500">Input:</span> <span className="text-gray-300">"Show me AI companies with government contracts"</span><br />
                    <span className="text-gray-500">Intent:</span> <span className="text-green-400">concept_query</span>
                  </div>
                </div>

                {/* Step 2 */}
                <div className="bg-dark-800 border-l-4 border-blue-500 p-4 md:p-6 rounded-r-lg">
                  <h4 className="font-semibold text-blue-400 mb-2 text-base md:text-lg">Step 2: Query Planning</h4>
                  <p className="text-gray-400 mb-3 text-sm md:text-base">
                    GPT-4 receives the full database schema (collections, fields, relationships) and
                    generates optimized AQL (ArangoDB Query Language) with proper joins and filters.
                  </p>
                  <div className="bg-dark-900 p-2 md:p-3 rounded font-mono text-[10px] md:text-xs overflow-x-auto">
                    <pre className="text-gray-300">{`FOR award IN Award
  FILTER COSINE_SIMILARITY(
    award.description_embedding,
    @query_vector
  ) >= 0.75
  FOR company IN Company
    FILTER company.ticker == award.ticker
    RETURN {company, award}`}</pre>
                  </div>
                </div>

                {/* Step 3 */}
                <div className="bg-dark-800 border-l-4 border-purple-500 p-4 md:p-6 rounded-r-lg">
                  <h4 className="font-semibold text-purple-400 mb-2 text-base md:text-lg">Step 3: Parallel Execution</h4>
                  <p className="text-gray-400 mb-3 text-sm md:text-base">
                    Two queries run simultaneously:
                  </p>
                  <ul className="text-gray-400 text-xs md:text-sm space-y-2">
                    <li>• <strong className="text-purple-300">Database Query:</strong> AQL executes against ArangoDB (historical data)</li>
                    <li>• <strong className="text-purple-300">Web Search:</strong> Perplexity searches for current events (real-time context)</li>
                  </ul>
                </div>

                {/* Step 4 */}
                <div className="bg-dark-800 border-l-4 border-orange-500 p-4 md:p-6 rounded-r-lg">
                  <h4 className="font-semibold text-orange-400 mb-2 text-base md:text-lg">Step 4: Synthesis & Analysis</h4>
                  <p className="text-gray-400 mb-3 text-sm md:text-base">
                    GPT-4 combines database results with web context, analyzes patterns, and generates:
                  </p>
                  <ul className="text-gray-400 text-xs md:text-sm space-y-2">
                    <li>• <strong className="text-orange-300">Markdown Tables:</strong> Formatted results with key metrics</li>
                    <li>• <strong className="text-orange-300">Insights:</strong> Trends, correlations, anomalies</li>
                    <li>• <strong className="text-orange-300">Follow-up Questions:</strong> Suggested deeper dives</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Technical Stack */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">🔧</span>
              Technology Stack
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-gold mb-4">Frontend</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li><strong className="text-gray-300">Framework:</strong> Next.js 14 (App Router)</li>
                  <li><strong className="text-gray-300">UI:</strong> React 18, TypeScript</li>
                  <li><strong className="text-gray-300">Styling:</strong> Tailwind CSS</li>
                  <li><strong className="text-gray-300">Animations:</strong> Framer Motion</li>
                  <li><strong className="text-gray-300">Graph Viz:</strong> ReactFlow</li>
                  <li><strong className="text-gray-300">Hosting:</strong> Vercel</li>
                </ul>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-gold mb-4">Backend</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li><strong className="text-gray-300">Framework:</strong> FastAPI (Python 3.13)</li>
                  <li><strong className="text-gray-300">Validation:</strong> Pydantic</li>
                  <li><strong className="text-gray-300">Security:</strong> SlowAPI rate limiting</li>
                  <li><strong className="text-gray-300">LLM:</strong> OpenAI GPT-4</li>
                  <li><strong className="text-gray-300">Web Search:</strong> Perplexity AI</li>
                  <li><strong className="text-gray-300">Hosting:</strong> Railway</li>
                </ul>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-gold mb-4">Database</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li><strong className="text-gray-300">Platform:</strong> ArangoDB Cloud</li>
                  <li><strong className="text-gray-300">Type:</strong> Multi-model (Document + Graph)</li>
                  <li><strong className="text-gray-300">Query Language:</strong> AQL</li>
                  <li><strong className="text-gray-300">Embeddings:</strong> OpenAI text-embedding-3-small</li>
                  <li><strong className="text-gray-300">Size:</strong> ~5GB (2M+ documents)</li>
                  <li><strong className="text-gray-300">Location:</strong> Germany (GDPR compliant)</li>
                </ul>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h3 className="text-lg md:text-xl font-semibold text-gold mb-4">Data Pipeline</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li><strong className="text-gray-300">Orchestration:</strong> Apache Airflow</li>
                  <li><strong className="text-gray-300">Processing:</strong> Python, Pandas, NumPy</li>
                  <li><strong className="text-gray-300">Sentiment:</strong> FinBERT</li>
                  <li><strong className="text-gray-300">Schedule:</strong> Daily updates at 2 AM UTC</li>
                  <li><strong className="text-gray-300">Monitoring:</strong> Airflow UI + logs</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Performance */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">⚡</span>
              Performance & Scale
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
              <div className="bg-gradient-to-br from-green-900/20 to-green-800/10 border border-green-500/30 rounded-lg p-6 text-center">
                <div className="text-4xl font-bold text-green-400 mb-2">&lt; 3s</div>
                <div className="text-sm text-gray-400">Average Query Time</div>
                <div className="text-xs text-gray-500 mt-2">(DB + AI analysis)</div>
              </div>

              <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/10 border border-blue-500/30 rounded-lg p-6 text-center">
                <div className="text-4xl font-bold text-blue-400 mb-2">2M+</div>
                <div className="text-sm text-gray-400">Documents in Graph</div>
                <div className="text-xs text-gray-500 mt-2">(Companies, markets, filings)</div>
              </div>

              <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/10 border border-purple-500/30 rounded-lg p-6 text-center">
                <div className="text-4xl font-bold text-purple-400 mb-2">50ms</div>
                <div className="text-sm text-gray-400">Graph Traversal Time</div>
                <div className="text-xs text-gray-500 mt-2">(3-hop relationships)</div>
              </div>
            </div>

            <div className="mt-6 md:mt-8 bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
              <h4 className="font-semibold text-gold mb-3 md:mb-4 text-base md:text-lg">Performance Optimizations</h4>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4 text-gray-400 text-sm">
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Persistent indexes on ticker, date, volume fields
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Skip-list indexes for range queries
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Edge collections for O(1) relationship lookups
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Query result caching (5-minute TTL)
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Parallel DB + web search execution
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  Streaming results with batch_size=1000
                </li>
              </ul>
            </div>
          </section>

          {/* Future Roadmap */}
          <section>
            <h2 className="text-2xl md:text-3xl font-semibold text-gold mb-4 md:mb-6 flex items-center">
              <span className="text-4xl mr-3">📈</span>
              Future Enhancements
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">⚡ Real-time Data</h4>
                <p className="text-gray-400 text-sm">
                  WebSocket connections for live market data updates, streaming prediction
                  market probability changes as they happen.
                </p>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">📊 Portfolio Tracking</h4>
                <p className="text-gray-400 text-sm">
                  User accounts to track favorite companies, save queries, and set up alerts
                  for specific market conditions.
                </p>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">🤖 Advanced ML Models</h4>
                <p className="text-gray-400 text-sm">
                  Time-series forecasting with LSTM, anomaly detection for unusual trading
                  patterns, correlation discovery between data sources.
                </p>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">📊 More Data Sources</h4>
                <p className="text-gray-400 text-sm">
                  Twitter sentiment, Reddit discussions, earnings call transcripts, patent
                  filings, and international market data.
                </p>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">📊 Custom Dashboards</h4>
                <p className="text-gray-400 text-sm">
                  Drag-and-drop dashboard builder with custom charts, metrics, and KPIs
                  tailored to individual research needs.
                </p>
              </div>

              <div className="bg-dark-800 border border-gold/20 rounded-lg p-4 md:p-6">
                <h4 className="font-semibold text-gold mb-3">🔗 API Access</h4>
                <p className="text-gray-400 text-sm">
                  Public API with authentication for programmatic access to KARGA
                  capabilities, enabling integrations with trading platforms.
                </p>
              </div>
            </div>
          </section>

          {/* Open Source */}
          <section className="bg-gradient-to-r from-gold/10 to-gold/5 border-2 border-gold/30 rounded-lg p-8">
            <h2 className="text-3xl font-semibold text-gold mb-4 flex items-center">
              <span className="text-4xl mr-3">🔓</span>
              Open Source & Contributions
            </h2>
            <p className="text-gray-300 text-lg leading-relaxed mb-4">
              KARGA Markets is built with transparency in mind. While the core application is
              proprietary, we're exploring open-sourcing components of the query planning system
              and graph schema to help others build similar systems.
            </p>
            <p className="text-gray-400 leading-relaxed">
              Interested in collaborating or have ideas for improvement? Reach out at{' '}
              <a href="mailto:karga.analytics@gmail.com" className="text-gold hover:underline">
                karga.analytics@gmail.com
              </a>
            </p>
          </section>

          {/* Bottom CTA */}
          <div className="text-center pt-6 md:pt-8">
            <a
              href="/"
              className="inline-block px-8 py-4 bg-gold/20 border-2 border-gold/40 rounded-lg text-gold font-semibold hover:bg-gold/30 hover:border-gold/60 transition-all text-lg"
            >
              ← Back to Platform
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
