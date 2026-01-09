'use client'

import { useState } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Welcome to GraphRAG! Ask me anything about financial markets, SEC filings, prediction markets, or run complex queries across our knowledge graph.',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // TODO: Replace with actual API call
    setTimeout(() => {
      const assistantMessage: Message = {
        role: 'assistant',
        content: `I received your question: "${input}". This is a mock response. Once we connect the backend, I'll execute AQL queries against the knowledge graph and provide real insights!`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1500)
  }

  const suggestedQuestions = [
    'Show me AAPL stock performance over the last 30 days',
    'What are the top 10 government contracts by value?',
    'Find defense contracts related to AI and cybersecurity',
    'Show me companies with the highest P/E ratios in tech',
    'What is the current federal funds rate?',
    'Show me prediction markets about the 2028 election',
  ]

  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-gold mb-2">💬 AI Query Interface</h1>
        <p className="text-gray-500">Natural language queries over financial knowledge graph</p>
      </div>

      {/* Chat Container */}
      <div className="bg-dark-800 border border-gold/20 rounded-lg shadow-xl mb-6">
        {/* Messages */}
        <div className="h-[500px] overflow-y-auto p-6 space-y-4">
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  message.role === 'user'
                    ? 'bg-gold/20 border border-gold/40 text-gray-100'
                    : 'bg-dark-700 border border-gold/20 text-gray-300'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className="text-xl">
                    {message.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm mb-2 leading-relaxed">{message.content}</div>
                    <div className="text-xs text-gray-600">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-dark-700 border border-gold/20 rounded-lg p-4">
                <div className="flex items-center space-x-3">
                  <div className="text-xl">🤖</div>
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gold rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-gold/20 p-4">
          <form onSubmit={handleSubmit} className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about markets, companies, or contracts..."
              className="flex-1 bg-dark-700 border border-gold/30 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-gold/20 border border-gold/40 rounded-lg text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
            >
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Suggested Questions */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gold mb-4">💡 Suggested Questions</h2>
        <div className="grid grid-cols-2 gap-3">
          {suggestedQuestions.map((question, idx) => (
            <button
              key={idx}
              onClick={() => setInput(question)}
              className="bg-dark-800 border border-gold/20 rounded-lg p-4 text-left text-sm text-gray-400 hover:border-gold/40 hover:text-gold transition-all"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-2xl mb-2">📊</div>
          <div className="text-gold font-semibold mb-1">Market Data</div>
          <div className="text-xs text-gray-500">OHLCV, technical indicators, fundamentals</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-2xl mb-2">🏛️</div>
          <div className="text-gold font-semibold mb-1">Gov Contracts</div>
          <div className="text-xs text-gray-500">Federal awards with semantic search</div>
        </div>
        <div className="bg-dark-800 border border-gold/20 rounded-lg p-5">
          <div className="text-2xl mb-2">📄</div>
          <div className="text-gold font-semibold mb-1">SEC Filings</div>
          <div className="text-xs text-gray-500">Sentiment analysis, sections, sentences</div>
        </div>
      </div>
    </div>
  )
}
