'use client'

import { useState } from 'react'
import MarkdownRenderer from '@/components/MarkdownRenderer'
import ResultsTable from '@/components/ResultsTable'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  results?: any[]
  useMarkdown?: boolean
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
    const currentInput = input
    setInput('')
    setIsLoading(true)

    try {
      // Import api at the top of the file
      const { api } = await import('@/lib/api')

      const response = await api.executeQuery(currentInput)

      // Build analysis message with metadata header
      let resultText = `**Query Results:** ${response.count} results in ${response.execution_time.toFixed(2)}s\n\n`

      // Use AI-generated analysis (should include markdown tables)
      if (response.analysis) {
        resultText += response.analysis
      }

      // Add follow-up questions
      if (response.follow_up_questions && response.follow_up_questions.length > 0) {
        resultText += '\n\n**Follow-up questions:**\n'
        response.follow_up_questions.forEach((q: string) => {
          resultText += `- ${q}\n`
        })
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: resultText,
        timestamp: new Date(),
        results: response.results,
        useMarkdown: true,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const suggestedQuestions = [
    'What do prediction markets say about the 2024 election outcome?',
    'Show me defense contracts related to AI and cybersecurity',
    'Which tech stocks have the highest institutional ownership?',
    'Find government contracts mentioning China or Taiwan',
    'What are the biggest bets on Polymarket right now?',
    'Show me companies with recent SEC filings mentioning recession',
  ]

  return (
    <div className="container mx-auto px-6 py-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-gold mb-2">AI Query Interface</h1>
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
                  <div className="text-xs font-semibold text-gray-500 uppercase">
                    {message.role === 'user' ? 'You' : 'AI'}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm mb-2 leading-relaxed">
                      {message.useMarkdown ? (
                        <MarkdownRenderer content={message.content} />
                      ) : (
                        message.content
                      )}
                    </div>
                    {message.results && message.results.length > 0 && !message.content.includes('|') && (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-xs text-gold hover:text-gold/80 font-semibold">
                          View raw data table ({message.results.length} rows)
                        </summary>
                        <div className="mt-2">
                          <ResultsTable data={message.results} maxRows={20} />
                        </div>
                      </details>
                    )}
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
                  <div className="text-xs font-semibold text-gray-500 uppercase">AI</div>
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
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                // Submit on Enter, but allow Shift+Enter for new line
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              placeholder="Ask a question about markets, companies, or contracts... (Shift+Enter for new line)"
              rows={4}
              className="flex-1 bg-dark-700 border border-gold/30 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gold/60 focus:ring-2 focus:ring-gold/20 resize-y min-h-[100px] max-h-[300px]"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-gold/20 border border-gold/40 rounded-lg text-gold hover:bg-gold/30 hover:border-gold/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold self-end"
            >
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Suggested Questions */}
      <div>
        <h2 className="text-lg font-semibold text-gold mb-4">Suggested Questions</h2>
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
    </div>
  )
}
