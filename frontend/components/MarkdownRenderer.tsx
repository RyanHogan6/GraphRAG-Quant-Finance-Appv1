import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Tables
        table: ({ node, ...props }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border border-gold/30 rounded-lg" {...props} />
          </div>
        ),
        thead: ({ node, ...props }) => (
          <thead className="bg-gold/10 border-b border-gold/30" {...props} />
        ),
        tbody: ({ node, ...props }) => (
          <tbody className="divide-y divide-gold/20" {...props} />
        ),
        tr: ({ node, ...props }) => (
          <tr className="hover:bg-gold/5 transition-colors" {...props} />
        ),
        th: ({ node, ...props }) => (
          <th className="px-4 py-2 text-left text-xs font-semibold text-gold uppercase tracking-wider" {...props} />
        ),
        td: ({ node, ...props }) => (
          <td className="px-4 py-2 text-sm text-gray-300" {...props} />
        ),

        // Headers
        h1: ({ node, ...props }) => (
          <h1 className="text-2xl font-bold text-gold mt-4 mb-2" {...props} />
        ),
        h2: ({ node, ...props }) => (
          <h2 className="text-xl font-bold text-gold mt-3 mb-2" {...props} />
        ),
        h3: ({ node, ...props }) => (
          <h3 className="text-lg font-semibold text-gold mt-3 mb-2" {...props} />
        ),

        // Lists
        ul: ({ node, ...props }) => (
          <ul className="list-disc list-inside my-2 space-y-1" {...props} />
        ),
        ol: ({ node, ...props }) => (
          <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
        ),
        li: ({ node, ...props }) => (
          <li className="text-gray-300" {...props} />
        ),

        // Paragraphs
        p: ({ node, ...props }) => (
          <p className="my-2 text-gray-300 leading-relaxed" {...props} />
        ),

        // Strong/Bold
        strong: ({ node, ...props }) => (
          <strong className="font-semibold text-gold" {...props} />
        ),

        // Code blocks
        code: ({ node, inline, ...props }: any) => {
          return inline ? (
            <code className="bg-dark-700 px-1.5 py-0.5 rounded text-gold text-sm font-mono" {...props} />
          ) : (
            <code className="block bg-dark-700 p-3 rounded-lg text-gray-300 text-sm font-mono overflow-x-auto my-2" {...props} />
          )
        },

        // Links
        a: ({ node, ...props }) => (
          <a className="text-gold hover:text-gold/80 underline" target="_blank" rel="noopener noreferrer" {...props} />
        ),

        // Blockquotes
        blockquote: ({ node, ...props }) => (
          <blockquote className="border-l-4 border-gold/40 pl-4 my-2 text-gray-400 italic" {...props} />
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
