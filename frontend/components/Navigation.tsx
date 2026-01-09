'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function Navigation() {
  const pathname = usePathname()

  const links = [
    { href: '/', label: '💬 Query' },
    { href: '/markets', label: '📊 Markets' },
    { href: '/database', label: '🗄️ Database' },
  ]

  return (
    <nav className="border-b border-gold/20 bg-dark-800">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="text-2xl font-bold text-gold">⚡</div>
            <h1 className="text-xl font-bold text-gold">GraphRAG</h1>
          </div>

          <div className="flex space-x-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`px-4 py-2 rounded-lg transition-all ${
                  pathname === link.href
                    ? 'bg-gold/20 text-gold border border-gold/40'
                    : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">Status: <span className="text-green-400">●</span> Online</div>
          </div>
        </div>
      </div>
    </nav>
  )
}
