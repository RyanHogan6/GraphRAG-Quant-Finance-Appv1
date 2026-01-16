'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'

export default function Navigation() {
  const pathname = usePathname()
  const [activeSection, setActiveSection] = useState('query')

  // Smooth scroll to section
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  // Track which section is currently in view
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['query', 'markets', 'database']
      const scrollPosition = window.scrollY + 100

      for (const sectionId of sections) {
        const element = document.getElementById(sectionId)
        if (element) {
          const { offsetTop, offsetHeight } = element
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(sectionId)
            break
          }
        }
      }
    }

    window.addEventListener('scroll', handleScroll)
    handleScroll() // Check initial position
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const links = [
    { id: 'query', label: 'Query' },
    { id: 'markets', label: 'Markets' },
    { id: 'database', label: 'Database' },
  ]

  return (
    <nav className="border-b border-gold/20 bg-dark-800 sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-gold">GraphRAG</h1>
          </div>

          <div className="flex space-x-1">
            {links.map((link) => (
              <button
                key={link.id}
                onClick={() => scrollToSection(link.id)}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeSection === link.id
                    ? 'bg-gold/20 text-gold border border-gold/40'
                    : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                }`}
              >
                {link.label}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">Status: <span className="text-green-400">Online</span></div>
          </div>
        </div>
      </div>
    </nav>
  )
}
