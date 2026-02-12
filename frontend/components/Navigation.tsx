'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import Link from 'next/link'

export default function Navigation() {
  const pathname = usePathname()
  const [activeSection, setActiveSection] = useState('query')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Smooth scroll to section
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    setMobileMenuOpen(false) // Close menu after navigation
  }

  // Track which section is currently in view
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['query', 'database', 'about']
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
    { id: 'database', label: 'Database' },
    { id: 'about', label: 'About' },
  ]

  return (
    <nav className="border-b border-gold/20 bg-dark-800 sticky top-0 z-50">
      <div className="container mx-auto px-4 md:px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo + Beta */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => scrollToSection('query')}
              className="text-lg md:text-xl font-bold text-gold font-mono hover:text-gold/80 transition-colors cursor-pointer"
            >
              KARGA
            </button>
            <span className="text-[10px] uppercase tracking-wider text-gold/70 border border-gold/30 px-2 py-0.5 rounded font-medium">Beta</span>
            <a
              href="mailto:karga.analytics@gmail.com?subject=KARGA%20Feedback"
              className="text-xs text-gray-400 hover:text-gold transition-colors"
            >
              Feedback
            </a>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex flex-1 items-center justify-center">
            <div className="flex space-x-1">
              {links.slice(0, 2).map((link) => (
                <button
                  key={link.id}
                  onClick={() => scrollToSection(link.id)}
                  className={`px-4 py-2 rounded-lg transition-all ${activeSection === link.id
                    ? 'bg-gold/20 text-gold border border-gold/40'
                    : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                    }`}
                >
                  {link.label}
                </button>
              ))}
            </div>
          </div>

          <div className="hidden md:flex items-center space-x-1">
            <Link
              href="/signals"
              className={`px-4 py-2 rounded-lg transition-all ${pathname === '/signals'
                ? 'bg-gold/20 text-gold border border-gold/40'
                : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                }`}
            >
              Signals
            </Link>
            <button
              onClick={() => scrollToSection('about')}
              className={`px-4 py-2 rounded-lg transition-all ${activeSection === 'about'
                ? 'bg-gold/20 text-gold border border-gold/40'
                : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                }`}
            >
              About
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-gray-400 hover:text-gold transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden mt-4 pb-4 space-y-2">
            {links.map((link) => (
              <button
                key={link.id}
                onClick={() => scrollToSection(link.id)}
                className={`w-full text-left px-4 py-3 rounded-lg transition-all ${activeSection === link.id
                  ? 'bg-gold/20 text-gold border border-gold/40'
                  : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                  }`}
              >
                {link.label}
              </button>
            ))}
            <Link
              href="/signals"
              onClick={() => setMobileMenuOpen(false)}
              className={`block w-full text-left px-4 py-3 rounded-lg transition-all ${pathname === '/signals'
                ? 'bg-gold/20 text-gold border border-gold/40'
                : 'text-gray-400 hover:text-gold hover:bg-gold/10'
                }`}
            >
              Signals
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}
