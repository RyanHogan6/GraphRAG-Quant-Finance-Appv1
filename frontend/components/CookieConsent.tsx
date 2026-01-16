'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

export default function CookieConsent() {
  const [showBanner, setShowBanner] = useState(false)

  useEffect(() => {
    // Check if user has already consented
    const consent = localStorage.getItem('cookie-consent')
    if (!consent) {
      // Show banner after a short delay
      setTimeout(() => setShowBanner(true), 1000)
    }
  }, [])

  const handleAccept = () => {
    localStorage.setItem('cookie-consent', 'accepted')
    setShowBanner(false)
  }

  const handleDecline = () => {
    localStorage.setItem('cookie-consent', 'declined')
    setShowBanner(false)
  }

  if (!showBanner) return null

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 md:p-6">
      <div className="max-w-4xl mx-auto bg-dark-800 border-2 border-gold/40 rounded-lg shadow-2xl p-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gold mb-2">🍪 Cookie Notice</h3>
            <p className="text-sm text-gray-300 leading-relaxed">
              We use essential cookies to provide core functionality and analytics cookies to
              understand how you use our site. By clicking "Accept", you consent to our use of cookies.{' '}
              <Link href="/privacy" className="text-gold hover:underline">
                Learn more
              </Link>
            </p>
          </div>

          <div className="flex gap-3 shrink-0">
            <button
              onClick={handleDecline}
              className="px-4 py-2 bg-dark-700 border border-gray-600 rounded-lg text-gray-300 hover:bg-dark-600 hover:border-gray-500 transition-all text-sm font-semibold"
            >
              Decline
            </button>
            <button
              onClick={handleAccept}
              className="px-6 py-2 bg-gold/20 border border-gold/40 rounded-lg text-gold hover:bg-gold/30 hover:border-gold/60 transition-all text-sm font-semibold"
            >
              Accept
            </button>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gold/10">
          <p className="text-xs text-gray-500">
            Essential cookies are always active. Analytics cookies help us improve the site.
            We do not sell your personal information.
          </p>
        </div>
      </div>
    </div>
  )
}
