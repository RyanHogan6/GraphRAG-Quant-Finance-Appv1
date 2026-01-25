import type { Metadata } from 'next'
import { Inter, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'
import Navigation from '@/components/Navigation'
import NeuralBackground from '@/components/NeuralBackground'
import Footer from '@/components/Footer'
import DisclaimerBanner from '@/components/DisclaimerBanner'
import CookieConsent from '@/components/CookieConsent'

const inter = Inter({
  weight: ['300', '400', '500', '600', '700'],
  subsets: ['latin'],
  variable: '--font-inter',
})

const ibmPlexMono = IBM_Plex_Mono({
  weight: ['400', '500', '600', '700'],
  subsets: ['latin'],
  variable: '--font-ibm-plex-mono',
})

export const metadata: Metadata = {
  title: 'KARGA - Intelligence Terminal for Financial Data',
  description: 'Natural language financial intelligence platform. Query 19 interconnected data sources: stocks, commodities, SEC filings, government contracts, prediction markets, and options flow. Built on ArangoDB graph database.',
  keywords: 'financial data, prediction markets, stock analysis, SEC filings, government contracts, options flow, commodities, natural language query, graph database',
  authors: [{ name: 'KARGA Analytics' }],
  creator: 'KARGA Analytics',
  publisher: 'KARGA Analytics',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://karga-ai.com',
    siteName: 'KARGA Intelligence Terminal',
    title: 'KARGA - Intelligence Terminal for Financial Data',
    description: 'Natural language financial intelligence platform. Query stocks, commodities, SEC filings, contracts, prediction markets, and options flow in plain English.',
    images: [
      {
        url: 'https://karga-ai.com/og-image.png',
        width: 1200,
        height: 630,
        alt: 'KARGA Intelligence Terminal - Financial Data Graph Database',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'KARGA - Intelligence Terminal for Financial Data',
    description: 'Ask questions in plain English. Get answers from 19 interconnected financial data sources.',
    images: ['https://karga-ai.com/og-image.png'],
    creator: '@karga_io',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    // Add these when you have them:
    // google: 'your-google-verification-code',
    // yandex: 'your-yandex-verification-code',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${ibmPlexMono.variable} font-sans flex flex-col min-h-screen`}>
        <NeuralBackground />
        <DisclaimerBanner />
        <Navigation />
        <main className="snap-y snap-proximity overflow-y-auto flex-1 scroll-smooth">
          {children}
        </main>
        <Footer />
        <CookieConsent />
      </body>
    </html>
  )
}
