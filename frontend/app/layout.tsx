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
  title: 'GraphRAG Markets',
  description: 'AI-powered prediction markets and financial data explorer',
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
