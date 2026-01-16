import type { Metadata } from 'next'
import { IBM_Plex_Mono } from 'next/font/google'
import './globals.css'
import Navigation from '@/components/Navigation'
import NeuralBackground from '@/components/NeuralBackground'

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
      <body className={`${ibmPlexMono.variable} font-mono`}>
        <NeuralBackground />
        <Navigation />
        <main className="snap-y snap-mandatory overflow-y-auto h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
