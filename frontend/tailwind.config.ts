import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        gold: {
          400: '#fbbf24',
          500: '#d4af37',
          600: '#b8941f',
        },
        dark: {
          900: '#0a0a0a',
          800: '#121212',
          700: '#1a1a1a',
        },
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-ibm-plex-mono)', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}
export default config
