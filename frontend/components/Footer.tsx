import Link from 'next/link'

export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t border-gold/20 bg-dark-900 mt-auto">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* About */}
          <div>
            <h3 className="text-gold font-semibold text-lg mb-4">KARGA</h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              AI-powered prediction markets and financial data explorer built on knowledge graphs.
            </p>
          </div>

          {/* Legal */}
          <div>
            <h4 className="text-gray-200 font-semibold mb-4">Legal</h4>
            <ul className="space-y-3 text-sm">
              <li>
                <Link href="/privacy" className="text-gray-400 hover:text-gold transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="text-gray-400 hover:text-gold transition-colors">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/disclaimer" className="text-gray-400 hover:text-gold transition-colors">
                  Disclaimer
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-gray-200 font-semibold mb-4">Resources</h4>
            <ul className="space-y-3 text-sm">
              <li>
                <a
                  href="https://github.com/anthropics/claude-code"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gold transition-colors"
                >
                  GitHub
                </a>
              </li>
              <li>
                <a
                  href="https://docs.arangodb.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gold transition-colors"
                >
                  ArangoDB Docs
                </a>
              </li>
              <li>
                <a
                  href="https://platform.openai.com/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gold transition-colors"
                >
                  OpenAI API
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="text-gray-200 font-semibold mb-4">Contact</h4>
            <ul className="space-y-3 text-sm">
              <li className="text-gray-400">
                <strong className="text-gray-300">General:</strong><br />
                <a href="mailto:karga.analytics@gmail.com" className="hover:text-gold transition-colors">
                  karga.analytics@gmail.com
                </a>
              </li>
              <li className="text-gray-400">
                <strong className="text-gray-300">Privacy:</strong><br />
                <a href="mailto:karga.analytics@gmail.com" className="hover:text-gold transition-colors">
                  karga.analytics@gmail.com
                </a>
              </li>
              <li className="text-gray-400">
                <strong className="text-gray-300">Legal:</strong><br />
                <a href="mailto:karga.analytics@gmail.com" className="hover:text-gold transition-colors">
                  karga.analytics@gmail.com
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gold/10 mt-10 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-gray-500 text-center md:text-left">
              © {currentYear} KARGA. All rights reserved.
            </p>

            <div className="flex flex-wrap justify-center gap-4 text-xs text-gray-500">
              <span>Built with ArangoDB</span>
              <span>•</span>
              <span>Powered by OpenAI</span>
              <span>•</span>
              <span>Hosted on Vercel</span>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-6 p-4 bg-yellow-900/10 border border-yellow-700/20 rounded-lg">
            <p className="text-xs text-yellow-200/80 text-center leading-relaxed">
              <strong>⚠️ Not Financial Advice:</strong> This platform is for informational purposes only.
              Market data may be delayed or inaccurate. Prediction market probabilities are not guarantees.
              Always conduct your own research and consult qualified professionals before making financial decisions.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
