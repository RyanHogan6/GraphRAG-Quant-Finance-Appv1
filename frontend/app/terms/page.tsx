export default function TermsOfService() {
  return (
    <div className="min-h-screen bg-dark-900 text-gray-200 py-16 px-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gold mb-4">Terms of Service</h1>
        <p className="text-sm text-gray-400 mb-8">Last Updated: January 15, 2026</p>

        <div className="space-y-8 text-gray-300">
          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">1. Acceptance of Terms</h2>
            <p>
              By accessing and using KARGA ("Service"), you accept and agree to be
              bound by these Terms of Service. If you do not agree, do not use the Service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">2. Description of Service</h2>
            <p>
              KARGA is a financial data aggregation and visualization platform that:
            </p>
            <ul className="list-disc pl-6 space-y-2 mt-4">
              <li>Aggregates publicly available market data from various sources</li>
              <li>Displays prediction market information from Polymarket and Kalshi</li>
              <li>Provides AI-powered natural language querying of financial data</li>
              <li>Visualizes relationships between companies, markets, and economic data</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">3. No Financial Advice</h2>
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6 my-4">
              <p className="font-semibold text-red-300 mb-2">⚠️ IMPORTANT DISCLAIMER</p>
              <p className="text-sm">
                <strong>This Service does NOT provide financial, investment, legal, or tax advice.</strong>
                All information is provided for informational and educational purposes only.
                Market predictions, probabilities, and data visualizations are NOT guarantees of
                future performance. You should conduct your own research and consult with qualified
                financial advisors before making any investment decisions.
              </p>
            </div>
            <p className="mt-4">
              We are not a registered investment advisor, broker-dealer, or financial institution.
              Trading and investing carry significant risk of loss.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">4. Data Accuracy and Availability</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong>No Guarantees:</strong> We do not guarantee the accuracy, completeness,
                or timeliness of any data displayed
              </li>
              <li>
                <strong>Third-Party Data:</strong> Market data is sourced from third parties and
                may contain errors or delays
              </li>
              <li>
                <strong>Service Availability:</strong> We do not guarantee uninterrupted access.
                The Service may be temporarily unavailable for maintenance or technical issues
              </li>
              <li>
                <strong>API Limitations:</strong> Query rate limits may apply to prevent abuse
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">5. Acceptable Use</h2>
            <p className="mb-4">You agree NOT to:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Use the Service for any illegal purpose or in violation of applicable laws</li>
              <li>Attempt to circumvent security measures or access controls</li>
              <li>Use automated tools (bots, scrapers) without written permission</li>
              <li>Overload or disrupt the Service infrastructure</li>
              <li>Reverse engineer, decompile, or disassemble any part of the Service</li>
              <li>Resell, redistribute, or sublicense access to the Service</li>
              <li>Impersonate any person or entity</li>
              <li>Submit malicious code, viruses, or harmful content</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">6. Intellectual Property</h2>
            <h3 className="text-xl font-semibold text-gray-100 mt-4 mb-2">6.1 Our Rights</h3>
            <p>
              All content, features, and functionality of the Service (including but not limited to
              text, graphics, logos, code, and software) are owned by KARGA and protected
              by copyright, trademark, and other intellectual property laws.
            </p>

            <h3 className="text-xl font-semibold text-gray-100 mt-4 mb-2">6.2 Your Content</h3>
            <p>
              You retain ownership of queries you submit. By using the Service, you grant us a
              non-exclusive license to process your queries for the purpose of providing the Service.
            </p>

            <h3 className="text-xl font-semibold text-gray-100 mt-4 mb-2">6.3 Third-Party Data</h3>
            <p>
              Market data is owned by respective third parties (Polymarket, Kalshi, Yahoo Finance,
              Federal Reserve, etc.). Use of such data is subject to their respective terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">7. Limitation of Liability</h2>
            <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-6 my-4">
              <p className="font-semibold text-yellow-300 mb-2">IMPORTANT LEGAL NOTICE</p>
              <p className="text-sm">
                TO THE MAXIMUM EXTENT PERMITTED BY LAW, GRAPHRAG MARKETS SHALL NOT BE LIABLE FOR
                ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES,
                INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, OR GOODWILL, ARISING FROM:
              </p>
              <ul className="list-disc pl-6 mt-2 text-sm space-y-1">
                <li>Your use or inability to use the Service</li>
                <li>Errors, inaccuracies, or omissions in data or content</li>
                <li>Unauthorized access to your data</li>
                <li>Investment decisions made based on Service data</li>
                <li>Service interruptions or delays</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">8. Indemnification</h2>
            <p>
              You agree to indemnify and hold harmless KARGA from any claims, damages,
              losses, or expenses (including legal fees) arising from your use of the Service or
              violation of these Terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">9. Termination</h2>
            <p>
              We reserve the right to suspend or terminate your access to the Service at any time,
              with or without notice, for any reason including violation of these Terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">10. Changes to Terms</h2>
            <p>
              We may modify these Terms at any time. Continued use of the Service after changes
              constitutes acceptance of the new Terms. Material changes will be posted with an
              updated "Last Updated" date.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">11. Governing Law</h2>
            <p>
              These Terms shall be governed by and construed in accordance with the laws of
              [Your State/Country], without regard to conflict of law principles.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">12. Dispute Resolution</h2>
            <p>
              Any disputes arising from these Terms or the Service shall be resolved through
              binding arbitration in accordance with the rules of the American Arbitration Association.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">13. Severability</h2>
            <p>
              If any provision of these Terms is found to be unenforceable, the remaining provisions
              shall continue in full force and effect.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">14. Contact Information</h2>
            <p>
              For questions about these Terms, contact us at:
            </p>
            <p className="mt-2">
              <strong>Email:</strong> <a href="mailto:legal@graphragmarkets.com" className="text-gold hover:underline">legal@graphragmarkets.com</a>
            </p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t border-gold/20">
          <a href="/" className="text-gold hover:underline">← Back to Home</a>
        </div>
      </div>
    </div>
  )
}
