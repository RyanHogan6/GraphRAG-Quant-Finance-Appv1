export default function PrivacyPolicy() {
  return (
    <div className="min-h-screen bg-dark-900 text-gray-200 py-16 px-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gold mb-4">Privacy Policy</h1>
        <p className="text-sm text-gray-400 mb-8">Last Updated: January 15, 2026</p>

        <div className="space-y-8 text-gray-300">
          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">1. Introduction</h2>
            <p>
              KARGA ("we," "our," or "us") operates a financial data aggregation and
              prediction market visualization platform. This Privacy Policy explains how we collect,
              use, and protect your information when you use our website.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">2. Information We Collect</h2>

            <h3 className="text-xl font-semibold text-gray-100 mt-4 mb-2">2.1 Information You Provide</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Queries:</strong> Natural language questions you submit to our AI-powered search</li>
              <li><strong>Interactions:</strong> Markets you view, filters you apply, data you browse</li>
            </ul>

            <h3 className="text-xl font-semibold text-gray-100 mt-4 mb-2">2.2 Automatically Collected Information</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Log Data:</strong> IP address, browser type, device information, timestamps</li>
              <li><strong>Usage Data:</strong> Pages visited, time spent, click patterns</li>
              <li><strong>Cookies:</strong> Session cookies for functionality, analytics cookies (with consent)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">3. How We Use Your Information</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Provide Services:</strong> Process queries, display market data, generate analytics</li>
              <li><strong>Improve Platform:</strong> Analyze usage patterns to enhance user experience</li>
              <li><strong>Security:</strong> Detect and prevent fraud, abuse, and security threats</li>
              <li><strong>Analytics:</strong> Understand how users interact with our platform</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">4. Third-Party Services</h2>
            <p className="mb-4">We use the following third-party services that may collect data:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>ArangoDB Cloud:</strong> Database hosting (Germany/EU)</li>
              <li><strong>OpenAI:</strong> AI query processing (queries are sent for analysis)</li>
              <li><strong>Perplexity AI:</strong> Web search integration for current events</li>
              <li><strong>Vercel:</strong> Frontend hosting and analytics</li>
              <li><strong>Railway:</strong> Backend API hosting</li>
            </ul>
            <p className="mt-4 text-sm text-gray-400">
              These services have their own privacy policies. We recommend reviewing them.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">5. Data Retention</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Query Logs:</strong> Retained for 90 days for service improvement</li>
              <li><strong>Analytics Data:</strong> Aggregated and anonymized after 30 days</li>
              <li><strong>Error Logs:</strong> Retained for 30 days for debugging</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">6. Your Rights</h2>
            <p className="mb-4">You have the right to:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Access:</strong> Request a copy of your data</li>
              <li><strong>Deletion:</strong> Request deletion of your data</li>
              <li><strong>Opt-Out:</strong> Decline analytics cookies (functionality cookies still required)</li>
              <li><strong>Portability:</strong> Request your data in a machine-readable format</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">7. Security</h2>
            <p>
              We implement industry-standard security measures including HTTPS encryption,
              secure API authentication, database access controls, and regular security audits.
              However, no method of transmission over the Internet is 100% secure.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">8. Children's Privacy</h2>
            <p>
              Our service is not intended for individuals under 18 years of age. We do not
              knowingly collect personal information from children.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">9. Changes to This Policy</h2>
            <p>
              We may update this Privacy Policy from time to time. Changes will be posted on
              this page with an updated "Last Updated" date.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-gold mb-4">10. Contact Us</h2>
            <p>
              For privacy-related questions or to exercise your rights, contact us at:
            </p>
            <p className="mt-2">
              <strong>Email:</strong> <a href="mailto:privacy@graphragmarkets.com" className="text-gold hover:underline">privacy@graphragmarkets.com</a>
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
