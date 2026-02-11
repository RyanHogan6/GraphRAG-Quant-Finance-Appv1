/**
 * SEC EDGAR URL helpers for linking to filings and company pages.
 */

const SEC_ARCHIVES_BASE = 'https://www.sec.gov/Archives/edgar/data'

/**
 * Build direct SEC filing URL from accession number (e.g. 0000320193-24-000123).
 * Links to the filing's index page on SEC.gov.
 */
export function secFilingDocumentUrl(accession: string | null | undefined): string | null {
  if (!accession || typeof accession !== 'string') return null
  const firstPart = accession.split('-')[0]
  if (!firstPart) return null
  const cik = parseInt(firstPart, 10)
  if (Number.isNaN(cik)) return null
  const accessionNoDashes = accession.replace(/-/g, '')
  return `${SEC_ARCHIVES_BASE}/${cik}/${accessionNoDashes}/`
}

/**
 * SEC company search page (all filings for a ticker).
 */
export function secCompanyUrl(ticker: string | null | undefined): string | null {
  if (!ticker || typeof ticker !== 'string') return null
  return `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=${encodeURIComponent(ticker)}`
}
