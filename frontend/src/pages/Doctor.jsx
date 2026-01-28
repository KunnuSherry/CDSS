import { useMemo, useState } from 'react'
import { api, setToken } from '../lib/api.js'

export function Doctor() {
  const [email, setEmail] = useState('doctor@example.com')
  const [password, setPassword] = useState('doctordoctor')
  const [token, setTok] = useState(localStorage.getItem('token') || '')
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [selectedChunkId, setSelectedChunkId] = useState(null)
  const [aiExplanation, setAiExplanation] = useState('')
  const [aiExplanationLoading, setAiExplanationLoading] = useState(false)
  const [aiStatus, setAiStatus] = useState(null) // null | 'ok' | 'rate_limit' | 'unavailable'
  const [aiCache, setAiCache] = useState({}) // chunk_id -> explanation string
  const [showAdvanced, setShowAdvanced] = useState(false)

  const isAuthed = useMemo(() => Boolean(token), [token])

  async function doLogin() {
    setError('')
    setBusy(true)
    try {
      const res = await api.login(email, password)
      setTok(res.access_token)
      setToken(res.access_token)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function doSignupDoctor() {
    setError('')
    setBusy(true)
    try {
      await api.signup(email, password, 'doctor')
      await doLogin()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function doSearch() {
    setError('')
    setBusy(true)
    setData(null)
    // Clear any prior AI explanation state when a new query runs.
    setSelectedChunkId(null)
    setAiExplanation('')
    setAiExplanationLoading(false)
    setAiStatus(null)
    try {
      const res = await api.doctorSearch(query.trim())
      setData(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleSummarizeChunk(chunk) {
    if (!chunk || !chunk.text) return

    setSelectedChunkId(chunk.chunk_id)
    setAiStatus(null)

    // Per-chunk explanation: first check cache to avoid duplicate calls and
    // reduce the chance of hitting Gemini rate limits.
    const cached = aiCache[chunk.chunk_id]
    if (cached) {
      setAiExplanation(cached)
      setAiStatus('ok')
      return
    }

    setAiExplanation('')
    setAiExplanationLoading(true)
    try {
      const res = await api.explainWithGemini(chunk.text)
      if (!res.ok) {
        if (res.reason === 'rate_limit') {
          setAiStatus('rate_limit')
          setAiExplanation('AI explanation temporarily unavailable (rate limit).')
        } else {
          setAiStatus('unavailable')
          setAiExplanation('AI explanation unavailable for this section.')
        }
        return
      }
      setAiStatus('ok')
      setAiExplanation(res.text)
      setAiCache((prev) => ({ ...prev, [chunk.chunk_id]: res.text }))
    } catch {
      setAiStatus('unavailable')
      setAiExplanation('AI explanation unavailable for this section.')
    } finally {
      setAiExplanationLoading(false)
    }
  }

  function logout() {
    setTok('')
    setToken('')
  }

  if (!isAuthed) {
    return (
      <div className="apple-container">
        <section className="apple-section">
          <div className="apple-login-panel">
            <h2>Search</h2>
            <p className="apple-muted">Sign in or create a doctor account to search guidelines.</p>
            <div className="apple-form-group">
              <label htmlFor="doctor-email">Email</label>
              <input
                id="doctor-email"
                type="email"
                className="apple-input"
                placeholder="doctor@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="apple-form-group">
              <label htmlFor="doctor-password">Password</label>
              <input
                id="doctor-password"
                type="password"
                className="apple-input"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div className="apple-form-actions">
              <button
                type="button"
                className="apple-btn apple-btn-primary"
                disabled={busy}
                onClick={doLogin}
              >
                Sign in
              </button>
              <button
                type="button"
                className="apple-btn apple-btn-secondary"
                disabled={busy}
                onClick={doSignupDoctor}
              >
                Create account
              </button>
            </div>
            {error ? <p className="apple-error">{error}</p> : null}
          </div>
        </section>
      </div>
    )
  }

  return (
    <div className="apple-container">
      <section className="apple-section">
        <div className="apple-row apple-row-between" style={{ alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
          <h1 className="apple-headline apple-headline-lg" style={{ marginBottom: 0 }}>Search</h1>
          <button type="button" className="apple-btn apple-btn-ghost" onClick={logout}>
            Sign out
          </button>
        </div>
        <p className="apple-muted" style={{ marginTop: 'var(--space-2)' }}>
          Query guidelines. Results are extractive from uploaded PDFs. If missing: “Not found in uploaded documents.”
        </p>
      </section>

      <section className="apple-section" style={{ paddingTop: 0 }}>
        <div className="apple-search-bar">
          <input
            type="search"
            className="apple-input"
            placeholder='e.g. NSTEMI, Acute coronary syndrome'
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && doSearch()}
          />
          <button
            type="button"
            className="apple-btn apple-btn-primary"
            disabled={busy || !query.trim()}
            onClick={doSearch}
          >
            Search
          </button>
        </div>
        {error ? <p className="apple-error" style={{ marginTop: 'var(--space-4)' }}>{error}</p> : null}
      </section>

      {data ? (
        <section className="apple-section apple-results-section" style={{ paddingTop: 0, display: 'flex', gap: 'var(--space-5)' }}>
          <div style={{ flex: 1 }}>
            {data.note ? (
              <div className="apple-card" style={{ marginBottom: 'var(--space-5)' }}>
                <p className="apple-muted" style={{ margin: 0 }}>{data.note}</p>
              </div>
            ) : null}

            {data.results?.length ? (
              <>
                <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>
                  Guideline Excerpts (Verbatim)
                  {data.source_pdf ? ` — Extracted from ${data.source_pdf}` : ''}
                </h2>
                <p className="apple-muted" style={{ marginBottom: 'var(--space-5)' }}>
                  Verbatim guideline text, cleaned for readability and grouped by page and section headings (when available).
                </p>
                {(() => {
                  const byPage = {}
                  for (const r of data.results) {
                    const page = r.page || 0
                    if (!byPage[page]) byPage[page] = []
                    byPage[page].push(r)
                  }
                  const pages = Object.keys(byPage)
                    .map((p) => Number(p))
                    .sort((a, b) => a - b)
                  return pages.map((page) => (
                    <div key={page} style={{ marginBottom: 'var(--space-6)' }}>
                      <h3 className="apple-headline" style={{ fontSize: 'var(--font-size-lg)', marginBottom: 'var(--space-2)' }}>
                        Page {page}
                      </h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                        {byPage[page].map((r) => (
                          <div key={r.chunk_id} className="apple-result-card">
                            <p className="apple-muted">
                              {r.heading ? `${r.heading} — ` : ''}
                              {r.pdf_name ? r.pdf_name : ''}
                              {r.page ? ` — Page ${r.page}` : ''}
                            </p>
                            <pre>{r.text}</pre>
                            <button
                              type="button"
                              className="apple-btn"
                              style={{
                                marginTop: 'var(--space-3)',
                                background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
                                color: 'white',
                                border: 'none',
                                boxShadow:
                                  selectedChunkId === r.chunk_id
                                    ? '0 0 0 1px rgba(129, 140, 248, 0.8), 0 12px 24px rgba(15, 23, 42, 0.35)'
                                    : '0 10px 20px rgba(15, 23, 42, 0.25)',
                                transition: 'transform 120ms ease, box-shadow 120ms ease, filter 120ms ease',
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'translateY(-1px)'
                                e.currentTarget.style.filter = 'brightness(1.03)'
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'translateY(0)'
                                e.currentTarget.style.filter = 'none'
                              }}
                              onClick={() => handleSummarizeChunk(r)}
                            >
                              Summarize with AI
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                })()}

                {data.advanced_results?.length ? (
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <button
                      type="button"
                      className="apple-btn apple-btn-secondary"
                      onClick={() => setShowAdvanced((v) => !v)}
                    >
                      {showAdvanced ? 'Hide additional guideline excerpts' : 'Show additional guideline excerpts (advanced)'}
                    </button>
                    {showAdvanced && (() => {
                      const byPage = {}
                      for (const r of data.advanced_results) {
                        const page = r.page || 0
                        if (!byPage[page]) byPage[page] = []
                        byPage[page].push(r)
                      }
                      const pages = Object.keys(byPage)
                        .map((p) => Number(p))
                        .sort((a, b) => a - b)
                      return pages.map((page) => (
                        <div key={`adv-${page}`} style={{ marginTop: 'var(--space-5)' }}>
                          <h3 className="apple-headline" style={{ fontSize: 'var(--font-size-md)', marginBottom: 'var(--space-2)' }}>
                            Additional excerpts — Page {page}
                          </h3>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                            {byPage[page].map((r) => (
                              <div key={r.chunk_id} className="apple-result-card">
                                <p className="apple-muted">
                                  {r.heading ? `${r.heading} — ` : ''}
                                  {r.pdf_name ? r.pdf_name : ''}
                                  {r.page ? ` — Page ${r.page}` : ''}
                                </p>
                                <pre>{r.text}</pre>
                                <button
                                  type="button"
                                  className="apple-btn"
                                  style={{
                                    marginTop: 'var(--space-3)',
                                    background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
                                    color: 'white',
                                    border: 'none',
                                    boxShadow:
                                      selectedChunkId === r.chunk_id
                                        ? '0 0 0 1px rgba(129, 140, 248, 0.8), 0 12px 24px rgba(15, 23, 42, 0.35)'
                                        : '0 10px 20px rgba(15, 23, 42, 0.25)',
                                    transition: 'transform 120ms ease, box-shadow 120ms ease, filter 120ms ease',
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateY(-1px)'
                                    e.currentTarget.style.filter = 'brightness(1.03)'
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateY(0)'
                                    e.currentTarget.style.filter = 'none'
                                  }}
                                  onClick={() => handleSummarizeChunk(r)}
                                >
                                  Summarize with AI
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))
                    })()}
                  </div>
                ) : null}
              </>
            ) : null}
          </div>

          {data.results?.length && (selectedChunkId || aiExplanationLoading) ? (
            <div style={{ flex: 1 }}>
              <div className="apple-card">
                <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>AI Explanation (Plain Language)</h2>
                <p className="apple-muted" style={{ marginBottom: 'var(--space-4)', fontSize: 'var(--font-size-sm)' }}>
                  AI-assisted explanation based only on the selected guideline excerpt.
                </p>
                {aiExplanationLoading ? (
                  <p>Loading AI explanation...</p>
                ) : (
                  <pre style={{ whiteSpace: 'pre-wrap' }}>{aiExplanation}</pre>
                )}
                {aiStatus === 'rate_limit' ? (
                  <p className="apple-muted" style={{ marginTop: 'var(--space-2)', fontSize: 'var(--font-size-sm)' }}>
                    AI explanation temporarily unavailable (rate limit).
                  </p>
                ) : null}
              </div>
            </div>
          ) : null}
        </section>
      ) : null}

      {data?.coverage_note ? (
        <section className="apple-section" style={{ paddingTop: 0 }}>
          <div className="apple-card" style={{ marginTop: 'var(--space-6)', background: 'var(--color-bg-subtle)' }}>
            <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
              <strong>Coverage note:</strong> {data.coverage_note}
            </p>
          </div>
        </section>
      ) : null}

      {data?.images?.length ? (
        <section className="apple-section" style={{ paddingTop: 0 }}>
          <div style={{ marginTop: 'var(--space-8)' }}>
            <h2 className="apple-headline">Related images</h2>
            <p className="apple-muted" style={{ marginBottom: 'var(--space-4)' }}>
              From the same pages as the excerpts.
            </p>
            <div className="apple-image-grid">
              {data.images.map((im) => (
                <div key={`${im.page}-${im.image_index}`} className="apple-image-card">
                  <img src={im.url} alt={`Page ${im.page}`} />
                  <p className="apple-muted">Page {im.page}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : null}
    </div>
  )
}
