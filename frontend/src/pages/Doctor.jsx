import { useMemo, useState } from 'react'
import { api, setToken } from '../lib/api.js'

const CDSS_FORM_DEFAULT = {
  age: '',
  heart_rate: '',
  systolic_bp: '',
  serum_creatinine: '',
  killip_class: 1,
  cardiac_arrest_at_admission: false,
  st_deviation_ecg: false,
  elevated_cardiac_enzymes: false,
}

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

  // CDSS: mode 'search' | 'cdss'; form state and results
  const [mode, setMode] = useState('search')
  const [cdssForm, setCdssForm] = useState({ ...CDSS_FORM_DEFAULT })
  const [cdssData, setCdssData] = useState(null)
  const [cdssError, setCdssError] = useState('')
  const [cdssBusy, setCdssBusy] = useState(false)
  const [showSupportingEvidence, setShowSupportingEvidence] = useState(false)

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

  async function doCdss() {
    setCdssError('')
    setCdssBusy(true)
    setCdssData(null)
    setShowSupportingEvidence(false)
    try {
      const body = {
        age: Number(cdssForm.age),
        heart_rate: Number(cdssForm.heart_rate),
        systolic_bp: Number(cdssForm.systolic_bp),
        serum_creatinine: Number(cdssForm.serum_creatinine),
        killip_class: Number(cdssForm.killip_class),
        cardiac_arrest_at_admission: Boolean(cdssForm.cardiac_arrest_at_admission),
        st_deviation_ecg: Boolean(cdssForm.st_deviation_ecg),
        elevated_cardiac_enzymes: Boolean(cdssForm.elevated_cardiac_enzymes),
      }
      const res = await api.doctorCdss(body)
      setCdssData(res)
    } catch (e) {
      setCdssError(e.message)
    } finally {
      setCdssBusy(false)
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
          <h1 className="apple-headline apple-headline-lg" style={{ marginBottom: 0 }}>Clinician</h1>
          <button type="button" className="apple-btn apple-btn-ghost" onClick={logout}>
            Sign out
          </button>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-2)', marginTop: 'var(--space-2)', flexWrap: 'wrap' }}>
          <button type="button" className={`apple-btn ${mode === 'search' ? 'apple-btn-primary' : 'apple-btn-ghost'}`} onClick={() => { setMode('search'); setData(null); setCdssData(null); setCdssError(''); setError(''); }}>Guideline search</button>
          <button type="button" className={`apple-btn ${mode === 'cdss' ? 'apple-btn-primary' : 'apple-btn-ghost'}`} onClick={() => { setMode('cdss'); setData(null); setCdssData(null); setCdssError(''); setError(''); }}>NSTEMI decision support</button>
        </div>
        <p className="apple-muted" style={{ marginTop: 'var(--space-2)' }}>
          {mode === 'search' ? 'Query guidelines. Results are extractive from uploaded PDFs.' : 'Structured patient input for NSTEMI. GRACE score, guideline excerpts, and cited case studies from PubMed. Advisory only.'}
        </p>
      </section>

      {mode === 'search' && (
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
      )}

      {mode === 'cdss' && (
      <section className="apple-section" style={{ paddingTop: 0 }}>
        <h2 className="apple-headline" style={{ marginBottom: 'var(--space-4)' }}>Patient input (NSTEMI)</h2>
        <p className="apple-muted" style={{ marginBottom: 'var(--space-4)' }}>
          All fields required. No free-text. Advisory output only; final decisions rest with the clinician.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <div className="apple-form-group">
            <label htmlFor="cdss-age">Age (years)</label>
            <input
              id="cdss-age"
              type="number"
              min={0}
              max={120}
              className="apple-input"
              value={cdssForm.age}
              onChange={(e) => setCdssForm((f) => ({ ...f, age: e.target.value }))}
            />
          </div>
          <div className="apple-form-group">
            <label htmlFor="cdss-hr">Heart rate (bpm)</label>
            <input
              id="cdss-hr"
              type="number"
              min={0}
              max={300}
              className="apple-input"
              value={cdssForm.heart_rate}
              onChange={(e) => setCdssForm((f) => ({ ...f, heart_rate: e.target.value }))}
            />
          </div>
          <div className="apple-form-group">
            <label htmlFor="cdss-sbp">Systolic BP (mmHg)</label>
            <input
              id="cdss-sbp"
              type="number"
              min={0}
              max={300}
              className="apple-input"
              value={cdssForm.systolic_bp}
              onChange={(e) => setCdssForm((f) => ({ ...f, systolic_bp: e.target.value }))}
            />
          </div>
          <div className="apple-form-group">
            <label htmlFor="cdss-cr">Serum creatinine (mg/dL)</label>
            <input
              id="cdss-cr"
              type="number"
              min={0}
              step={0.1}
              className="apple-input"
              value={cdssForm.serum_creatinine}
              onChange={(e) => setCdssForm((f) => ({ ...f, serum_creatinine: e.target.value }))}
            />
          </div>
          <div className="apple-form-group">
            <label htmlFor="cdss-killip">Killip class (I–IV)</label>
            <select
              id="cdss-killip"
              className="apple-input"
              value={cdssForm.killip_class}
              onChange={(e) => setCdssForm((f) => ({ ...f, killip_class: Number(e.target.value) }))}
            >
              <option value={1}>I (no CHF)</option>
              <option value={2}>II (rales/JVD)</option>
              <option value={3}>III (pulmonary edema)</option>
              <option value={4}>IV (cardiogenic shock)</option>
            </select>
          </div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <input
              type="checkbox"
              checked={cdssForm.cardiac_arrest_at_admission}
              onChange={(e) => setCdssForm((f) => ({ ...f, cardiac_arrest_at_admission: e.target.checked }))}
            />
            Cardiac arrest at admission
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <input
              type="checkbox"
              checked={cdssForm.st_deviation_ecg}
              onChange={(e) => setCdssForm((f) => ({ ...f, st_deviation_ecg: e.target.checked }))}
            />
            ST-segment deviation on ECG
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <input
              type="checkbox"
              checked={cdssForm.elevated_cardiac_enzymes}
              onChange={(e) => setCdssForm((f) => ({ ...f, elevated_cardiac_enzymes: e.target.checked }))}
            />
            Elevated cardiac enzymes
          </label>
        </div>
        <div className="apple-form-actions">
          <button
            type="button"
            className="apple-btn apple-btn-primary"
            disabled={cdssBusy || cdssForm.age === '' || cdssForm.heart_rate === '' || cdssForm.systolic_bp === '' || cdssForm.serum_creatinine === ''}
            onClick={doCdss}
          >
            {cdssBusy ? 'Loading…' : 'Get advisory'}
          </button>
        </div>
        {cdssError ? <p className="apple-error" style={{ marginTop: 'var(--space-4)' }}>{cdssError}</p> : null}
      </section>
      )}

      {mode === 'cdss' && cdssData && (
      <section className="apple-section" style={{ paddingTop: 0 }}>
        <div style={{ display: 'flex', gap: 'var(--space-5)', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
        <div className="apple-card" style={{ marginBottom: 'var(--space-5)', background: 'var(--color-bg-subtle)' }}>
          <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>Patient summary</h2>
          <p className="apple-muted" style={{ margin: 0 }}>
            Age {cdssData.patient_summary?.age} · HR {cdssData.patient_summary?.heart_rate} bpm · SBP {cdssData.patient_summary?.systolic_bp} mmHg ·
            Creatinine {cdssData.patient_summary?.serum_creatinine} mg/dL · Killip {cdssData.patient_summary?.killip_class} ·
            Cardiac arrest: {cdssData.patient_summary?.cardiac_arrest_at_admission ? 'Yes' : 'No'} ·
            ST deviation: {cdssData.patient_summary?.st_deviation_ecg ? 'Yes' : 'No'} ·
            Elevated enzymes: {cdssData.patient_summary?.elevated_cardiac_enzymes ? 'Yes' : 'No'}
          </p>
        </div>
        <div className="apple-card" style={{ marginBottom: 'var(--space-5)' }}>
          <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>GRACE score</h2>
          <p style={{ margin: 0, fontSize: 'var(--font-size-lg)' }}>
            <strong>Total: {cdssData.grace_score?.total_score}</strong> — {cdssData.grace_score?.risk_category} risk
          </p>
          <p className="apple-muted" style={{ marginTop: 'var(--space-2)', marginBottom: 0 }}>
            {cdssData.grace_score?.category_description}
          </p>
        </div>
        {cdssData.guideline_excerpts?.length > 0 && (
          <div style={{ marginBottom: 'var(--space-5)' }}>
            <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>Applicable guideline excerpts</h2>
            <p className="apple-muted" style={{ marginBottom: 'var(--space-4)' }}>
              At most 5 blocks (diagnostic, risk, management, harm). Verbatim from uploaded guidelines (RAG). Advisory only.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
              {cdssData.guideline_excerpts.map((ex) => (
                <div key={ex.chunk_id} className="apple-result-card">
                  {ex.section_title ? <p className="apple-muted" style={{ marginBottom: 'var(--space-2)' }}>{ex.section_title}</p> : null}
                  <p className="apple-muted">{ex.pdf_name}{ex.page ? ` — Page ${ex.page}` : ''}</p>
                  <pre style={{ whiteSpace: 'pre-wrap' }}>{ex.text}</pre>
                  <button
                    type="button"
                    className="apple-btn"
                    style={{
                      marginTop: 'var(--space-3)',
                      background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
                      color: 'white',
                      border: 'none',
                      boxShadow:
                        selectedChunkId === ex.chunk_id
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
                    onClick={() => handleSummarizeChunk(ex)}
                  >
                    Summarize with AI
                  </button>
                </div>
              ))}
            </div>
            {cdssData.supporting_evidence?.length > 0 && (
              <div style={{ marginTop: 'var(--space-4)' }}>
                <button
                  type="button"
                  className="apple-btn apple-btn-secondary"
                  onClick={() => setShowSupportingEvidence((v) => !v)}
                >
                  {showSupportingEvidence ? 'Hide supporting evidence (advanced)' : 'Show supporting evidence (advanced)'}
                </button>
                {showSupportingEvidence && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', marginTop: 'var(--space-4)' }}>
                    {cdssData.supporting_evidence.map((ex) => (
                      <div key={ex.chunk_id} className="apple-result-card">
                        {ex.section_title ? <p className="apple-muted" style={{ marginBottom: 'var(--space-2)' }}>{ex.section_title}</p> : null}
                        <p className="apple-muted">{ex.pdf_name}{ex.page ? ` — Page ${ex.page}` : ''}</p>
                        <pre style={{ whiteSpace: 'pre-wrap' }}>{ex.text}</pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {cdssData.case_studies?.length > 0 && (
          <div style={{ marginBottom: 'var(--space-5)' }}>
            <h2 className="apple-headline" style={{ marginBottom: 'var(--space-2)' }}>Relevant case studies (PubMed)</h2>
            <p className="apple-muted" style={{ marginBottom: 'var(--space-4)' }}>
              Cited case reports from PubMed. Extractive; not AI-generated.
            </p>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {cdssData.case_studies.map((cs, i) => (
                <li key={i} style={{ marginBottom: 'var(--space-3)' }}>
                  <div className="apple-card" style={{ padding: 'var(--space-3)' }}>
                    <strong>{cs.title}</strong>
                    <p className="apple-muted" style={{ margin: 'var(--space-1) 0 0', fontSize: 'var(--font-size-sm)' }}>
                      {cs.source}{cs.pmid ? ` · PMID ${cs.pmid}` : ''}
                    </p>
                    {cs.url ? (
                      <a href={cs.url} target="_blank" rel="noopener noreferrer" className="apple-btn apple-btn-ghost" style={{ marginTop: 'var(--space-2)' }}>
                        View on PubMed
                      </a>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="apple-card" style={{ background: 'var(--color-bg-subtle)', borderLeft: '4px solid var(--color-border)' }}>
          <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
            <strong>Disclaimer:</strong> {cdssData.disclaimer}
          </p>
        </div>
          </div>

          {cdssData.guideline_excerpts?.length && (selectedChunkId || aiExplanationLoading) ? (
            <div style={{ flex: 1, minWidth: 300 }}>
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
        </div>
      </section>
      )}

      {mode === 'search' && data ? (
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
