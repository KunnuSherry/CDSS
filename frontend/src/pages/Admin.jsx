import { useEffect, useMemo, useRef, useState } from 'react'
import { api, setToken } from '../lib/api.js'

function Dropzone({ onFile, disabled }) {
  const [isOver, setIsOver] = useState(false)
  const inputRef = useRef(null)

  return (
    <div
      className={`apple-dropzone ${isOver ? 'drag-over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setIsOver(true)
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setIsOver(false)
        if (disabled) return
        const f = e.dataTransfer.files?.[0]
        if (f?.name?.toLowerCase().endsWith('.pdf')) onFile(f)
      }}
    >
      <p className="apple-muted">Drag and drop a PDF here, or choose a file.</p>
      <div style={{ marginTop: 'var(--space-4)' }}>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          disabled={disabled}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
          }}
          style={{ display: 'none' }}
        />
        <button
          type="button"
          className="apple-btn apple-btn-secondary"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          Choose file
        </button>
      </div>
    </div>
  )
}

export function Admin() {
  const [email, setEmail] = useState('admin@example.com')
  const [password, setPassword] = useState('adminadmin')
  const [token, setTok] = useState(localStorage.getItem('token') || '')
  const [error, setError] = useState('')
  const [docs, setDocs] = useState([])
  const [busy, setBusy] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')

  const isAuthed = useMemo(() => Boolean(token), [token])

  async function refresh() {
    const data = await api.listDocuments()
    setDocs(data)
  }

  useEffect(() => {
    if (!isAuthed) return
    refresh().catch((e) => setError(e.message))
  }, [isAuthed])

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

  async function doSignupAdmin() {
    setError('')
    setBusy(true)
    try {
      await api.signup(email, password, 'admin')
      await doLogin()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleUpload(file) {
    setError('')
    setUploadMsg('')
    setBusy(true)
    try {
      const res = await api.uploadPdf(file)
      setUploadMsg(`Uploaded ${res.pdf_name}: ${res.chunks} chunks, ${res.images} images.`)
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
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
            <h2>Admin</h2>
            <p className="apple-muted">Sign in or create an admin account.</p>
            <div className="apple-form-group">
              <label htmlFor="admin-email">Email</label>
              <input
                id="admin-email"
                type="email"
                className="apple-input"
                placeholder="admin@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="apple-form-group">
              <label htmlFor="admin-password">Password</label>
              <input
                id="admin-password"
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
                onClick={doSignupAdmin}
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
          <h1 className="apple-headline apple-headline-lg" style={{ marginBottom: 0 }}>Admin</h1>
          <button type="button" className="apple-btn apple-btn-ghost" onClick={logout}>
            Sign out
          </button>
        </div>
        <p className="apple-muted" style={{ marginTop: 'var(--space-2)' }}>
          Upload guideline PDFs. Text is extracted from the PDF layer (no OCR).
        </p>
      </section>

      <section className="apple-section" style={{ paddingTop: 0 }}>
        <Dropzone onFile={handleUpload} disabled={busy} />
        {uploadMsg ? (
          <p className="apple-muted" style={{ marginTop: 'var(--space-4)', color: 'var(--color-link)' }}>
            {uploadMsg}
          </p>
        ) : null}
        {error ? <p className="apple-error" style={{ marginTop: 'var(--space-4)' }}>{error}</p> : null}
      </section>

      <section className="apple-section" style={{ background: 'var(--color-bg-subtle)', marginTop: 0 }}>
        <h2 className="apple-headline">Uploaded</h2>
        <p className="apple-muted" style={{ marginBottom: 'var(--space-5)' }}>
          {docs.length === 0 ? 'No PDFs yet. Upload one above.' : 'Your guideline documents.'}
        </p>

        {docs.length > 0 ? (
          <div className="apple-scroll-row">
            {docs.map((d) => (
              <div key={d.id} className="apple-scroll-item">
                <div className="apple-card-dark">
                  <div className="apple-muted">Document</div>
                  <h3 style={{ margin: 'var(--space-2) 0', fontSize: 'var(--font-size-lg)' }}>
                    {d.pdf_name}
                  </h3>
                  <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
                    Uploaded {d.uploaded_at}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  )
}
