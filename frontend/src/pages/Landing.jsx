import { Link, useNavigate } from 'react-router-dom'
import { DocIcon, StethoscopeIcon, ChevronRightIcon } from '../components/Icons.jsx'

export function Landing() {
  const nav = useNavigate()

  return (
    <>
      <section className="apple-section">
        <div className="apple-container">
          <h1 className="apple-headline apple-headline-lg">Platform</h1>
          <p className="apple-muted" style={{ marginBottom: 0 }}>
            Medical guideline study and reference. Upload PDFs, search extractive excerpts.
          </p>

          <div className="apple-category-grid">
            <Link to="/admin" className="apple-category-item">
              <span className="apple-category-icon">
                <DocIcon size={24} />
              </span>
              Admin
            </Link>
            <Link to="/doctor" className="apple-category-item">
              <span className="apple-category-icon">
                <StethoscopeIcon size={24} />
              </span>
              Search
            </Link>
          </div>
        </div>
      </section>

      <section className="apple-section" style={{ background: 'var(--color-bg-subtle)' }}>
        <div className="apple-container">
          <div className="apple-cta-right">
            <h2 className="apple-headline">The best way to study the guidelines you need.</h2>
            <div className="apple-cta-links">
              <Link to="/admin">
                Connect as Admin
                <ChevronRightIcon />
              </Link>
              <Link to="/doctor">
                Search guidelines
                <ChevronRightIcon />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="apple-section">
        <div className="apple-container">
          <h2 className="apple-headline">Get started. Upload or search.</h2>
          <p className="apple-muted" style={{ marginBottom: 'var(--space-5)' }}>
            Take a look at what you can do right now.
          </p>

          <div className="apple-scroll-row">
            <div className="apple-scroll-item">
              <div
                className="apple-card-dark"
                style={{ cursor: 'pointer' }}
                onClick={() => nav('/admin')}
                onKeyDown={(e) => e.key === 'Enter' && nav('/admin')}
                role="button"
                tabIndex={0}
              >
                <div className="apple-muted">Admin</div>
                <h3 style={{ margin: 'var(--space-2) 0', fontSize: 'var(--font-size-xl)' }}>
                  Upload PDFs
                </h3>
                <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
                  Add guideline PDFs. Extract text and images for search.
                </p>
                <Link to="/admin" style={{ marginTop: 'var(--space-4)' }}>
                  Go to Admin →
                </Link>
              </div>
            </div>
            <div className="apple-scroll-item">
              <div
                className="apple-card-dark"
                style={{ cursor: 'pointer' }}
                onClick={() => nav('/doctor')}
                onKeyDown={(e) => e.key === 'Enter' && nav('/doctor')}
                role="button"
                tabIndex={0}
              >
                <div className="apple-muted">Search</div>
                <h3 style={{ margin: 'var(--space-2) 0', fontSize: 'var(--font-size-xl)' }}>
                  Search guidelines
                </h3>
                <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
                  Query NSTEMI, acute coronary syndrome, and more. Verbatim excerpts only.
                </p>
                <Link to="/doctor" style={{ marginTop: 'var(--space-4)' }}>
                  Search →
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

    </>
  )
}
