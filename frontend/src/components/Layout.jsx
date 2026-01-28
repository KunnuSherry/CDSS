import { NavLink, Link } from 'react-router-dom'
import { SearchIcon } from './Icons.jsx'
import './Layout.css'

export function Layout({ children }) {
  return (
    <div className="apple-app">
      <header className="apple-header">
        <div className="apple-header-inner">
          <Link to="/" className="apple-logo">
            CDSS
          </Link>
          <nav className="apple-nav">
            <NavLink to="/" end className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
              Platform
            </NavLink>
            <NavLink to="/admin" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
              Admin
            </NavLink>
            <NavLink to="/doctor" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
              Search
            </NavLink>
            <a href="#support" className="nav-link">Support</a>
          </nav>
          <div className="apple-header-actions">
            <Link to="/doctor" aria-label="Search">
              <SearchIcon size={18} />
            </Link>
          </div>
        </div>
      </header>

      <div className="apple-banner">
        <span>
          Study clinical guidelines. Retrieval-first, extractive outputs. For reference only — not a medical decision system.
        </span>
        <Link to="/doctor">
          Search guidelines
          <span className="apple-banner-chevron">›</span>
        </Link>
      </div>

      <main className="apple-main">
        {children}
      </main>

      <footer id="support" className="apple-footer">
        <div className="apple-container">
          <p className="apple-muted" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
            For study and reference only. Retrieval-first, extractive outputs. Not a medical decision system.
          </p>
        </div>
      </footer>
    </div>
  )
}
