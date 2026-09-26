import { useEffect, useState } from 'react'
import InputScreen from './InputScreen.jsx'
import ProgressScreen from './ProgressScreen.jsx'
import StockingScreen from './StockingScreen.jsx'
import AuthScreen from './AuthScreen.jsx'
import './App.css'

async function getSession() {
  const response = await fetch('/api/auth/me')
  if (!response.ok) throw new Error('Could not connect to Theia. Please try again.')
  return (await response.json()).user
}

function useHash() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash
}

function App() {
  const hash = useHash()
  const [search, setSearch] = useState(null)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [loggingOut, setLoggingOut] = useState(false)

  async function loadSession() {
    try {
      const account = await getSession()
      setError('')
      setUser(account)
      if (!account) setSearch(null)
    } catch {
      setError('Could not connect to Theia. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    getSession().then((account) => {
      if (!cancelled) setUser(account)
    }).catch(() => {
      if (!cancelled) setError('Could not connect to Theia. Please try again.')
    }).finally(() => {
      if (!cancelled) setLoading(false)
    })
    const onExpired = () => {
      setUser(null)
      setSearch(null)
      setError('')
    }
    window.addEventListener('session-expired', onExpired)
    return () => {
      cancelled = true
      window.removeEventListener('session-expired', onExpired)
    }
  }, [])

  useEffect(() => {
    if (user?.role === 'patient' && hash !== '#/') window.location.hash = '#/'
  }, [user, hash])

  function signedIn(account) {
    setSearch(null)
    setError('')
    setUser(account)
    window.location.hash = '#/'
  }

  async function logout() {
    setLoggingOut(true)
    setError('')
    try {
      const response = await fetch('/api/auth/logout', { method: 'POST' })
      if (!response.ok) throw new Error('Could not log out. Please try again.')
      setUser(null)
      setSearch(null)
      window.location.hash = '#/'
    } catch {
      setError('Could not log out. Please try again.')
    } finally {
      setLoggingOut(false)
    }
  }

  if (loading) return <main className="screen"><p role="status">Opening Theia…</p></main>
  if (!user && error) return <main className="screen"><p role="alert">{error}</p><button className="secondary-button" onClick={() => { setLoading(true); loadSession() }}>Try again</button></main>
  if (!user) return <AuthScreen onSignedIn={signedIn} />

  const stocking = user.role === 'staff' && hash === '#/stock'

  return (
    <div className="signed-in-app">
      <header className="account-header">
        <a className="app-brand" href="#/">Theia</a>
        {user.role === 'staff' && <nav aria-label="Scanners">
          <a href="#/" aria-current={!stocking ? 'page' : undefined}>Find An Item</a>
          <a href="#/stock" aria-current={stocking ? 'page' : undefined}>Stock A Box</a>
        </nav>}
        <div className="account-controls">
          <span className="account-role">{user.role === 'staff' ? 'Staff' : 'Patient'}</span>
          <button className="secondary-button" onClick={logout} disabled={loggingOut}>{loggingOut ? 'Logging out…' : 'Log out'}</button>
        </div>
      </header>
      {error && <p className="auth-error account-error" role="alert">{error}</p>}
      {stocking ? <StockingScreen /> : search ? (
        <ProgressScreen key={search.id} initialSearch={search} onDone={() => setSearch(null)} />
      ) : <InputScreen onSearchStarted={setSearch} />}
    </div>
  )
}

export default App
