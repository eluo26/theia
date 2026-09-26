import { useEffect, useState } from 'react'
import InputScreen from './InputScreen.jsx'
import ProgressScreen from './ProgressScreen.jsx'
import AuthScreen from './AuthScreen.jsx'
import './App.css'

async function getSession() {
  const response = await fetch('/api/auth/me')
  if (!response.ok) throw new Error('Could not connect to Theia. Please try again.')
  return (await response.json()).user
}

function App() {
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

  function signedIn(account) {
    setSearch(null)
    setError('')
    setUser(account)
  }

  async function logout() {
    setLoggingOut(true)
    setError('')
    try {
      const response = await fetch('/api/auth/logout', { method: 'POST' })
      if (!response.ok) throw new Error('Could not log out. Please try again.')
      setUser(null)
      setSearch(null)
    } catch {
      setError('Could not log out. Please try again.')
    } finally {
      setLoggingOut(false)
    }
  }

  if (loading) return <main className="screen"><p role="status">Opening Theia…</p></main>
  if (!user && error) return <main className="screen"><p role="alert">{error}</p><button className="secondary-button" onClick={() => { setLoading(true); loadSession() }}>Try again</button></main>
  if (!user) return <AuthScreen onSignedIn={signedIn} />

  return (
    <div className="signed-in-app">
      <header className="account-header">
        <a className="app-brand" href="#/">Theia</a>
        <div className="account-controls">
          <span className="account-role">Patient</span>
          <button className="secondary-button" onClick={logout} disabled={loggingOut}>{loggingOut ? 'Logging out…' : 'Log out'}</button>
        </div>
      </header>
      {error && <p className="auth-error account-error" role="alert">{error}</p>}
      {search ? (
        <ProgressScreen key={search.id} initialSearch={search} onDone={() => setSearch(null)} />
      ) : <InputScreen onSearchStarted={setSearch} />}
    </div>
  )
}

export default App
