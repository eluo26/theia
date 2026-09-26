import { useState } from 'react'

export default function AuthScreen({ onSignedIn }) {
  const [mode, setMode] = useState('login')
  const [role, setRole] = useState('patient')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const registering = mode === 'register'

  async function submit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const response = await fetch(`/api/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, role }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Unable to sign in. Please try again.')
      onSignedIn(data.user)
    } catch (err) {
      setError(err instanceof TypeError ? 'Could not reach Theia. Please try again.' : err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-screen">
      <section className="auth-card" aria-labelledby="auth-title">
        <p className="auth-brand">Theia</p>
        <h1 id="auth-title">{registering ? 'Create your account' : 'Welcome back'}</h1>
        <p className="auth-intro">{registering ? 'Choose how you will use Theia.' : 'Log in to find what you need.'}</p>
        <form onSubmit={submit}>
          <fieldset disabled={busy} className="auth-fields">
            <fieldset className="role-picker">
              <legend>Account type</legend>
              <label className={role === 'patient' ? 'role-option selected' : 'role-option'}>
                <input type="radio" name="role" value="patient" checked={role === 'patient'} onChange={() => setRole('patient')} />
                <span><strong>Patient</strong><small>Find an item with the scanner.</small></span>
              </label>
              <label className={role === 'staff' ? 'role-option selected' : 'role-option'}>
                <input type="radio" name="role" value="staff" checked={role === 'staff'} onChange={() => setRole('staff')} />
                <span><strong>Healthcare Professional</strong><small>Find items and scan boxes into the catalog.</small></span>
              </label>
            </fieldset>
            <label className="auth-field" htmlFor="email">Email
              <input id="email" type="email" autoComplete="username" required maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} />
            </label>
            <label className="auth-field" htmlFor="password">Password
              <input id="password" type="password" autoComplete={registering ? 'new-password' : 'current-password'} required minLength={8} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} aria-describedby={registering ? 'password-hint' : undefined} />
            </label>
            {registering && <p id="password-hint" className="auth-hint">Use at least 8 characters.</p>}
            {error && <p className="auth-error" role="alert">{error}</p>}
            <button className="primary-button auth-submit" type="submit">
              {busy ? 'Please wait…' : registering ? 'Create account' : 'Log in'}
            </button>
          </fieldset>
        </form>
        <p className="auth-switch">
          {registering ? 'Already have an account? ' : 'New to Theia? '}
          <button className="link-button" type="button" disabled={busy} onClick={() => {
            setMode(registering ? 'login' : 'register')
            setPassword('')
            setError('')
          }}>{registering ? 'Log in' : 'Create account'}</button>
        </p>
      </section>
    </main>
  )
}
