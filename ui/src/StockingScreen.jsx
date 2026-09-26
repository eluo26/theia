import { useState } from 'react'
import { apiFetch } from './api.js'

const FIELDS = [
  { key: 'drug_name', label: 'Drug name', required: true },
  { key: 'strength', label: 'Strength', required: true, placeholder: '10 mg' },
  { key: 'ndc', label: 'NDC', placeholder: '99001-0110-30' },
  { key: 'lot', label: 'Lot' },
  { key: 'expiration_date', label: 'Expiration date', type: 'date' },
  { key: 'box_color', label: 'Box color' },
]

const EMPTY_FORM = Object.fromEntries(FIELDS.map((field) => [field.key, '']))

function CameraPreview() {
  // A new URL per connection stops the browser from showing a cached frame from an old stream.
  const [attempt, setAttempt] = useState(() => Date.now())
  const [status, setStatus] = useState('connecting') // connecting | live | offline

  function retry() {
    setStatus('connecting')
    setAttempt(Date.now())
  }

  return (
    <div className={`camera-preview ${status}`}>
      {status !== 'offline' && (
        <img
          key={attempt}
          src={`/api/camera/preview?attempt=${attempt}`}
          alt="Live camera feed"
          className={status === 'live' ? '' : 'hidden'}
          onLoad={() => setStatus('live')}
          onError={() => setStatus('offline')}
        />
      )}
      {status === 'offline' && (
        <div className="camera-status">
          <p className="camera-error">Camera not connected</p>
          <p>Connect the phone camera to see the live feed.</p>
          <button type="button" className="link-button" onClick={retry}>
            Try again
          </button>
        </div>
      )}
    </div>
  )
}

function StockingScreen() {
  const [phase, setPhase] = useState('idle') // idle | scanning | review | saving | saved
  const [scan, setScan] = useState(null)
  const [form, setForm] = useState({ ...EMPTY_FORM, count: 1, shelf: '', bin: '' })
  const [saved, setSaved] = useState(null)
  const [error, setError] = useState('')

  async function runScan(body) {
    setPhase('scanning')
    setError('')
    setScan(null)
    try {
      const response = await apiFetch('/api/stock/scan', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) {
        setScan(data.image_url ? { image_url: data.image_url } : null)
        throw new Error(data.error || `server returned ${response.status}`)
      }
      setScan(data)
      setForm({
        ...EMPTY_FORM,
        ...Object.fromEntries(FIELDS.map((field) => [field.key, data.fields[field.key] ?? ''])),
        count: 1,
        shelf: data.suggested_location.shelf ?? '',
        bin: data.suggested_location.bin ?? '',
      })
      setPhase('review')
    } catch (err) {
      setError(err.message)
      setPhase('idle')
    }
  }

  function scanFromCamera() {
    runScan(new FormData())
  }

  function scanFromFile(event) {
    const file = event.target.files[0]
    if (!file) return
    const body = new FormData()
    body.append('photo', file)
    runScan(body)
    event.target.value = ''
  }

  function reset() {
    setPhase('idle')
    setScan(null)
    setSaved(null)
    setError('')
  }

  function update(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setPhase('saving')
    setError('')
    try {
      const response = await apiFetch('/api/stock/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || `server returned ${response.status}`)
      setSaved(data)
      setPhase('saved')
    } catch (err) {
      setError(err.message)
      setPhase('review')
    }
  }

  return (
    <main className="screen stock-screen">
      <h1>{phase === 'saved' ? 'Box Successfully Stocked' : 'Stock a sample box'}</h1>

      {(phase === 'idle' || phase === 'scanning') && (
        <section className="stock-start">
          <p className="stock-hint">Hold the box up to the phone camera so the label fills the picture, then scan.</p>
          <CameraPreview />
          <button type="button" className="primary-button" onClick={scanFromCamera} disabled={phase === 'scanning'}>
            {phase === 'scanning' ? 'Reading label…' : 'Scan box'}
          </button>
          <label className="file-link">
            or choose a photo
            <input type="file" accept="image/jpeg,image/png" onChange={scanFromFile} disabled={phase === 'scanning'} />
          </label>
          {error && <p className="scan-error" role="alert">{error}</p>}
          {scan?.image_url && error && <img className="scan-photo small" src={scan.image_url} alt="Last scan" />}
        </section>
      )}

      {(phase === 'review' || phase === 'saving') && scan && (
        <form className="stock-review" onSubmit={handleSubmit}>
          <div className="scan-column">
            <img className="scan-photo" src={scan.image_url} alt="Scanned box label" />
            <button type="button" className="secondary-button" onClick={reset}>
              Retake
            </button>
          </div>

          <div className="fields-column">
            {scan.warnings.length > 0 && (
              <ul className="warnings">
                {scan.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}

            <div className="field-grid">
              {FIELDS.map((field) => (
                <label key={field.key}>
                  {field.label}
                  <input
                    type={field.type || 'text'}
                    value={form[field.key]}
                    required={field.required}
                    placeholder={field.placeholder}
                    onChange={(event) => update(field.key, event.target.value)}
                  />
                </label>
              ))}
              <label>
                Boxes
                <input type="number" min="1" value={form.count} onChange={(event) => update('count', event.target.value)} />
              </label>
              <label>
                Shelf
                <input type="text" value={form.shelf} onChange={(event) => update('shelf', event.target.value)} />
              </label>
              <label>
                Bin
                <input type="text" value={form.bin} onChange={(event) => update('bin', event.target.value)} />
              </label>
            </div>

            {error && <p className="scan-error" role="alert">{error}</p>}
            <button type="submit" className="primary-button" disabled={phase === 'saving'}>
              {phase === 'saving' ? 'Saving…' : 'Add to catalog'}
            </button>
          </div>
        </form>
      )}

      {phase === 'saved' && saved && (
        <section className="stock-start">
          <p className="stock-done" role="status">
            Added {saved.drug_name} {saved.strength} (lot {saved.lot || 'n/a'}). Now {saved.count} in stock
            {saved.shelf ? ` on shelf ${saved.shelf}, bin ${saved.bin}` : ''}.
          </p>
          <button type="button" className="primary-button" onClick={reset}>
            Scan another box
          </button>
        </section>
      )}
    </main>
  )
}

export default StockingScreen
