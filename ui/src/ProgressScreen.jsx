import { useEffect, useState } from 'react'
import { apiFetch } from './api.js'

const STAGES = [
  { key: 'scanning', label: 'Scanning' },
  { key: 'searching', label: 'Searching' },
  { key: 'pointing', label: 'Pointing' },
  { key: 'on_target', label: 'On Target' },
]

const POLL_MS = 250

const SUCCESS_MESSAGE = 'The object has been located.'
const FAILURE_MESSAGE = "Couldn't find the object. Try moving the scanner somewhere else."

function stageText(search) {
  switch (search.stage) {
    case 'scanning':
      return `Scanning… ${search.photos_taken} of ${search.total_photos}`
    case 'searching':
      return `Checking photos… ${search.photos_checked} of ${search.total_photos}`
    case 'pointing':
      return `Pointing… ${search.distance_px} px away`
    case 'on_target':
      return 'Search completed.'
    case 'failed':
      return FAILURE_MESSAGE
    default:
      return ''
  }
}

function stepState(index, search) {
  if (search.stage === 'on_target') return 'done'
  const failed = search.stage === 'failed'
  const current = STAGES.findIndex((stage) => stage.key === (failed ? search.failed_at : search.stage))
  if (index < current) return 'done'
  if (index === current) return failed ? 'failed' : 'active'
  return 'pending'
}

function ProgressScreen({ initialSearch, onDone }) {
  const [search, setSearch] = useState(initialSearch)
  const [error, setError] = useState('')
  const finished = search.stage === 'on_target' || search.stage === 'failed'
  const spokenResult =
    search.stage === 'on_target' ? SUCCESS_MESSAGE : search.stage === 'failed' ? FAILURE_MESSAGE : null

  useEffect(() => {
    if (!spokenResult || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(spokenResult))
    return () => window.speechSynthesis.cancel()
  }, [spokenResult])

  useEffect(() => {
    if (finished) return
    const timer = setInterval(async () => {
      try {
        const response = await apiFetch(`/api/search/${initialSearch.id}`)
        if (!response.ok) throw new Error(`server returned ${response.status}`)
        setSearch(await response.json())
        setError('')
      } catch (err) {
        setError(`Lost contact with the server (${err.message}).`)
      }
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [initialSearch.id, finished])

  return (
    <main className="screen">
      <p className="query-echo">Looking for: {search.query}</p>

      <ol className="progress-bar">
        {STAGES.map((stage, index) => (
          <li key={stage.key} className={`progress-step ${stepState(index, search)}`}>
            <span className="progress-segment" />
            <span className="progress-label">{stage.label}</span>
          </li>
        ))}
      </ol>

      <h1 className="stage-text" role="status">{stageText(search)}</h1>

      {search.stage === 'scanning' && (
        <div className="thumbnail-strip" aria-hidden="true">
          {Array.from({ length: search.total_photos }, (_, index) => (
            <span key={index} className={`thumbnail${index < search.photos_taken ? ' taken' : ''}`} />
          ))}
        </div>
      )}

      {error && <p className="message">{error}</p>}

      {finished && (
        <button type="button" className="primary-button" onClick={onDone}>
          New search
        </button>
      )}
    </main>
  )
}

export default ProgressScreen
