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
const BEST_GUESS_MESSAGE = "I'm not sure, but this is my best guess."
const FAILURE_MESSAGE = "Couldn't find the object. Try moving the scanner somewhere else."

function formatNumber(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

function formatBox(box) {
  if (!Array.isArray(box) || box.length !== 4) return '—'
  return `${box[0]}, ${box[1]} → ${box[2]}, ${box[3]}`
}

function formatPoint(point) {
  if (!Array.isArray(point) || point.length !== 2) return '—'
  return `${point[0]}, ${point[1]}`
}

function progressPercent(search) {
  if (typeof search.progress_pct === 'number') {
    return Math.max(0, Math.min(100, search.progress_pct))
  }
  if (search.stage === 'on_target') return 100
  if (search.stage === 'scanning' && search.total_photos) {
    return Math.round((100 * search.photos_taken) / search.total_photos)
  }
  if (search.stage === 'searching' && search.total_photos) {
    return Math.round((100 * search.photos_checked) / search.total_photos)
  }
  return 0
}

function stageText(search) {
  if (search.detail && search.stage !== 'failed' && search.stage !== 'on_target') {
    return search.detail
  }
  switch (search.stage) {
    case 'scanning':
      return `Scanning… ${search.photos_taken} of ${search.total_photos}`
    case 'searching':
      return search.photos_checked
        ? `Checking photos… ${search.photos_checked} of ${search.total_photos}`
        : 'Identifying the object…'
    case 'pointing':
      return `Pointing… ${search.distance_px} px away`
    case 'on_target':
      if (search.detection?.announcement) return search.detection.announcement
      if (search.best_guess) {
        return search.detection?.label
          ? `Best guess: ${search.detection.label}.`
          : 'Pointing at the best guess.'
      }
      return search.detection?.label ? `Found ${search.detection.label}.` : 'Search completed.'
    case 'failed':
      return search.error || FAILURE_MESSAGE
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

function DetectionResult({ search }) {
  const detection = search.detection
  if (!detection && !search.has_preview) return null
  return (
    <section className="detection-result">
      {search.has_preview && (
        <img
          className="detection-preview"
          src={`/api/search/${search.id}/preview`}
          alt={detection?.label ? `Located ${detection.label}` : 'Scan frame'}
        />
      )}
      {detection && (
        <dl className="detection-coords">
          <div>
            <dt>Bounding box</dt>
            <dd>{formatBox(detection.bbox_px)}</dd>
          </div>
          <div>
            <dt>Center</dt>
            <dd>{formatPoint(detection.center_px)}</dd>
          </div>
          <div>
            <dt>Image size</dt>
            <dd>
              {detection.image_width && detection.image_height
                ? `${detection.image_width} × ${detection.image_height}`
                : '—'}
            </dd>
          </div>
          <div>
            <dt>Aim</dt>
            <dd>
              az {formatNumber(detection.azimuth_deg)}°, el {formatNumber(detection.elevation_deg)}°
            </dd>
          </div>
          {typeof search.distance_px === 'number' && (
            <div>
              <dt>Offset from center</dt>
              <dd>{search.distance_px} px</dd>
            </div>
          )}
        </dl>
      )}
    </section>
  )
}

function ProgressScreen({ initialSearch, onDone }) {
  const [search, setSearch] = useState(initialSearch)
  const [error, setError] = useState('')
  const finished = search.stage === 'on_target' || search.stage === 'failed'
  const percent = progressPercent(search)
  const spokenResult =
    search.stage === 'on_target'
      ? search.detection?.announcement || (search.best_guess ? BEST_GUESS_MESSAGE : SUCCESS_MESSAGE)
      : search.stage === 'failed'
        ? FAILURE_MESSAGE
        : null

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

      {!finished && (
        <div
          className="search-meter"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-label="Search progress"
        >
          <div className="search-meter-track">
            <div className="search-meter-fill" style={{ width: `${percent}%` }} />
          </div>
          <span className="search-meter-label">{percent}%</span>
        </div>
      )}

      {search.stage === 'scanning' && (
        <div className="thumbnail-strip" aria-hidden="true">
          {Array.from({ length: search.total_photos }, (_, index) => (
            <span key={index} className={`thumbnail${index < search.photos_taken ? ' taken' : ''}`} />
          ))}
        </div>
      )}

      {search.stage === 'searching' && (
        <div className="thumbnail-strip" aria-hidden="true">
          {Array.from({ length: search.total_photos }, (_, index) => (
            <span key={index} className={`thumbnail${index < search.photos_checked ? ' checked' : ''}`} />
          ))}
        </div>
      )}

      {finished && <DetectionResult search={search} />}

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
