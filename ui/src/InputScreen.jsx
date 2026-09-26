import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './api.js'

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

function InputScreen({ onSearchStarted }) {
  const [query, setQuery] = useState('')
  const [listening, setListening] = useState(false)
  const [message, setMessage] = useState('')
  const recognitionRef = useRef(null)

  useEffect(() => () => recognitionRef.current?.abort(), [])

  function toggleListening() {
    if (listening) {
      recognitionRef.current?.stop()
      return
    }

    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = true

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results, (result) => result[0].transcript).join('')
      setQuery(transcript)
    }
    recognition.onerror = (event) => {
      setMessage(
        event.error === 'not-allowed'
          ? 'Microphone access was blocked. Allow it in the browser and try again.'
          : `Voice input failed: ${event.error}`,
      )
    }
    recognition.onend = () => setListening(false)

    recognitionRef.current = recognition
    setMessage('')
    setListening(true)
    recognition.start()
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const text = query.trim()
    if (!text) return
    recognitionRef.current?.abort()

    try {
      const response = await apiFetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
      })
      if (!response.ok) throw new Error(`server returned ${response.status}`)
      onSearchStarted(await response.json())
    } catch (error) {
      setMessage(`Could not reach the server (${error.message}).`)
    }
  }

  return (
    <main className="screen">
      <h1>What are you looking for?</h1>

      <button
        type="button"
        className={`mic-button${listening ? ' listening' : ''}`}
        onClick={toggleListening}
        disabled={!SpeechRecognition}
        aria-label={listening ? 'Stop listening' : 'Start voice input'}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z" />
          <path d="M19 11a7 7 0 0 1-14 0M12 18v3M8 21h8" fill="none" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
      <p className="mic-hint">
        {!SpeechRecognition
          ? 'Voice input needs Chrome or Edge. You can still type below.'
          : listening
            ? 'Listening… tap to stop'
            : 'Tap to speak'}
      </p>

      <form className="query-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Where's the starter pack?"
          aria-label="What are you looking for?"
        />
        <button type="submit" disabled={!query.trim()}>
          Find it
        </button>
      </form>

      <p className="message" role="status">{message}</p>

    </main>
  )
}

export default InputScreen
