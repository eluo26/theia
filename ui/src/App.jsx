import { useState } from 'react'
import InputScreen from './InputScreen.jsx'
import ProgressScreen from './ProgressScreen.jsx'
import './App.css'

function App() {
  const [search, setSearch] = useState(null)

  return search ? (
    <ProgressScreen key={search.id} initialSearch={search} onDone={() => setSearch(null)} />
  ) : (
    <InputScreen onSearchStarted={setSearch} />
  )
}

export default App
