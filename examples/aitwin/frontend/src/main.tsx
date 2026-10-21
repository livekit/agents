import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// Avoid React StrictMode double-mount for LiveKit + AiTwin session setup.
createRoot(document.getElementById('root')!).render(<App />)
