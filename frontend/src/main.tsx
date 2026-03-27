import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import TrailPredictionPage from './components/TrailPredictionPage.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TrailPredictionPage />
  </StrictMode>,
)
