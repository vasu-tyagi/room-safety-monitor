'use client'
import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
} from 'react'

// Cascade order matches the LayerStatusBar pill order.
export const PIPELINE_CASCADE = ['L1', 'L2', 'gate', 'L3', 'L4', 'L5', 'L6'] as const

export interface PipelineState {
  layer: string | null
  status: 'idle' | 'processing' | 'complete'
}

const IDLE: PipelineState = { layer: null, status: 'idle' }

export interface PipelineProgressEvent {
  layer: string
  status: string
}

interface ContextValue {
  state: PipelineState
  dispatch: (event: PipelineProgressEvent) => void
}

export const PipelineProgressContext = createContext<ContextValue>({
  state: IDLE,
  dispatch: () => {},
})

export function PipelineProgressProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [state, setState] = useState<PipelineState>(IDLE)
  const highWaterRef = useRef(-1)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const dispatch = useCallback((event: PipelineProgressEvent) => {
    const idx = PIPELINE_CASCADE.indexOf(event.layer as typeof PIPELINE_CASCADE[number])
    if (idx === -1) return

    // Only advance the animation — never go backward.
    if (idx < highWaterRef.current) return
    highWaterRef.current = idx

    const newStatus =
      event.status === 'processing' ? 'processing'
      : event.status === 'complete' ? 'complete'
      : null
    if (!newStatus) return

    setState({ layer: event.layer, status: newStatus })

    // Reset to idle after 2 s of silence.
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setState(IDLE)
      highWaterRef.current = -1
    }, 2000)
  }, [])

  return (
    <PipelineProgressContext.Provider value={{ state, dispatch }}>
      {children}
    </PipelineProgressContext.Provider>
  )
}

export function usePipelineProgress() {
  return useContext(PipelineProgressContext)
}
