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

export type LayerStatus = 'idle' | 'processing' | 'complete'

export interface PipelineProgressEvent {
  layer: string
  status: string
}

const IDLE_STATES: Record<string, LayerStatus> = Object.fromEntries(
  PIPELINE_CASCADE.map((id) => [id, 'idle' as LayerStatus])
)

interface ContextValue {
  layerStates: Record<string, LayerStatus>
  dispatch: (event: PipelineProgressEvent) => void
  reset: () => void
}

export const PipelineProgressContext = createContext<ContextValue>({
  layerStates: { ...IDLE_STATES },
  dispatch: () => {},
  reset: () => {},
})

export function PipelineProgressProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [layerStates, setLayerStates] = useState<Record<string, LayerStatus>>(
    () => ({ ...IDLE_STATES })
  )
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const reset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setLayerStates({ ...IDLE_STATES })
  }, [])

  const dispatch = useCallback((event: PipelineProgressEvent) => {
    const validStatus: LayerStatus | null =
      event.status === 'processing' ? 'processing'
      : event.status === 'complete' ? 'complete'
      : null
    if (!validStatus) return
    if (!PIPELINE_CASCADE.includes(event.layer as typeof PIPELINE_CASCADE[number])) return

    if (event.layer === 'L1' && validStatus === 'complete') {
      // New pipeline run: reset all layers, then mark L1 complete.
      setLayerStates({ ...IDLE_STATES, L1: 'complete' })
    } else {
      setLayerStates((prev) => ({ ...prev, [event.layer]: validStatus }))
    }

    // Reset all to idle after 30 s of silence.
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setLayerStates({ ...IDLE_STATES })
    }, 30_000)
  }, [])

  return (
    <PipelineProgressContext.Provider value={{ layerStates, dispatch, reset }}>
      {children}
    </PipelineProgressContext.Provider>
  )
}

export function usePipelineProgress() {
  return useContext(PipelineProgressContext)
}
