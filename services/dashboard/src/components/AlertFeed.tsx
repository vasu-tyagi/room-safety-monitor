'use client'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity } from 'lucide-react'
import { AlertCard } from '@/components/AlertCard'
import { Switch } from '@/components/Switch'
import { useAlertFeed } from '@/hooks/useAlertFeed'
import { usePipelineProgress } from '@/context/PipelineProgressContext'
import { useAlerts } from '@/context/AlertContext'
import type { Incident } from '@/types/incident'
import { cn } from '@/lib/utils'

const WS_DOT: Record<string, string> = {
  connected:    'bg-emerald-500',
  connecting:   'bg-amber-500 animate-pulse',
  disconnected: 'bg-rose-500',
}

const WS_LABEL: Record<string, string> = {
  connected:    'Connected',
  connecting:   'Connecting',
  disconnected: 'Disconnected',
}

const LS_KEY = 'alertFeed.showPending'

export function AlertFeed({ initialIncidents }: { initialIncidents: Incident[] }) {
  const { alerts, mergeInitial, addOrUpdateAlert } = useAlerts()
  const [newIds, setNewIds] = useState<Set<string>>(new Set())

  const [showPending, setShowPending] = useState(false)
  const [initialized, setInitialized] = useState(false)

  // Read localStorage on mount (client-only, after hydration).
  useEffect(() => {
    const stored = localStorage.getItem(LS_KEY)
    if (stored !== null) setShowPending(stored === 'true')
    setInitialized(true)
  }, [])

  // Write on change, but only after initialization so the first mount
  // doesn't overwrite the stored value before it has been read.
  useEffect(() => {
    if (initialized) localStorage.setItem(LS_KEY, String(showPending))
  }, [showPending, initialized])

  // Merge SSR-prefetched incidents into the shared context on first mount.
  // The ref prevents re-merging if this component remounts during navigation.
  const mergedRef = useRef(false)
  useEffect(() => {
    if (mergedRef.current) return
    mergedRef.current = true
    if (initialIncidents.length > 0) mergeInitial(initialIncidents)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const onAlert = useCallback((incident: Incident) => {
    addOrUpdateAlert(incident)
    setNewIds((ids) => new Set(Array.from(ids).concat(incident.id)))
    setTimeout(() => {
      setNewIds((ids) => {
        const next = new Set(ids)
        next.delete(incident.id)
        return next
      })
    }, 800)
  }, [addOrUpdateAlert])

  const { dispatch: dispatchProgress } = usePipelineProgress()
  const status = useAlertFeed(onAlert, dispatchProgress)

  const visible = showPending
    ? alerts.filter((a) => a.operator_decision === null)
    : alerts

  return (
    <div>
      {/* Page header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-medium text-zinc-50">Live Feed</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Real-time incidents from monitored rooms
          </p>
        </div>
        <div className="flex items-center gap-2 mt-1.5">
          <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', WS_DOT[status])} />
          <span
            className="text-xs text-zinc-500"
            data-testid="ws-status"
          >
            {WS_LABEL[status]}
          </span>
        </div>
      </div>

      {/* Controls bar */}
      <div className="flex items-center justify-between mb-5">
        <Switch
          checked={showPending}
          onChange={setShowPending}
          label="Show only pending"
        />
        <span className="text-xs text-zinc-600">
          {visible.length} {visible.length === 1 ? 'incident' : 'incidents'}
        </span>
      </div>

      {/* Feed */}
      {visible.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 border border-zinc-800 rounded-lg bg-zinc-900/40">
          <Activity className="w-8 h-8 text-zinc-700 mb-3" />
          <p className="text-sm font-medium text-zinc-400">
            {showPending ? 'No pending incidents' : 'Pipeline monitoring active'}
          </p>
          <p className="text-xs text-zinc-600 mt-1">
            {showPending
              ? 'All incidents have been reviewed.'
              : 'New incidents will appear here in real time.'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {visible.map((a) => (
            <AlertCard key={a.id} incident={a} isNew={newIds.has(a.id)} />
          ))}
        </div>
      )}
    </div>
  )
}
