'use client'
import { useCallback, useEffect, useState } from 'react'
import { AlertCard } from '@/components/AlertCard'
import { useAlertFeed } from '@/hooks/useAlertFeed'
import { fetchIncidents } from '@/lib/api'
import type { Incident } from '@/types/incident'

const STATUS_CLASSES: Record<string, string> = {
  connected:    'bg-green-900 text-green-300',
  connecting:   'bg-yellow-900 text-yellow-300',
  disconnected: 'bg-red-900 text-red-300',
}

export default function HomePage() {
  const [alerts, setAlerts] = useState<Incident[]>([])

  useEffect(() => {
    fetchIncidents({ limit: '20' })
      .then(setAlerts)
      .catch(() => {/* backend may not be running in CI */})
  }, [])

  const onAlert = useCallback((incident: Incident) => {
    setAlerts((prev) => {
      const exists = prev.some((a) => a.id === incident.id)
      if (exists) return prev.map((a) => (a.id === incident.id ? incident : a))
      return [incident, ...prev]
    })
  }, [])

  const status = useAlertFeed(onAlert)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold">Live Alert Feed</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            <a href="/history" className="hover:text-gray-300">View history →</a>
          </p>
        </div>
        <span
          className={`text-xs px-2 py-1 rounded-full ${STATUS_CLASSES[status]}`}
          data-testid="ws-status"
        >
          {status}
        </span>
      </div>

      {alerts.length === 0 ? (
        <p className="text-gray-500 text-sm">No alerts yet.</p>
      ) : (
        <div className="space-y-3">
          {alerts.map((a) => (
            <AlertCard key={a.id} incident={a} />
          ))}
        </div>
      )}
    </div>
  )
}
