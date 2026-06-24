'use client'
import { createContext, useCallback, useContext, useState } from 'react'
import type { Incident } from '@/types/incident'

interface AlertContextValue {
  alerts: Incident[]
  mergeInitial: (incidents: Incident[]) => void
  addOrUpdateAlert: (incident: Incident) => void
  updateDecision: (id: string, decision: string) => void
}

export const AlertContext = createContext<AlertContextValue>({
  alerts: [],
  mergeInitial: () => {},
  addOrUpdateAlert: () => {},
  updateDecision: () => {},
})

export function AlertProvider({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<Incident[]>([])

  const mergeInitial = useCallback((incoming: Incident[]) => {
    setAlerts((prev) => {
      const existingIds = new Set(prev.map((a) => a.id))
      const novel = incoming.filter((a) => !existingIds.has(a.id))
      if (novel.length === 0) return prev
      return [...prev, ...novel]
    })
  }, [])

  const addOrUpdateAlert = useCallback((incident: Incident) => {
    setAlerts((prev) => {
      const exists = prev.some((a) => a.id === incident.id)
      if (exists) return prev.map((a) => (a.id === incident.id ? incident : a))
      return [incident, ...prev]
    })
  }, [])

  const updateDecision = useCallback((id: string, decision: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, operator_decision: decision } : a))
    )
  }, [])

  return (
    <AlertContext.Provider value={{ alerts, mergeInitial, addOrUpdateAlert, updateDecision }}>
      {children}
    </AlertContext.Provider>
  )
}

export function useAlerts() {
  return useContext(AlertContext)
}
