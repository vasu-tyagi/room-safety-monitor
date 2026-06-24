'use client'
import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '@/lib/api'
import type { Incident } from '@/types/incident'
import type { PipelineProgressEvent } from '@/context/PipelineProgressContext'

export type FeedStatus = 'connecting' | 'connected' | 'disconnected'

export function useAlertFeed(
  onAlert: (incident: Incident) => void,
  onProgress?: (event: PipelineProgressEvent) => void,
): FeedStatus {
  const [status, setStatus] = useState<FeedStatus>('connecting')
  const wsRef = useRef<WebSocket | null>(null)
  const backoffRef = useRef(1000)
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const mountedRef = useRef(true)
  // Stable refs so callbacks never cause the WebSocket to restart.
  const onAlertRef = useRef(onAlert)
  const onProgressRef = useRef(onProgress)
  useEffect(() => { onAlertRef.current = onAlert })
  useEffect(() => { onProgressRef.current = onProgress })

  useEffect(() => {
    mountedRef.current = true

    function connect() {
      if (!mountedRef.current) return
      setStatus('connecting')

      const ws = new WebSocket(wsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return }
        backoffRef.current = 1000
        setStatus('connected')
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        }, 30_000)
      }

      ws.onmessage = (e: MessageEvent) => {
        console.log('[useAlertFeed] message received:', e.data)
        try {
          const msg = JSON.parse(e.data as string)
          if (msg.type === 'alert' && msg.incident) {
            onAlertRef.current(msg.incident as Incident)
          } else if (msg.type === 'pipeline_progress' && msg.layer) {
            onProgressRef.current?.({ layer: msg.layer, status: msg.status })
          }
        } catch {
          // malformed message — ignore
        }
      }

      ws.onclose = () => {
        // Guard against stale WS close events firing after a newer connection
        // has already replaced this one (e.g. React StrictMode double-mount).
        if (ws !== wsRef.current) return
        if (pingRef.current) clearInterval(pingRef.current)
        if (!mountedRef.current) return
        setStatus('disconnected')
        const delay = backoffRef.current
        backoffRef.current = Math.min(delay * 2, 30_000)
        setTimeout(connect, delay)
      }

      ws.onerror = () => ws.close()
    }

    connect()

    return () => {
      mountedRef.current = false
      wsRef.current?.close()
      if (pingRef.current) clearInterval(pingRef.current)
    }
  }, []) // only on mount/unmount — reconnect logic lives inside connect()

  return status
}
