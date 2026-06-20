'use client'
import { useEffect, useState } from 'react'

function format(dateStr: string): string {
  const secs = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000)
  if (secs < 60) return 'just now'
  if (secs < 3600) return `${Math.floor(secs / 60)} minutes ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)} hours ago`
  return `${Math.floor(secs / 86400)} days ago`
}

export function RelativeTime({ timestamp }: { timestamp: string }) {
  const [label, setLabel] = useState(format(timestamp))

  useEffect(() => {
    const id = setInterval(() => setLabel(format(timestamp)), 30_000)
    return () => clearInterval(id)
  }, [timestamp])

  return (
    <time
      dateTime={timestamp}
      title={new Date(timestamp).toLocaleString()}
      className="text-xs text-zinc-500"
    >
      {label}
    </time>
  )
}
