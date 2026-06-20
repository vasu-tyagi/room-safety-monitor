'use client'
import { useEffect, useState } from 'react'
import { fetchMetrics, type MetricsData, type OperatorDecisions } from '@/lib/api'

function StatCard({
  label,
  value,
}: {
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
      <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">{label}</p>
      <div className="text-2xl font-mono font-medium text-zinc-50">{value}</div>
    </div>
  )
}

function NoData() {
  return <span className="text-zinc-600 text-base">No data yet</span>
}

function SeverityBar({ data }: { data: Record<string, number> | null }) {
  if (!data || Object.keys(data).length === 0) return <NoData />
  const entries = Object.entries(data)
  const max = Math.max(...entries.map(([, v]) => v), 1)
  const SEV_COLOR: Record<string, string> = {
    high: 'bg-rose-500', medium: 'bg-amber-500', low: 'bg-sky-500',
  }
  return (
    <div className="space-y-2">
      {entries.map(([sev, count]) => (
        <div key={sev} className="flex items-center gap-3">
          <span className="text-xs text-zinc-400 uppercase w-14 shrink-0">{sev}</span>
          <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${SEV_COLOR[sev] ?? 'bg-zinc-500'}`}
              style={{ width: `${Math.round((count / max) * 100)}%` }}
            />
          </div>
          <span className="text-xs font-mono text-zinc-400 w-6 text-right">{count}</span>
        </div>
      ))}
    </div>
  )
}

function DecisionBar({ data }: { data: OperatorDecisions | null | undefined }) {
  if (!data) return <NoData />

  const { confirmed, dismissed, pending } = data
  const handled = confirmed + dismissed
  const total = handled + pending

  if (handled === 0) {
    return <span className="text-zinc-600 text-sm">No decisions yet</span>
  }

  const pctOf = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0)

  return (
    <div>
      <div className="flex h-3 rounded-full overflow-hidden mb-3 bg-zinc-800">
        {confirmed > 0 && (
          <div className="bg-emerald-500 h-full" style={{ width: `${pctOf(confirmed)}%` }} />
        )}
        {dismissed > 0 && (
          <div className="bg-zinc-600 h-full" style={{ width: `${pctOf(dismissed)}%` }} />
        )}
        {pending > 0 && (
          <div className="bg-blue-500 h-full" style={{ width: `${pctOf(pending)}%` }} />
        )}
      </div>
      <div className="flex gap-5 text-xs mb-2">
        <span>
          <span className="font-mono text-emerald-400">{confirmed}</span>
          <span className="text-zinc-500 ml-1">confirmed</span>
        </span>
        <span>
          <span className="font-mono text-zinc-400">{dismissed}</span>
          <span className="text-zinc-500 ml-1">dismissed</span>
        </span>
        <span>
          <span className="font-mono text-blue-400">{pending}</span>
          <span className="text-zinc-500 ml-1">pending</span>
        </span>
      </div>
      <p className="text-xs text-zinc-600">{confirmed} of {handled} handled</p>
    </div>
  )
}

export default function MetricsPage() {
  const [data, setData] = useState<MetricsData | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const m = await fetchMetrics()
        if (alive) { setData(m); setError(false) }
      } catch {
        if (alive) setError(true)
      }
    }
    load()
    const id = setInterval(load, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const fmt = (n: number | null | undefined) =>
    n == null ? <NoData /> : <>{n.toLocaleString()}</>

  const pct = (n: number | null | undefined) =>
    n == null ? <NoData /> : <>{(n * 100).toFixed(1)}%</>

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-medium text-zinc-50">Metrics</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Pipeline counters. Refreshes every 5 seconds.
          {error && <span className="ml-2 text-rose-400">Backend unreachable.</span>}
        </p>
      </div>

      {/* Row 1: core counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <StatCard label="Frames processed" value={fmt(data?.frames_processed_total)} />
        <StatCard label="Total incidents"  value={fmt(data?.incidents_total)} />
        <StatCard label="KB entries"        value={fmt(data?.kb_entry_count)} />
        <StatCard label="Alerts last hour"  value={fmt(data?.alerts_last_hour)} />
      </div>

      {/* Row 2: gate filter rate (full width) */}
      {/* TODO: add per-stage latency (p50/p95) here once pipeline timing is instrumented
               (tracked in docs/KNOWN_LIMITATIONS.md) */}
      <div className="mb-4">
        <div className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">Gate filter rate</p>
          <div className="text-2xl font-mono font-medium text-zinc-50 mb-1">
            {pct(data?.gate_filter_rate)}
          </div>
          <p className="text-xs text-zinc-600">
            Fraction of frames suppressed by the event gate.
          </p>
        </div>
      </div>

      {/* Row 3: severity breakdown + decision ratio */}
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
            Severity — last 24h
          </p>
          <SeverityBar data={data?.incidents_by_severity_24h ?? null} />
        </div>
        <div className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
            Decision ratio
          </p>
          <DecisionBar data={data?.operator_decisions} />
        </div>
      </div>
    </div>
  )
}
