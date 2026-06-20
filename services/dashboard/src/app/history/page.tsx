import Link from 'next/link'
import { ChevronUp, ChevronDown } from 'lucide-react'
import { fetchIncidents } from '@/lib/api'
import type { Incident } from '@/types/incident'

const SEV_BADGE: Record<string, string> = {
  high:   'bg-rose-950 text-rose-400 border border-rose-500/20',
  medium: 'bg-amber-950 text-amber-400 border border-amber-500/20',
  low:    'bg-sky-950 text-sky-400 border border-sky-500/20',
}

interface SearchParams {
  camera?: string
  room?: string
  state?: string
  severity?: string
  page?: string
  sort?: string
  dir?: string
}

const PAGE_SIZE = 20

const SORTABLE = ['created_at', 'severity', 'confidence', 'state'] as const
type SortField = typeof SORTABLE[number]

function sortIncidents(
  incidents: Incident[],
  field: string,
  dir: string
): Incident[] {
  if (!SORTABLE.includes(field as SortField)) return incidents
  return [...incidents].sort((a, b) => {
    const av = a[field as keyof Incident] ?? ''
    const bv = b[field as keyof Incident] ?? ''
    const cmp = av < bv ? -1 : av > bv ? 1 : 0
    return dir === 'asc' ? cmp : -cmp
  })
}

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: SearchParams
}) {
  const page = Math.max(1, parseInt(searchParams.page ?? '1', 10))
  const offset = (page - 1) * PAGE_SIZE
  const sortField = searchParams.sort ?? 'created_at'
  const sortDir   = searchParams.dir  ?? 'desc'

  const params: Record<string, string> = { limit: String(PAGE_SIZE), offset: String(offset) }
  if (searchParams.camera)   params.camera_id = searchParams.camera
  if (searchParams.room)     params.room_id   = searchParams.room
  if (searchParams.state)    params.state     = searchParams.state
  if (searchParams.severity) params.severity  = searchParams.severity

  let incidents: Incident[] = []
  try {
    incidents = await fetchIncidents(params)
  } catch {
    // backend may be offline; show empty state
  }

  const sorted = sortIncidents(incidents, sortField, sortDir)

  function buildUrl(overrides: SearchParams) {
    const merged = { ...searchParams, ...overrides }
    const qs = Object.entries(merged)
      .filter(([, v]) => v)
      .map(([k, v]) => `${k}=${encodeURIComponent(v!)}`)
      .join('&')
    return `/history${qs ? '?' + qs : ''}`
  }

  function sortUrl(field: string) {
    const newDir = sortField === field && sortDir === 'desc' ? 'asc' : 'desc'
    return buildUrl({ sort: field, dir: newDir, page: '1' })
  }

  function SortIcon({ field }: { field: string }) {
    if (sortField !== field)
      return <ChevronDown className="w-3 h-3 text-zinc-700 inline ml-1" />
    return sortDir === 'asc'
      ? <ChevronUp className="w-3 h-3 text-zinc-400 inline ml-1" />
      : <ChevronDown className="w-3 h-3 text-zinc-400 inline ml-1" />
  }

  const from = offset + 1
  const to   = offset + sorted.length

  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-medium text-zinc-50">Incident History</h1>
        <p className="text-sm text-zinc-500 mt-1">All recorded incidents, filterable and sortable.</p>
      </div>

      {/* Filter card */}
      <div className="border border-zinc-800 rounded-lg bg-zinc-900 p-4 mb-6">
        <form method="GET" action="/history" className="flex flex-wrap gap-3 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-500">Camera ID</label>
            <input
              name="camera"
              defaultValue={searchParams.camera ?? ''}
              placeholder="e.g. cam-01"
              className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 w-36 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-500">Room ID</label>
            <input
              name="room"
              defaultValue={searchParams.room ?? ''}
              placeholder="e.g. room-A"
              className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 w-36 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-500">State</label>
            <select
              name="state"
              defaultValue={searchParams.state ?? ''}
              className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors"
            >
              <option value="">All states</option>
              <option value="new">new</option>
              <option value="alert">alert</option>
              <option value="resolved">resolved</option>
              <option value="dismissed">dismissed</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-zinc-500">Severity</label>
            <select
              name="severity"
              defaultValue={searchParams.severity ?? ''}
              className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors"
            >
              <option value="">All severities</option>
              <option value="high">high</option>
              <option value="medium">medium</option>
              <option value="low">low</option>
            </select>
          </div>
          {/* Carry sort params through filter form */}
          <input type="hidden" name="sort" value={sortField} />
          <input type="hidden" name="dir"  value={sortDir} />
          <div className="flex gap-2 ml-auto">
            <button
              type="submit"
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 active:scale-[0.98] text-white text-sm font-medium rounded-lg transition-colors duration-150"
            >
              Filter
            </button>
            <Link
              href="/history"
              className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] text-zinc-300 text-sm rounded-lg border border-zinc-700 transition-colors duration-150"
            >
              Clear
            </Link>
          </div>
        </form>
      </div>

      {/* Table */}
      {sorted.length === 0 ? (
        <div className="border border-zinc-800 rounded-lg bg-zinc-900/40 py-16 text-center">
          <p className="text-sm font-medium text-zinc-500">No incidents found</p>
          <p className="text-xs text-zinc-600 mt-1">Try adjusting your filters.</p>
        </div>
      ) : (
        <div className="border border-zinc-800 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
              <tr className="text-left">
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider">
                  <Link href={sortUrl('created_at')} className="hover:text-zinc-300 transition-colors">
                    Time <SortIcon field="created_at" />
                  </Link>
                </th>
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider">
                  Event
                </th>
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider">
                  Room
                </th>
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider">
                  <Link href={sortUrl('severity')} className="hover:text-zinc-300 transition-colors">
                    Severity <SortIcon field="severity" />
                  </Link>
                </th>
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider">
                  <Link href={sortUrl('state')} className="hover:text-zinc-300 transition-colors">
                    State <SortIcon field="state" />
                  </Link>
                </th>
                <th className="px-4 py-2.5 font-medium text-zinc-500 text-xs uppercase tracking-wider text-right">
                  <Link href={sortUrl('confidence')} className="hover:text-zinc-300 transition-colors">
                    Conf. <SortIcon field="confidence" />
                  </Link>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {sorted.map((inc, i) => (
                <tr
                  key={inc.id}
                  className={`transition-colors hover:bg-zinc-800/50 ${i % 2 === 1 ? 'bg-zinc-900/40' : ''}`}
                >
                  <td className="px-4 py-3 text-xs text-zinc-500 font-mono whitespace-nowrap">
                    {new Date(inc.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/incidents/${inc.id}`}
                      className="text-zinc-300 hover:text-blue-400 capitalize transition-colors duration-150"
                    >
                      {inc.event_type.replace(/_/g, ' ')}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-zinc-400 text-xs">{inc.room_id}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`text-xs font-medium uppercase px-1.5 py-0.5 rounded tracking-wider ${SEV_BADGE[inc.severity] ?? 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
                    >
                      {inc.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-zinc-500 capitalize">{inc.state}</td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-400 text-right">
                    {inc.confidence.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      <div className="flex items-center justify-between mt-5 text-sm">
        <span className="text-xs text-zinc-600">
          {sorted.length > 0 ? `Showing ${from}–${to}` : 'No results'}
        </span>
        <div className="flex gap-2">
          {page > 1 && (
            <Link
              href={buildUrl({ page: String(page - 1) })}
              className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-[0.98] text-zinc-300 text-xs rounded-lg border border-zinc-800 transition-colors"
            >
              Previous
            </Link>
          )}
          {sorted.length === PAGE_SIZE && (
            <Link
              href={buildUrl({ page: String(page + 1) })}
              className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 active:scale-[0.98] text-zinc-300 text-xs rounded-lg border border-zinc-800 transition-colors"
            >
              Next
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}
