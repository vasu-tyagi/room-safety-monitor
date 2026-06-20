import { fetchArchitecture } from '@/lib/api'
import { ArchitectureContent } from '@/components/ArchitectureContent'

export default async function ArchitecturePage() {
  let data
  try {
    data = await fetchArchitecture()
  } catch {
    data = null
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-medium text-zinc-50">Architecture</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Six-layer pipeline — what is real vs. approximated vs. not yet built.
        </p>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-6 flex-wrap">
        {[
          { label: 'Real', cls: 'bg-emerald-950 text-emerald-400 border-emerald-500/20' },
          { label: 'Approximated / Partial', cls: 'bg-amber-950 text-amber-400 border-amber-500/20' },
          { label: 'Not built', cls: 'bg-blue-950 text-blue-400 border-blue-500/20' },
        ].map(({ label, cls }) => (
          <span key={label} className={`text-xs font-medium px-2 py-0.5 rounded border uppercase tracking-wider ${cls}`}>
            {label}
          </span>
        ))}
      </div>

      {data ? (
        <ArchitectureContent layers={data.layers} />
      ) : (
        <div className="border border-zinc-800 rounded-lg bg-zinc-900/40 py-16 text-center">
          <p className="text-sm text-zinc-500">Backend unreachable — architecture data unavailable.</p>
        </div>
      )}
    </div>
  )
}
