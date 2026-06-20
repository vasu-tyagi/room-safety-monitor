import type { LayerInfo } from '@/lib/api'

const STATUS_BADGE: Record<string, string> = {
  real:         'bg-emerald-950 text-emerald-400 border border-emerald-500/20',
  approximated: 'bg-amber-950 text-amber-400 border border-amber-500/20',
  partial:      'bg-amber-950 text-amber-400 border border-amber-500/20',
  not_built:    'bg-blue-950 text-blue-400 border border-blue-500/20',
}

const STATUS_LABEL: Record<string, string> = {
  real:         'Real',
  approximated: 'Approximated',
  partial:      'Partial',
  not_built:    'Not built',
}

export function ArchitectureContent({ layers }: { layers: LayerInfo[] }) {
  return (
    <div className="relative">
      {/* Vertical timeline line */}
      <div className="absolute left-[1.875rem] top-0 bottom-0 w-px bg-zinc-800 hidden md:block" />

      <ol className="space-y-6">
        {layers.map((layer) => (
          <li key={layer.id} className="relative md:pl-16">
            {/* Timeline dot */}
            <div className="absolute left-6 top-5 w-3 h-3 rounded-full border-2 border-zinc-700 bg-zinc-950 hidden md:block" />

            <details className="group border border-zinc-800 rounded-lg bg-zinc-900 overflow-hidden">
              <summary className="flex items-center justify-between px-4 py-3.5 cursor-pointer hover:bg-zinc-800/40 transition-colors duration-150 list-none">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-zinc-500 w-8 shrink-0">{layer.id}</span>
                  <span className="text-sm font-medium text-zinc-200">{layer.name}</span>
                  <span
                    className={`text-xs font-medium px-1.5 py-0.5 rounded uppercase tracking-wider ${STATUS_BADGE[layer.status] ?? STATUS_BADGE.not_built}`}
                  >
                    {STATUS_LABEL[layer.status] ?? layer.status}
                  </span>
                </div>
                <svg
                  className="w-4 h-4 text-zinc-600 group-open:rotate-180 transition-transform duration-150 shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </summary>

              <div className="px-4 pb-4 pt-2 border-t border-zinc-800">
                <p className="text-sm text-zinc-400 leading-relaxed mb-4">{layer.description}</p>

                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-zinc-600 border-b border-zinc-800">
                      <th className="pb-2 font-medium w-1/3">Component</th>
                      <th className="pb-2 font-medium w-1/3">Running now</th>
                      <th className="pb-2 font-medium w-1/3">Production target</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60">
                    {layer.components.map((c) => (
                      <tr key={c.name}>
                        <td className="py-2 text-zinc-400 font-medium">{c.name}</td>
                        <td className="py-2 text-zinc-400 font-mono">{c.real}</td>
                        <td className="py-2 text-zinc-500 font-mono">{c.production}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </li>
        ))}
      </ol>
    </div>
  )
}
