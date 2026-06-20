const SEV_BAR: Record<string, string> = {
  high:   'bg-rose-500',
  medium: 'bg-amber-500',
  low:    'bg-sky-500',
}

const SOURCE_BARS = [
  'bg-blue-500',
  'bg-violet-500',
  'bg-cyan-500',
  'bg-indigo-500',
]

interface ConfidenceBarProps {
  breakdown: Record<string, number>
  fused: number
  severity: string
}

export function ConfidenceBar({ breakdown, fused, severity }: ConfidenceBarProps) {
  const entries = Object.entries(breakdown)

  return (
    <div className="space-y-3">
      {entries.map(([src, score], i) => (
        <div key={src} className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-400 capitalize">{src}</span>
            <span className="text-xs font-mono text-zinc-300">{score.toFixed(3)}</span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${SOURCE_BARS[i % SOURCE_BARS.length]} transition-all duration-500`}
              style={{ width: `${Math.min(100, Math.round(score * 100))}%` }}
            />
          </div>
        </div>
      ))}

      {/* Fused total */}
      <div className="pt-2 border-t border-zinc-800 space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-300">Fused</span>
          <span className="text-xs font-mono font-medium text-zinc-200">{fused.toFixed(3)}</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${SEV_BAR[severity.toLowerCase()] ?? 'bg-blue-500'} transition-all duration-500`}
            style={{ width: `${Math.min(100, Math.round(fused * 100))}%` }}
          />
        </div>
      </div>
    </div>
  )
}
