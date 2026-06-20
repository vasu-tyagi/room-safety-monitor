const SOURCE_COLORS: Record<string, { bar: string; text: string }> = {
  yolo:     { bar: 'bg-blue-500',   text: 'text-blue-400' },
  pose:     { bar: 'bg-violet-500', text: 'text-violet-400' },
  action:   { bar: 'bg-cyan-500',   text: 'text-cyan-400' },
  vlm:      { bar: 'bg-indigo-500', text: 'text-indigo-400' },
  default:  { bar: 'bg-zinc-500',   text: 'text-zinc-400' },
}

const SEV_BAR: Record<string, string> = {
  high:   'bg-rose-500',
  medium: 'bg-amber-500',
  low:    'bg-sky-500',
}

interface Props {
  breakdown: Record<string, number>
  fused: number
  severity: string
}

export function ConfidenceBreakdown({ breakdown, fused, severity }: Props) {
  const entries = Object.entries(breakdown)

  return (
    <div className="space-y-3">
      {entries.map(([source, score]) => {
        const color = SOURCE_COLORS[source] ?? SOURCE_COLORS.default
        const pct = Math.min(100, Math.round(score * 100))
        return (
          <div key={source}>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className={`font-mono font-medium uppercase ${color.text}`}>{source}</span>
              <span className="text-zinc-500">{pct}%</span>
            </div>
            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${color.bar} transition-all`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}

      {/* Fused total */}
      <div className="border-t border-zinc-800 pt-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="font-medium text-zinc-300">Fused confidence</span>
          <span className="text-zinc-400">{Math.round(fused * 100)}%</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${SEV_BAR[severity] ?? 'bg-zinc-500'} transition-all`}
            style={{ width: `${Math.min(100, Math.round(fused * 100))}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-zinc-600 mt-1">
        Weights set in <span className="font-mono">config/policies/</span>. Sources missing from breakdown used default weight 1.0.
      </p>
    </div>
  )
}
