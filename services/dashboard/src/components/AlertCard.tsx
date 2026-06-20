import Link from 'next/link'
import type { Incident } from '@/types/incident'
import { RelativeTime } from './RelativeTime'
import { cn } from '@/lib/utils'

const SEV_BADGE: Record<string, string> = {
  high:   'bg-rose-950 text-rose-400 border border-rose-500/20',
  medium: 'bg-amber-950 text-amber-400 border border-amber-500/20',
  low:    'bg-sky-950 text-sky-400 border border-sky-500/20',
}

const CARD_BORDER: Record<string, string> = {
  high:   'border-l-rose-500',
  medium: 'border-l-amber-500',
  low:    'border-l-sky-500',
}

const DECISION_BADGE: Record<string, string> = {
  confirmed: 'bg-emerald-950 text-emerald-400 border border-emerald-500/20',
  dismissed: 'bg-zinc-800 text-zinc-500 border border-zinc-700',
}

interface AlertCardProps {
  incident: Incident
  isNew?: boolean
}

export function AlertCard({ incident, isNew }: AlertCardProps) {
  const sev = incident.severity.toLowerCase()
  const decision = incident.operator_decision
  const handled = decision !== null

  return (
    <Link href={`/incidents/${incident.id}`} className="block group">
      <div
        className={cn(
          'relative border border-zinc-800 border-l-4 rounded-lg bg-zinc-900 px-4 py-3',
          'transition-colors duration-200 hover:bg-zinc-800/50',
          'active:scale-[0.995]',
          CARD_BORDER[sev] ?? 'border-l-zinc-600',
          handled && 'opacity-60',
          isNew && 'animate-slide-in'
        )}
        data-testid="alert-card"
      >
        {/* Decision badge */}
        {handled && (
          <span
            className={cn(
              'absolute top-3 right-3 text-xs font-medium px-2 py-0.5 rounded-full',
              DECISION_BADGE[decision!] ?? 'bg-zinc-800 text-zinc-500'
            )}
          >
            {decision === 'confirmed' ? 'Confirmed' : 'Dismissed'}
          </span>
        )}

        {/* Top line: severity · room · camera · timestamp */}
        <div className="flex items-center gap-2 mb-1.5 min-w-0">
          <span
            className={cn(
              'text-xs font-medium px-1.5 py-0.5 rounded uppercase tracking-wider shrink-0',
              SEV_BADGE[sev] ?? 'bg-zinc-800 text-zinc-400 border border-zinc-700'
            )}
            data-testid="severity-badge"
          >
            {incident.severity}
          </span>
          <span className="text-sm font-medium text-zinc-200 truncate">
            {incident.room_id}
          </span>
          <span className="text-zinc-700 shrink-0">·</span>
          <span className="text-xs text-zinc-500 truncate">{incident.camera_id}</span>
          <span className="ml-auto shrink-0">
            <RelativeTime timestamp={incident.created_at} />
          </span>
        </div>

        {/* Rationale */}
        <p className={cn(
          'text-sm text-zinc-400 line-clamp-2 leading-snug',
          handled ? 'pr-24' : 'pr-2'
        )}>
          {incident.rationale}
        </p>
      </div>
    </Link>
  )
}
