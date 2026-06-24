'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { CheckCircle, XCircle } from 'lucide-react'
import { submitFeedback } from '@/lib/api'
import { usePipelineProgress } from '@/context/PipelineProgressContext'
import { useAlerts } from '@/context/AlertContext'
import { cn } from '@/lib/utils'

const DECISION_STYLE: Record<string, { cls: string }> = {
  confirmed: { cls: 'bg-emerald-950 text-emerald-400 border border-emerald-500/20' },
  dismissed: { cls: 'bg-zinc-800 text-zinc-400 border border-zinc-700' },
}

export function FeedbackButtons({
  incidentId,
  currentDecision,
}: {
  incidentId: string
  currentDecision: string | null
}) {
  const [pending, setPending] = useState(false)
  const [localDecision, setLocalDecision] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()
  const { reset: resetPipeline } = usePipelineProgress()
  const { updateDecision } = useAlerts()

  const decision = localDecision ?? currentDecision

  async function handle(action: 'confirm' | 'dismiss') {
    setPending(true)
    setError(null)
    try {
      await submitFeedback(incidentId, action)
      const resolved = action === 'confirm' ? 'confirmed' : 'dismissed'
      setLocalDecision(resolved)
      updateDecision(incidentId, resolved)
      resetPipeline()
      router.refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed. Try again.')
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="pt-4 border-t border-zinc-800 space-y-3">
      <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
        Operator Decision
      </p>

      {decision ? (
        <div
          className={cn(
            'inline-flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-full',
            DECISION_STYLE[decision]?.cls ?? 'bg-zinc-800 text-zinc-400 border border-zinc-700'
          )}
        >
          {decision === 'confirmed'
            ? <CheckCircle className="w-3.5 h-3.5 shrink-0" />
            : <XCircle className="w-3.5 h-3.5 shrink-0" />}
          {/* Lowercase to match test expectations */}
          {decision}
        </div>
      ) : (
        <div className="flex gap-2">
          <button
            onClick={() => handle('confirm')}
            disabled={pending}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98] text-white text-sm font-medium rounded-lg transition-colors duration-200 disabled:opacity-50"
          >
            {pending ? 'Saving…' : 'Confirm Alert'}
          </button>
          <button
            onClick={() => handle('dismiss')}
            disabled={pending}
            className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 active:scale-[0.98] text-zinc-200 text-sm font-medium rounded-lg border border-zinc-600 transition-colors duration-200 disabled:opacity-50"
          >
            {pending ? 'Saving…' : 'Dismiss'}
          </button>
        </div>
      )}

      {error && (
        <p className="text-xs text-rose-400" role="alert">{error}</p>
      )}
    </div>
  )
}
