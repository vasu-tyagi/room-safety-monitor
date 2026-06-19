'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { submitFeedback } from '@/lib/api'

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

  const decision = localDecision ?? currentDecision

  if (decision) {
    return (
      <p className="text-sm text-gray-500">
        Operator decision:{' '}
        <span className="font-medium text-gray-200">{decision}</span>
      </p>
    )
  }

  async function handle(action: 'confirm' | 'dismiss') {
    setPending(true)
    setError(null)
    try {
      await submitFeedback(incidentId, action)
      setLocalDecision(action === 'confirm' ? 'confirmed' : 'dismissed')
      router.refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed. Try again.')
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="pt-2">
      <div className="flex gap-3">
        <button
          onClick={() => handle('confirm')}
          disabled={pending}
          className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white text-sm rounded disabled:opacity-50"
        >
          {pending ? 'Saving…' : 'Confirm Alert'}
        </button>
        <button
          onClick={() => handle('dismiss')}
          disabled={pending}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded disabled:opacity-50"
        >
          {pending ? 'Saving…' : 'Dismiss'}
        </button>
      </div>
      {error && (
        <p className="mt-2 text-xs text-red-400" role="alert">{error}</p>
      )}
    </div>
  )
}
