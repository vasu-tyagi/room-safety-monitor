'use client'
import { X } from 'lucide-react'
import type { LayerInfo } from '@/lib/api'

interface Props {
  layer: LayerInfo
  onClose: () => void
}

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

export function ProvenanceDisclosure({ layer, onClose }: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`${layer.name} provenance`}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative w-full max-w-lg bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between p-5 pb-4 border-b border-zinc-800">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-zinc-500">{layer.id}</span>
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded uppercase tracking-wider ${STATUS_BADGE[layer.status] ?? STATUS_BADGE.not_built}`}
              >
                {STATUS_LABEL[layer.status] ?? layer.status}
              </span>
            </div>
            <h2 className="text-base font-medium text-zinc-50">{layer.name}</h2>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 transition-colors p-1 -mr-1 -mt-1"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Description */}
        <div className="px-5 py-4 border-b border-zinc-800">
          <p className="text-sm text-zinc-400 leading-relaxed">{layer.description}</p>
        </div>

        {/* Components table */}
        <div className="px-5 py-4">
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
            Components
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-zinc-600">
                <th className="pb-2 font-medium w-1/3">Component</th>
                <th className="pb-2 font-medium w-1/3">Running now</th>
                <th className="pb-2 font-medium w-1/3">Production target</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
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
      </div>
    </div>
  )
}
