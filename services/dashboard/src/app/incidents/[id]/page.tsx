import { notFound } from 'next/navigation'
import Link from 'next/link'
import { Microscope } from 'lucide-react'
import { fetchIncident } from '@/lib/api'
import { Breadcrumb } from '@/components/Breadcrumb'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { FeedbackButtons } from './FeedbackButtons'

const SEV_BADGE: Record<string, string> = {
  high:   'bg-rose-950 text-rose-400 border border-rose-500/20',
  medium: 'bg-amber-950 text-amber-400 border border-amber-500/20',
  low:    'bg-sky-950 text-sky-400 border border-sky-500/20',
}

const STATE_BADGE: Record<string, string> = {
  new:       'bg-zinc-800 text-zinc-400 border border-zinc-700',
  alert:     'bg-rose-950 text-rose-400 border border-rose-500/20',
  resolved:  'bg-emerald-950 text-emerald-400 border border-emerald-500/20',
  dismissed: 'bg-zinc-800 text-zinc-500 border border-zinc-700',
}

export default async function IncidentDetailPage({
  params,
}: {
  params: { id: string }
}) {
  let incident
  try {
    incident = await fetchIncident(params.id)
  } catch {
    notFound()
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
  const clipUrl = incident.evidence_clip_url
    ? `${apiUrl}/${incident.evidence_clip_url}`
    : null
  const sev = incident.severity.toLowerCase()

  return (
    <div>
      <Breadcrumb
        crumbs={[
          { label: 'Live Feed', href: '/' },
          { label: `Incident ${params.id.slice(0, 8)}` },
        ]}
      />

      {/* Page header row with Inspect link */}
      <div className="flex items-start justify-between mb-4">
        <div />
        <Link
          href={`/incidents/${params.id}/inspect`}
          className="flex items-center gap-1.5 px-3 py-1.5 border border-zinc-700 text-zinc-400 hover:text-zinc-50 hover:border-zinc-500 text-sm rounded-lg transition-colors"
        >
          <Microscope className="w-3.5 h-3.5" />
          Inspect
        </Link>
      </div>

      {/* Two-column on desktop: video left (sticky), metadata right (scroll) */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">

        {/* Left column — video + event meta */}
        <div className="w-full lg:w-[58%] lg:sticky lg:top-[7.5rem]">
          {clipUrl ? (
            <div className="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950">
              <video
                controls
                className="w-full max-h-[26rem] bg-black"
                src={clipUrl}
              />
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 border border-zinc-800 rounded-lg bg-zinc-900/50">
              <p className="text-sm text-zinc-600">No evidence clip</p>
            </div>
          )}

          <div className="mt-4 space-y-2">
            <h1 className="text-xl font-medium text-zinc-50 capitalize">
              {incident.event_type.replace(/_/g, ' ')}
            </h1>
            <p className="text-sm text-zinc-500 font-mono">
              {incident.camera_id} · {incident.room_id} ·{' '}
              {new Date(incident.created_at).toLocaleString()}
            </p>
            <div className="flex items-center gap-2 pt-1">
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded uppercase tracking-wider ${SEV_BADGE[sev] ?? 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
              >
                {incident.severity}
              </span>
              <span
                className={`text-xs px-2 py-0.5 rounded capitalize ${STATE_BADGE[incident.state] ?? 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
              >
                {incident.state}
              </span>
            </div>
          </div>
        </div>

        {/* Right column — scrollable metadata */}
        <div className="w-full lg:w-[42%] space-y-4">

          {/* Rationale */}
          <section className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
            <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
              Rationale
            </h2>
            <p className="text-sm text-zinc-300 leading-relaxed">{incident.rationale}</p>
          </section>

          {/* Confidence Breakdown */}
          {incident.confidence_breakdown && Object.keys(incident.confidence_breakdown).length > 0 && (
            <section className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
              <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
                Confidence Breakdown
              </h2>
              <ConfidenceBar
                breakdown={incident.confidence_breakdown}
                fused={incident.confidence}
                severity={incident.severity}
              />
            </section>
          )}

          {/* Gate Rules Fired */}
          {incident.fired_rules && incident.fired_rules.length > 0 && (
            <section className="border border-zinc-800 rounded-lg bg-zinc-900 p-4">
              <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">
                Gate Rules Fired
              </h2>
              <div className="flex flex-wrap gap-1.5">
                {incident.fired_rules.map((rule) => (
                  <span
                    key={rule}
                    className="text-xs font-mono text-zinc-300 bg-zinc-950 border border-zinc-800 px-2 py-0.5 rounded"
                  >
                    {rule}
                  </span>
                ))}
              </div>
            </section>
          )}

          {/* VLM Prompt — collapsible */}
          {incident.vlm_prompt && (
            <section className="border border-zinc-800 rounded-lg bg-zinc-900 overflow-hidden">
              <details>
                <summary className="px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-300 hover:bg-zinc-800/40 transition-colors duration-150">
                  VLM Prompt
                </summary>
                <div className="px-4 pb-4">
                  <pre className="text-xs text-zinc-400 bg-zinc-950 border border-zinc-800 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                    {incident.vlm_prompt}
                  </pre>
                </div>
              </details>
            </section>
          )}

          {/* KB Matches — collapsible */}
          {incident.kb_matches && incident.kb_matches.length > 0 && (
            <section className="border border-zinc-800 rounded-lg bg-zinc-900 overflow-hidden">
              <details>
                <summary className="px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-300 hover:bg-zinc-800/40 transition-colors duration-150">
                  KB Matches ({incident.kb_matches.length})
                </summary>
                <ul className="px-4 pb-4 space-y-2">
                  {incident.kb_matches.map((m) => (
                    <li key={m.id} className="border-l-2 border-zinc-700 pl-3 py-1">
                      <div className="flex items-center gap-2 text-xs text-zinc-500 mb-0.5">
                        <span className="font-mono">{m.similarity.toFixed(2)}</span>
                        <span className="text-zinc-700">·</span>
                        <span>{m.label}</span>
                        <span className="text-zinc-700">·</span>
                        <span>{m.operator_decision}</span>
                      </div>
                      <p className="text-xs text-zinc-400 line-clamp-2">{m.rationale}</p>
                    </li>
                  ))}
                </ul>
              </details>
            </section>
          )}

          {/* Decision controls */}
          <FeedbackButtons
            incidentId={incident.id}
            currentDecision={incident.operator_decision}
          />
        </div>
      </div>
    </div>
  )
}
