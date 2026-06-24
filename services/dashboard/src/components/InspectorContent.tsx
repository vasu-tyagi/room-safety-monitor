import { Eye, Brain, Filter, MessagesSquare, GitBranch, Database } from 'lucide-react'
import type { IncidentDetail, AuditEntry } from '@/types/incident'
import { ConfidenceBreakdown } from '@/components/ConfidenceBreakdown'

interface SectionProps {
  icon: React.ReactNode
  id: string
  title: string
  children: React.ReactNode
}

function Section({ icon, id, title, children }: SectionProps) {
  return (
    <section className="border border-zinc-800 rounded-lg bg-zinc-900 overflow-hidden">
      <div className="flex items-center gap-2.5 px-4 py-3 border-b border-zinc-800 bg-zinc-950/50">
        <span className="text-zinc-500">{icon}</span>
        <span className="text-xs font-mono text-zinc-500 font-medium">{id}</span>
        <span className="text-sm font-medium text-zinc-200">{title}</span>
      </div>
      <div className="p-4">{children}</div>
    </section>
  )
}

function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-3 rounded-lg bg-zinc-950 border border-zinc-800 px-3 py-2.5 text-xs text-zinc-500">
      {children}
    </div>
  )
}

interface Props {
  incident: IncidentDetail
}

export function InspectorContent({ incident }: Props) {
  const auditEntries: AuditEntry[] = incident.audit_entries ?? []

  return (
    <div className="space-y-4">

      {/* L1 Stream Ingest */}
      <Section
        icon={<Eye className="w-4 h-4" />}
        id="L1"
        title="Stream Ingest"
      >
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
          <dt className="text-zinc-500">Camera</dt>
          <dd className="text-zinc-300 font-mono">{incident.camera_id}</dd>
          <dt className="text-zinc-500">Room</dt>
          <dd className="text-zinc-300 font-mono">{incident.room_id}</dd>
        </dl>
        <InfoBox>
          Production: RTSP per-camera workers with hardware decode. Running: <span className="font-mono">cv2.VideoCapture(file_path)</span>.
        </InfoBox>
      </Section>

      {/* L2 Fast CV */}
      <Section
        icon={<Brain className="w-4 h-4" />}
        id="L2"
        title="Fast Classical CV"
      >
        {incident.confidence_breakdown && Object.keys(incident.confidence_breakdown).length > 0 ? (
          <ConfidenceBreakdown
            breakdown={incident.confidence_breakdown}
            fused={incident.confidence}
            severity={incident.severity}
          />
        ) : (
          <p className="text-xs text-zinc-500">No per-source confidence data for this incident.</p>
        )}
        <InfoBox>
          Models run in-process on CPU. Production would use GPU inference serving to meet sub-50ms per-frame latency.
        </InfoBox>
      </Section>

      {/* Event Gate */}
      <Section
        icon={<Filter className="w-4 h-4" />}
        id="Gate"
        title="Event Gate"
      >
        {incident.fired_rules && incident.fired_rules.length > 0 ? (
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
        ) : (
          <p className="text-xs text-zinc-500">No gate rules recorded for this incident.</p>
        )}
        <InfoBox>
          Seven deterministic rules. Policies in <span className="font-mono">config/rooms.yaml</span>.
        </InfoBox>
      </Section>

      {/* L3 VLM */}
      <Section
        icon={<MessagesSquare className="w-4 h-4" />}
        id="L3"
        title="VLM Deep Analysis"
      >
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs mb-3">
          <dt className="text-zinc-500">Rationale</dt>
          <dd className="text-zinc-300 leading-relaxed">{incident.rationale}</dd>
        </dl>
        {incident.vlm_prompt && (
          <details className="mt-2">
            <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">
              VLM prompt (expand)
            </summary>
            <pre className="mt-2 text-xs text-zinc-400 bg-zinc-950 border border-zinc-800 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
              {incident.vlm_prompt}
            </pre>
          </details>
        )}
        <InfoBox>
          Qwen 2.5 VL 72B via HF Inference Providers. Falls back to stub when <span className="font-mono">VLM_MODE=stub</span>.
        </InfoBox>
      </Section>

      {/* L4 Agent */}
      <Section
        icon={<GitBranch className="w-4 h-4" />}
        id="L4"
        title="AI Agent"
      >
        {auditEntries.length > 0 ? (
          <ol className="space-y-2">
            {auditEntries.map((entry) => (
              <li key={entry.id} className="flex items-start gap-3 text-xs">
                <span className="font-mono text-zinc-600 shrink-0 mt-px">
                  {new Date(entry.created_at).toLocaleTimeString()}
                </span>
                <div>
                  <span className="text-zinc-400 font-mono">{entry.from_state}</span>
                  <span className="text-zinc-700 mx-1.5">→</span>
                  <span className="text-zinc-300 font-mono">{entry.to_state}</span>
                  {entry.agent_node && (
                    <span className="text-zinc-600 ml-1.5">via {entry.agent_node}</span>
                  )}
                  {entry.reason && (
                    <p className="text-zinc-500 mt-0.5">{entry.reason}</p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-xs text-zinc-500">No FSM transitions recorded for this incident.</p>
        )}
        <InfoBox>
          LangGraph 6-node StateGraph. FSM transitions above are from the <span className="font-mono">incident_audit</span> table.
        </InfoBox>
      </Section>

      {/* L5 Persistence + KB */}
      <Section
        icon={<Database className="w-4 h-4" />}
        id="L5"
        title="Persistence + KB"
      >
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
          <dt className="text-zinc-500">Incident ID</dt>
          <dd className="text-zinc-300 font-mono break-all">{incident.id}</dd>
          <dt className="text-zinc-500">Evidence clip</dt>
          <dd className="text-zinc-300 font-mono">{incident.evidence_clip_url ?? 'none'}</dd>
          <dt className="text-zinc-500">Operator decision</dt>
          <dd className="text-zinc-300 font-mono">{incident.operator_decision ?? 'pending'}</dd>
        </dl>
        {incident.kb_matches && incident.kb_matches.length > 0 && (
          <details className="mt-3">
            <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">
              KB matches ({incident.kb_matches.length})
            </summary>
            <ul className="mt-2 space-y-2">
              {incident.kb_matches.map((m) => (
                <li key={m.id} className="border-l-2 border-zinc-700 pl-3 text-xs">
                  <div className="text-zinc-500 mb-0.5">
                    <span className="font-mono">{m.similarity.toFixed(2)}</span>
                    <span className="text-zinc-700 mx-1.5">·</span>
                    {m.label}
                  </div>
                  <p className="text-zinc-400 line-clamp-2">{m.rationale}</p>
                </li>
              ))}
            </ul>
          </details>
        )}
        <InfoBox>
          Postgres + pgvector. Clips save to local filesystem; MinIO upload deferred to Slice 9.
        </InfoBox>
      </Section>

    </div>
  )
}
