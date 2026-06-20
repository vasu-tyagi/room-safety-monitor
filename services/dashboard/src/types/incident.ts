export type Severity = 'high' | 'medium' | 'low'
export type IncidentState = 'new' | 'alert' | 'resolved' | 'dismissed'

export interface Incident {
  id: string
  camera_id: string
  room_id: string
  event_type: string
  severity: Severity
  confidence: number
  rationale: string
  state: IncidentState
  created_at: string
  evidence_clip_url: string | null
  operator_decision: string | null
}

export interface KBMatch {
  id: string
  label: string
  rationale: string
  operator_decision: string
  similarity: number
}

export interface AuditEntry {
  id: string
  from_state: string
  to_state: string
  reason: string | null
  agent_node: string | null
  created_at: string
}

export interface IncidentDetail extends Incident {
  vlm_prompt: string | null
  fired_rules: string[] | null
  confidence_breakdown: Record<string, number> | null
  kb_matches: KBMatch[] | null
  audit_entries: AuditEntry[] | null
}
