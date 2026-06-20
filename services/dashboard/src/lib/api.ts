import type { Incident, IncidentDetail } from '@/types/incident'

export interface MetricsData {
  frames_processed_total: number | null
  incidents_total: number | null
  alerts_last_hour: number | null
  gate_filter_rate: number | null
  kb_entry_count: number | null
  incidents_by_severity_24h: Record<string, number> | null
}

export interface LayerComponent {
  name: string
  real: string
  production: string
}

export interface LayerInfo {
  id: string
  name: string
  status: 'real' | 'approximated' | 'partial' | 'not_built'
  description: string
  components: LayerComponent[]
}

export interface ArchitectureData {
  layers: LayerInfo[]
}

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export function wsUrl(): string {
  return API_URL.replace(/^https?/, (p) => (p === 'https' ? 'wss' : 'ws')) + '/ws/alerts'
}

export async function fetchIncidents(
  params?: Record<string, string>
): Promise<Incident[]> {
  const url = new URL(`${API_URL}/incidents`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v))
  }
  const res = await fetch(url.toString(), { cache: 'no-store' })
  if (!res.ok) throw new Error(`GET /incidents ${res.status}`)
  return res.json()
}

export async function fetchIncident(id: string): Promise<IncidentDetail> {
  const res = await fetch(`${API_URL}/incidents/${id}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`GET /incidents/${id} ${res.status}`)
  return res.json()
}

export async function fetchMetrics(): Promise<MetricsData> {
  const res = await fetch(`${API_URL}/metrics`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`GET /metrics ${res.status}`)
  return res.json()
}

export async function fetchArchitecture(): Promise<ArchitectureData> {
  const res = await fetch(`${API_URL}/architecture`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`GET /architecture ${res.status}`)
  return res.json()
}

export async function submitFeedback(
  id: string,
  decision: 'confirm' | 'dismiss',
  note?: string
): Promise<{ status: string; operator_decision: string }> {
  const res = await fetch(`${API_URL}/incidents/${id}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, note }),
  })
  if (!res.ok) throw new Error(`POST /incidents/${id}/feedback ${res.status}`)
  return res.json()
}
