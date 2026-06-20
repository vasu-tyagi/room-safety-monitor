import { render, screen } from '@testing-library/react'
import { InspectorContent } from '@/components/InspectorContent'
import type { IncidentDetail } from '@/types/incident'

const INCIDENT: IncidentDetail = {
  id: 'test-id-1',
  camera_id: 'cam-01',
  room_id: 'room-A',
  event_type: 'fall_detected',
  severity: 'high',
  confidence: 0.88,
  rationale: 'Pose geometry indicates fall.',
  state: 'alert',
  created_at: '2026-01-01T10:00:00Z',
  evidence_clip_url: null,
  operator_decision: null,
  vlm_prompt: 'Describe what you see.',
  fired_rules: ['pose_fall_angle', 'low_aspect_ratio'],
  confidence_breakdown: { yolo: 0.9, pose: 0.85 },
  kb_matches: null,
  audit_entries: [
    {
      id: 'audit-1',
      from_state: 'new',
      to_state: 'alert',
      reason: 'confidence above threshold',
      agent_node: 'decide',
      created_at: '2026-01-01T10:00:01Z',
    },
  ],
}

test('InspectorContent renders all six section headings', () => {
  render(<InspectorContent incident={INCIDENT} />)
  expect(screen.getByText('Stream Ingest')).toBeInTheDocument()
  expect(screen.getByText('Fast Classical CV')).toBeInTheDocument()
  expect(screen.getByText('Event Gate')).toBeInTheDocument()
  expect(screen.getByText('VLM Deep Analysis')).toBeInTheDocument()
  expect(screen.getByText('AI Agent')).toBeInTheDocument()
  expect(screen.getByText('Persistence + KB')).toBeInTheDocument()
})

test('InspectorContent renders fired rules', () => {
  render(<InspectorContent incident={INCIDENT} />)
  expect(screen.getByText('pose_fall_angle')).toBeInTheDocument()
  expect(screen.getByText('low_aspect_ratio')).toBeInTheDocument()
})

test('InspectorContent renders audit entry FSM transition', () => {
  render(<InspectorContent incident={INCIDENT} />)
  expect(screen.getByText('new')).toBeInTheDocument()
  expect(screen.getByText('alert')).toBeInTheDocument()
})

test('InspectorContent renders camera and room ID in L1', () => {
  render(<InspectorContent incident={INCIDENT} />)
  expect(screen.getByText('cam-01')).toBeInTheDocument()
  expect(screen.getByText('room-A')).toBeInTheDocument()
})
