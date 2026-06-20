import { render } from '@testing-library/react'
import { AlertCard } from '@/components/AlertCard'
import type { Incident } from '@/types/incident'

const base: Incident = {
  id: 'abc-123',
  camera_id: 'cam-01',
  room_id: 'room-A',
  event_type: 'fall_detected',
  severity: 'high',
  confidence: 0.92,
  rationale: 'Person fell near bed.',
  state: 'alert',
  created_at: '2026-06-01T10:00:00Z',
  evidence_clip_url: null,
  operator_decision: null,
}

test('AlertCard high severity snapshot (pending)', () => {
  const { container } = render(<AlertCard incident={base} />)
  expect(container).toMatchSnapshot()
})

test('AlertCard low severity snapshot (pending)', () => {
  const low: Incident = { ...base, id: 'def-456', severity: 'low', state: 'new' }
  const { container } = render(<AlertCard incident={low} />)
  expect(container).toMatchSnapshot()
})

test('AlertCard confirmed snapshot (dimmed + badge)', () => {
  const confirmed: Incident = { ...base, id: 'ghi-789', operator_decision: 'confirmed' }
  const { container } = render(<AlertCard incident={confirmed} />)
  expect(container).toMatchSnapshot()
})

test('AlertCard dismissed snapshot (dimmed + badge)', () => {
  const dismissed: Incident = { ...base, id: 'jkl-012', operator_decision: 'dismissed' }
  const { container } = render(<AlertCard incident={dismissed} />)
  expect(container).toMatchSnapshot()
})
