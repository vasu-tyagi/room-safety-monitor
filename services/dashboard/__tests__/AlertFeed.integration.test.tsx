/**
 * Integration test: mock WebSocket delivers a message → AlertCard appears in the live feed.
 *
 * We replace the global WebSocket with a minimal stub that lets the test
 * control what messages the hook receives.
 */
import { render, screen, act, fireEvent } from '@testing-library/react'

// Must be set before the module under test imports useAlertFeed
let capturedOnMessage: ((ev: MessageEvent) => void) | null = null

class FakeWebSocket {
  static OPEN = 1
  readyState = FakeWebSocket.OPEN
  onopen: (() => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(_url: string) {
    capturedOnMessage = (ev) => this.onmessage?.(ev)
    // trigger onopen asynchronously so the hook's effect can register handlers first
    setTimeout(() => this.onopen?.(), 0)
  }

  send(_data: string) {}
  close() {}
}

// Mock wsUrl — AlertFeed uses it indirectly via useAlertFeed
jest.mock('@/lib/api', () => ({
  wsUrl: jest.fn().mockReturnValue('ws://localhost:8000/ws/alerts'),
}))

// Inject our fake WebSocket before the module is evaluated
beforeAll(() => {
  ;(global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket
})

// Clear localStorage between tests so the showPending toggle starts clean.
beforeEach(() => localStorage.clear())

import { AlertFeed } from '@/components/AlertFeed'
import { AlertProvider, useAlerts } from '@/context/AlertContext'

test('receives WebSocket alert and renders AlertCard', async () => {
  render(
    <AlertProvider>
      <AlertFeed initialIncidents={[]} />
    </AlertProvider>
  )

  // Wait for onopen to fire and connection to be established
  await act(async () => {
    await new Promise((r) => setTimeout(r, 50))
  })

  const incident = {
    id: 'ws-001',
    camera_id: 'cam-1',
    room_id: 'room-B',
    event_type: 'fall_detected',
    severity: 'high',
    confidence: 0.88,
    rationale: 'Person on floor.',
    state: 'alert',
    created_at: new Date().toISOString(),
    evidence_clip_url: null,
    operator_decision: null,
  }

  await act(async () => {
    capturedOnMessage?.(
      new MessageEvent('message', {
        data: JSON.stringify({ type: 'alert', incident }),
      })
    )
  })

  // AlertCard renders room_id and rationale — verify both appear
  expect(screen.getByText(/room-B/i)).toBeInTheDocument()
  expect(screen.getByText(/Person on floor/i)).toBeInTheDocument()
})

test('renders pre-populated incidents from initialIncidents prop', async () => {
  const incidents = [
    {
      id: 'pre-001',
      camera_id: 'cam-2',
      room_id: 'room-C',
      event_type: 'fall',
      severity: 'high' as const,
      confidence: 0.9,
      rationale: 'Initial server incident.',
      state: 'alert' as const,
      created_at: new Date().toISOString(),
      evidence_clip_url: null,
      operator_decision: null,
    },
  ]
  await act(async () => {
    render(
      <AlertProvider>
        <AlertFeed initialIncidents={incidents} />
      </AlertProvider>
    )
  })
  expect(screen.getByText(/room-C/i)).toBeInTheDocument()
  expect(screen.queryByText(/No alerts yet/i)).toBeNull()
})

test('pending alert disappears from feed after operator submits feedback', async () => {
  const pendingIncident = {
    id: 'pending-001',
    camera_id: 'cam-P',
    room_id: 'room-P',
    event_type: 'fall',
    severity: 'high' as const,
    confidence: 0.9,
    rationale: 'Pending fall incident.',
    state: 'alert' as const,
    created_at: new Date().toISOString(),
    evidence_clip_url: null,
    operator_decision: null,
  }

  function FeedbackTrigger() {
    const { updateDecision } = useAlerts()
    return (
      <button onClick={() => updateDecision('pending-001', 'confirmed')}>
        confirm
      </button>
    )
  }

  await act(async () => {
    render(
      <AlertProvider>
        <FeedbackTrigger />
        <AlertFeed initialIncidents={[pendingIncident]} />
      </AlertProvider>
    )
  })

  // Card visible initially — pending-only filter is on by default
  expect(screen.getByText(/room-P/i)).toBeInTheDocument()

  // Simulate operator confirming
  await act(async () => {
    fireEvent.click(screen.getByText('confirm'))
  })

  // Card removed — incident is no longer pending
  expect(screen.queryByText(/room-P/i)).toBeNull()
})
