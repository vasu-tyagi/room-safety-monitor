/**
 * Tests for AlertContext (Bug 9: alert state persists across page navigation).
 */
import { render, screen, act } from '@testing-library/react'
import { AlertProvider, useAlerts } from '@/context/AlertContext'
import type { Incident } from '@/types/incident'

const mockIncident: Incident = {
  id: 'test-001',
  camera_id: 'cam-1',
  room_id: 'room-A',
  event_type: 'fall',
  severity: 'high',
  confidence: 0.9,
  rationale: 'Test fall',
  state: 'alert',
  created_at: new Date().toISOString(),
  evidence_clip_url: null,
  operator_decision: null,
}

function AlertList() {
  const { alerts } = useAlerts()
  return <ul>{alerts.map((a) => <li key={a.id}>{a.room_id}</li>)}</ul>
}

function AddAlert() {
  const { addOrUpdateAlert } = useAlerts()
  return <button onClick={() => addOrUpdateAlert(mockIncident)}>add</button>
}

function MergeButton({ incidents }: { incidents: Incident[] }) {
  const { mergeInitial } = useAlerts()
  return <button onClick={() => mergeInitial(incidents)}>merge</button>
}

function TestApp({ showList }: { showList: boolean }) {
  return (
    <AlertProvider>
      <AddAlert />
      {showList && <AlertList />}
    </AlertProvider>
  )
}

describe('AlertContext', () => {
  it('starts with empty alerts list', () => {
    render(<AlertProvider><AlertList /></AlertProvider>)
    expect(screen.queryByRole('listitem')).toBeNull()
  })

  it('addOrUpdateAlert prepends incident to list', () => {
    render(<AlertProvider><AddAlert /><AlertList /></AlertProvider>)
    expect(screen.queryByText('room-A')).toBeNull()
    act(() => { screen.getByText('add').click() })
    expect(screen.getByText('room-A')).toBeInTheDocument()
  })

  it('addOrUpdateAlert updates existing incident without duplicating', () => {
    render(<AlertProvider><AddAlert /><AlertList /></AlertProvider>)
    act(() => { screen.getByText('add').click() })
    act(() => { screen.getByText('add').click() })
    expect(screen.getAllByText('room-A')).toHaveLength(1)
  })

  it('mergeInitial adds new incidents without overwriting existing', async () => {
    const extra: Incident = { ...mockIncident, id: 'extra-002', room_id: 'room-B' }
    render(<AlertProvider><AddAlert /><MergeButton incidents={[extra]} /><AlertList /></AlertProvider>)
    act(() => { screen.getByText('add').click() })
    act(() => { screen.getByText('merge').click() })
    expect(screen.getByText('room-A')).toBeInTheDocument()
    expect(screen.getByText('room-B')).toBeInTheDocument()
  })

  it('alert persists after consumer component unmounts and remounts (simulates page navigation)', () => {
    const { rerender } = render(<TestApp showList={true} />)

    // Add an alert (simulates WS message received while on home page)
    act(() => { screen.getByText('add').click() })
    expect(screen.getByText('room-A')).toBeInTheDocument()

    // Navigate away: AlertList unmounts, AlertProvider stays
    rerender(<TestApp showList={false} />)
    expect(screen.queryByText('room-A')).toBeNull()

    // Navigate back: AlertList remounts
    rerender(<TestApp showList={true} />)
    expect(screen.getByText('room-A')).toBeInTheDocument()
  })
})
