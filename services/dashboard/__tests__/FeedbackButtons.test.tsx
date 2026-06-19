import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { FeedbackButtons } from '@/app/incidents/[id]/FeedbackButtons'

// Mock Next.js router
jest.mock('next/navigation', () => ({ useRouter: () => ({ refresh: jest.fn() }) }))

// Mock the API module
jest.mock('@/lib/api', () => ({
  submitFeedback: jest.fn(),
}))

import { submitFeedback } from '@/lib/api'
const mockSubmit = submitFeedback as jest.MockedFunction<typeof submitFeedback>

describe('FeedbackButtons', () => {
  beforeEach(() => mockSubmit.mockReset())

  it('shows existing decision instead of buttons', () => {
    render(<FeedbackButtons incidentId="id1" currentDecision="confirmed" />)
    expect(screen.queryByRole('button')).toBeNull()
    expect(screen.getByText('confirmed')).toBeInTheDocument()
  })

  it('shows optimistic decision immediately on success', async () => {
    mockSubmit.mockResolvedValueOnce({ status: 'ok', operator_decision: 'dismissed' } as never)
    render(<FeedbackButtons incidentId="id1" currentDecision={null} />)
    fireEvent.click(screen.getByText('Dismiss'))
    await waitFor(() => expect(screen.getByText('dismissed')).toBeInTheDocument())
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('shows error message on failure', async () => {
    mockSubmit.mockRejectedValueOnce(new Error('Network error'))
    render(<FeedbackButtons incidentId="id1" currentDecision={null} />)
    fireEvent.click(screen.getByText('Confirm Alert'))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert').textContent).toContain('Network error')
    // Buttons are still present so the user can retry
    expect(screen.getAllByRole('button').length).toBe(2)
  })
})
