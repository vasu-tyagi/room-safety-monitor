import { render, screen, fireEvent, waitFor } from '@testing-library/react'

jest.mock('next/navigation', () => ({ useRouter: () => ({ refresh: jest.fn() }) }))
jest.mock('@/lib/api', () => ({ submitFeedback: jest.fn() }))

import { submitFeedback } from '@/lib/api'
const mockSubmit = submitFeedback as jest.MockedFunction<typeof submitFeedback>

import { FeedbackButtons } from '@/app/incidents/[id]/FeedbackButtons'

beforeEach(() => mockSubmit.mockReset())

test('clicking Confirm Alert calls submitFeedback with confirm decision', async () => {
  mockSubmit.mockResolvedValueOnce({ status: 'ok', operator_decision: 'confirmed' } as never)
  render(<FeedbackButtons incidentId="abc-123" currentDecision={null} />)

  fireEvent.click(screen.getByText('Confirm Alert'))

  await waitFor(() => expect(mockSubmit).toHaveBeenCalledTimes(1))
  expect(mockSubmit).toHaveBeenCalledWith('abc-123', 'confirm')
})

test('clicking Dismiss calls submitFeedback with dismiss decision', async () => {
  mockSubmit.mockResolvedValueOnce({ status: 'ok', operator_decision: 'dismissed' } as never)
  render(<FeedbackButtons incidentId="abc-123" currentDecision={null} />)

  fireEvent.click(screen.getByText('Dismiss'))

  await waitFor(() => expect(mockSubmit).toHaveBeenCalledTimes(1))
  expect(mockSubmit).toHaveBeenCalledWith('abc-123', 'dismiss')
})

test('after confirm decision displays as lowercase confirmed', async () => {
  mockSubmit.mockResolvedValueOnce({ status: 'ok', operator_decision: 'confirmed' } as never)
  render(<FeedbackButtons incidentId="abc-123" currentDecision={null} />)

  fireEvent.click(screen.getByText('Confirm Alert'))

  await waitFor(() => expect(screen.getByText('confirmed')).toBeInTheDocument())
  expect(screen.queryByRole('button')).toBeNull()
})
