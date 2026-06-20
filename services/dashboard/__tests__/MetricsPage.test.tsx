import { render, screen, waitFor } from '@testing-library/react'

jest.mock('@/lib/api', () => ({
  fetchMetrics: jest.fn(),
}))

import { fetchMetrics } from '@/lib/api'
const mockFetch = fetchMetrics as jest.MockedFunction<typeof fetchMetrics>

import MetricsPage from '@/app/metrics/page'

beforeEach(() => mockFetch.mockReset())

const BASE_METRICS = {
  frames_processed_total: 1200,
  incidents_total: 5,
  alerts_last_hour: 2,
  gate_filter_rate: 0.75,
  kb_entry_count: 10,
  incidents_by_severity_24h: { high: 3, low: 2 },
  operator_decisions: { confirmed: 3, dismissed: 1, pending: 1 },
}

test('MetricsPage renders stat card labels', async () => {
  mockFetch.mockResolvedValue(BASE_METRICS)
  render(<MetricsPage />)
  expect(screen.getByText('Frames processed')).toBeInTheDocument()
  expect(screen.getByText('Total incidents')).toBeInTheDocument()
  expect(screen.getByText('KB entries')).toBeInTheDocument()
  expect(screen.getByText('Alerts last hour')).toBeInTheDocument()
})

test('MetricsPage shows values returned by fetchMetrics', async () => {
  mockFetch.mockResolvedValue({ ...BASE_METRICS, frames_processed_total: 999 })
  render(<MetricsPage />)
  await waitFor(() => expect(screen.getByText('999')).toBeInTheDocument())
})

test('MetricsPage shows "No data yet" when values are null', async () => {
  mockFetch.mockResolvedValue({
    frames_processed_total: null,
    incidents_total: null,
    alerts_last_hour: null,
    gate_filter_rate: null,
    kb_entry_count: null,
    incidents_by_severity_24h: null,
    operator_decisions: null,
  })
  render(<MetricsPage />)
  await waitFor(() => {
    const noData = screen.getAllByText('No data yet')
    expect(noData.length).toBeGreaterThan(0)
  })
})

test('MetricsPage does not contain Latency placeholder', async () => {
  mockFetch.mockResolvedValue(BASE_METRICS)
  render(<MetricsPage />)
  expect(screen.queryByText(/Not yet instrumented/i)).toBeNull()
  expect(screen.queryByText('Latency')).toBeNull()
})

test('DecisionBar shows stacked bar and counts when decisions exist', async () => {
  mockFetch.mockResolvedValue({
    ...BASE_METRICS,
    operator_decisions: { confirmed: 4, dismissed: 2, pending: 1 },
  })
  render(<MetricsPage />)
  await waitFor(() => {
    expect(screen.getByText(/4 of 6 handled/i)).toBeInTheDocument()
  })
  expect(screen.getByText('confirmed')).toBeInTheDocument()
  expect(screen.getByText('dismissed')).toBeInTheDocument()
  expect(screen.getByText('pending')).toBeInTheDocument()
})

test('DecisionBar shows "No decisions yet" when confirmed + dismissed = 0', async () => {
  mockFetch.mockResolvedValue({
    ...BASE_METRICS,
    operator_decisions: { confirmed: 0, dismissed: 0, pending: 7 },
  })
  render(<MetricsPage />)
  await waitFor(() => {
    expect(screen.getByText('No decisions yet')).toBeInTheDocument()
  })
})
