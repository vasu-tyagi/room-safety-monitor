import { render, screen, waitFor } from '@testing-library/react'

jest.mock('@/lib/api', () => ({
  fetchMetrics: jest.fn(),
}))

import { fetchMetrics } from '@/lib/api'
const mockFetch = fetchMetrics as jest.MockedFunction<typeof fetchMetrics>

import MetricsPage from '@/app/metrics/page'

beforeEach(() => mockFetch.mockReset())

test('MetricsPage renders stat card labels', async () => {
  mockFetch.mockResolvedValue({
    frames_processed_total: 1200,
    incidents_total: 5,
    alerts_last_hour: 2,
    gate_filter_rate: 0.75,
    kb_entry_count: 10,
    incidents_by_severity_24h: { high: 3, low: 2 },
  })
  render(<MetricsPage />)
  expect(screen.getByText('Frames processed')).toBeInTheDocument()
  expect(screen.getByText('Total incidents')).toBeInTheDocument()
  expect(screen.getByText('KB entries')).toBeInTheDocument()
  expect(screen.getByText('Alerts last hour')).toBeInTheDocument()
})

test('MetricsPage shows values returned by fetchMetrics', async () => {
  mockFetch.mockResolvedValue({
    frames_processed_total: 999,
    incidents_total: 7,
    alerts_last_hour: 1,
    gate_filter_rate: 0.6,
    kb_entry_count: 3,
    incidents_by_severity_24h: {},
  })
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
  })
  render(<MetricsPage />)
  await waitFor(() => {
    const noData = screen.getAllByText('No data yet')
    expect(noData.length).toBeGreaterThan(0)
  })
})
