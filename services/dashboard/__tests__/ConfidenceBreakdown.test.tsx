import { render, screen } from '@testing-library/react'
import { ConfidenceBreakdown } from '@/components/ConfidenceBreakdown'

const BREAKDOWN = { yolo: 0.85, pose: 0.72, vlm: 0.9 }

test('ConfidenceBreakdown renders all source labels', () => {
  render(<ConfidenceBreakdown breakdown={BREAKDOWN} fused={0.82} severity="high" />)
  expect(screen.getByText('yolo')).toBeInTheDocument()
  expect(screen.getByText('pose')).toBeInTheDocument()
  expect(screen.getByText('vlm')).toBeInTheDocument()
})

test('ConfidenceBreakdown renders fused confidence label', () => {
  render(<ConfidenceBreakdown breakdown={BREAKDOWN} fused={0.82} severity="high" />)
  expect(screen.getByText('Fused confidence')).toBeInTheDocument()
})

test('ConfidenceBreakdown snapshot', () => {
  const { container } = render(
    <ConfidenceBreakdown breakdown={BREAKDOWN} fused={0.82} severity="high" />
  )
  expect(container).toMatchSnapshot()
})
