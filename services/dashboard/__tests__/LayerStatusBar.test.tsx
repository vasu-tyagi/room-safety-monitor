import { render, screen } from '@testing-library/react'
import { LayerStatusBar } from '@/components/LayerStatusBar'

test('LayerStatusBar renders 6 layer dots', () => {
  const { container } = render(<LayerStatusBar />)
  const dots = container.querySelectorAll('[data-testid^="layer-L"]')
  expect(dots).toHaveLength(6)
})

test('LayerStatusBar snapshot', () => {
  const { container } = render(<LayerStatusBar />)
  expect(container).toMatchSnapshot()
})
