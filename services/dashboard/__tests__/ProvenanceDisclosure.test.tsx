import { render, screen, fireEvent } from '@testing-library/react'
import { ProvenanceDisclosure } from '@/components/ProvenanceDisclosure'
import type { LayerInfo } from '@/lib/api'

const LAYER: LayerInfo = {
  id: 'L2',
  name: 'Fast CV',
  status: 'approximated',
  description: 'Real models on CPU.',
  components: [
    { name: 'Detection', real: 'YOLOv8n', production: 'GPU-optimized' },
    { name: 'Pose', real: 'RTMPose', production: 'GPU-optimized' },
  ],
}

test('ProvenanceDisclosure renders layer name and status', () => {
  render(<ProvenanceDisclosure layer={LAYER} onClose={jest.fn()} />)
  expect(screen.getByText('Fast CV')).toBeInTheDocument()
  expect(screen.getByText('Approximated')).toBeInTheDocument()
})

test('ProvenanceDisclosure renders component rows', () => {
  render(<ProvenanceDisclosure layer={LAYER} onClose={jest.fn()} />)
  expect(screen.getByText('Detection')).toBeInTheDocument()
  expect(screen.getByText('YOLOv8n')).toBeInTheDocument()
  expect(screen.getAllByText('GPU-optimized')).toHaveLength(2)
})

test('ProvenanceDisclosure calls onClose when close button clicked', () => {
  const onClose = jest.fn()
  render(<ProvenanceDisclosure layer={LAYER} onClose={onClose} />)
  fireEvent.click(screen.getByLabelText('Close'))
  expect(onClose).toHaveBeenCalledTimes(1)
})

test('ProvenanceDisclosure snapshot', () => {
  const { container } = render(<ProvenanceDisclosure layer={LAYER} onClose={jest.fn()} />)
  expect(container).toMatchSnapshot()
})
