import { render, screen } from '@testing-library/react'
import { ArchitectureContent } from '@/components/ArchitectureContent'
import type { LayerInfo } from '@/lib/api'

const LAYERS: LayerInfo[] = [
  {
    id: 'L1', name: 'Stream Ingest', status: 'not_built',
    description: 'File-based ingest.',
    components: [{ name: 'Video source', real: 'cv2.VideoCapture', production: 'RTSP' }],
  },
  {
    id: 'L2', name: 'Fast CV', status: 'approximated',
    description: 'Models on CPU.',
    components: [{ name: 'Detection', real: 'YOLOv8n', production: 'TensorRT' }],
  },
  {
    id: 'gate', name: 'Event Gate', status: 'real',
    description: 'Deterministic rules.',
    components: [{ name: 'Rules engine', real: 'engine.py', production: 'Same' }],
  },
  {
    id: 'L3', name: 'VLM Deep Analysis', status: 'real',
    description: 'Qwen 2.5 VL.',
    components: [{ name: 'Model', real: 'Qwen2.5-VL-72B', production: 'Same' }],
  },
  {
    id: 'L4', name: 'AI Agent', status: 'real',
    description: 'LangGraph agent.',
    components: [{ name: 'Framework', real: 'LangGraph', production: 'Same' }],
  },
  {
    id: 'L5', name: 'Persistence + KB', status: 'partial',
    description: 'Postgres + pgvector.',
    components: [{ name: 'DB', real: 'Postgres', production: 'Same' }],
  },
  {
    id: 'L6', name: 'Service Plane', status: 'partial',
    description: 'FastAPI + WebSocket.',
    components: [{ name: 'REST API', real: 'FastAPI', production: 'Same' }],
  },
]

test('ArchitectureContent renders all layer names', () => {
  render(<ArchitectureContent layers={LAYERS} />)
  expect(screen.getByText('Stream Ingest')).toBeInTheDocument()
  expect(screen.getByText('Fast CV')).toBeInTheDocument()
  expect(screen.getByText('Event Gate')).toBeInTheDocument()
  expect(screen.getByText('VLM Deep Analysis')).toBeInTheDocument()
  expect(screen.getByText('AI Agent')).toBeInTheDocument()
  expect(screen.getByText('Persistence + KB')).toBeInTheDocument()
  expect(screen.getByText('Service Plane')).toBeInTheDocument()
})

test('ArchitectureContent renders status badges', () => {
  render(<ArchitectureContent layers={LAYERS} />)
  expect(screen.getAllByText('Real').length).toBeGreaterThan(0)
  expect(screen.getByText('Approximated')).toBeInTheDocument()
  expect(screen.getByText('Not built')).toBeInTheDocument()
  expect(screen.getAllByText('Partial').length).toBeGreaterThan(0)
})

test('ArchitectureContent snapshot', () => {
  const { container } = render(<ArchitectureContent layers={LAYERS} />)
  expect(container).toMatchSnapshot()
})
