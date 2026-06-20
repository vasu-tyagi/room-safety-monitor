/**
 * Tests for Slice 8d: pipeline progress animation.
 * Covers PipelineProgressContext dispatch logic and LayerStatusBar dot states.
 */
import { render, screen, act } from '@testing-library/react'
import { PipelineProgressProvider, usePipelineProgress } from '@/context/PipelineProgressContext'
import { LayerStatusBar } from '@/components/LayerStatusBar'

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function DispatchButton({ layer, status }: { layer: string; status: string }) {
  const { dispatch } = usePipelineProgress()
  return (
    <button onClick={() => dispatch({ layer, status })}>
      {layer}:{status}
    </button>
  )
}

function StateDisplay() {
  const { state } = usePipelineProgress()
  return (
    <div>
      <span data-testid="state-layer">{state.layer ?? 'null'}</span>
      <span data-testid="state-status">{state.status}</span>
    </div>
  )
}

function TestHarness({ layer, status }: { layer: string; status: string }) {
  return (
    <PipelineProgressProvider>
      <StateDisplay />
      <DispatchButton layer={layer} status={status} />
    </PipelineProgressProvider>
  )
}

// ------------------------------------------------------------------
// Context tests
// ------------------------------------------------------------------

describe('PipelineProgressContext', () => {
  it('starts idle', () => {
    render(<TestHarness layer="L2" status="processing" />)
    expect(screen.getByTestId('state-layer').textContent).toBe('null')
    expect(screen.getByTestId('state-status').textContent).toBe('idle')
  })

  it('dispatching processing updates state', () => {
    render(<TestHarness layer="L3" status="processing" />)
    act(() => { screen.getByText('L3:processing').click() })
    expect(screen.getByTestId('state-layer').textContent).toBe('L3')
    expect(screen.getByTestId('state-status').textContent).toBe('processing')
  })

  it('dispatching complete updates state', () => {
    render(<TestHarness layer="L4" status="complete" />)
    act(() => { screen.getByText('L4:complete').click() })
    expect(screen.getByTestId('state-layer').textContent).toBe('L4')
    expect(screen.getByTestId('state-status').textContent).toBe('complete')
  })

  it('backward events are ignored', () => {
    // Dispatch L4 complete first, then try to go back to L2
    function BackwardTest() {
      const { dispatch, state } = usePipelineProgress()
      return (
        <div>
          <span data-testid="state-layer">{state.layer ?? 'null'}</span>
          <button onClick={() => dispatch({ layer: 'L4', status: 'complete' })}>L4</button>
          <button onClick={() => dispatch({ layer: 'L2', status: 'processing' })}>L2</button>
        </div>
      )
    }
    render(<PipelineProgressProvider><BackwardTest /></PipelineProgressProvider>)
    act(() => { screen.getByText('L4').click() })
    expect(screen.getByTestId('state-layer').textContent).toBe('L4')
    act(() => { screen.getByText('L2').click() })
    // State should still be L4 — backward event ignored
    expect(screen.getByTestId('state-layer').textContent).toBe('L4')
  })

  it('unknown layer id is ignored', () => {
    render(<TestHarness layer="UNKNOWN" status="processing" />)
    act(() => { screen.getByText('UNKNOWN:processing').click() })
    expect(screen.getByTestId('state-layer').textContent).toBe('null')
  })
})

// ------------------------------------------------------------------
// LayerStatusBar dot state tests
// ------------------------------------------------------------------

describe('LayerStatusBar pipeline animation', () => {
  function renderBar(layer: string, status: string) {
    function Setter() {
      const { dispatch } = usePipelineProgress()
      return (
        <button data-testid="trigger" onClick={() => dispatch({ layer, status })}>
          go
        </button>
      )
    }
    render(
      <PipelineProgressProvider>
        <Setter />
        <LayerStatusBar />
      </PipelineProgressProvider>
    )
    act(() => { screen.getByTestId('trigger').click() })
  }

  it('all dots are idle when pipeline is idle', () => {
    render(
      <PipelineProgressProvider>
        <LayerStatusBar />
      </PipelineProgressProvider>
    )
    // All 6 L-prefixed dots should exist
    const dots = document.querySelectorAll('[data-testid^="layer-L"]')
    expect(dots).toHaveLength(6)
  })

  it('shows pipeline label "Pipeline: idle" by default', () => {
    render(
      <PipelineProgressProvider>
        <LayerStatusBar />
      </PipelineProgressProvider>
    )
    expect(screen.getByTestId('pipeline-label').textContent).toBe('Pipeline: idle')
  })

  it('shows processing label when L3 is active', () => {
    renderBar('L3', 'processing')
    expect(screen.getByTestId('pipeline-label').textContent).toContain('processing')
    expect(screen.getByTestId('pipeline-label').textContent).toContain('L3')
  })

  it('shows "Pipeline: complete" on complete status', () => {
    renderBar('L5', 'complete')
    expect(screen.getByTestId('pipeline-label').textContent).toBe('Pipeline: complete')
  })

  it('LayerStatusBar snapshot (idle)', () => {
    const { container } = render(
      <PipelineProgressProvider>
        <LayerStatusBar />
      </PipelineProgressProvider>
    )
    expect(container).toMatchSnapshot()
  })
})
