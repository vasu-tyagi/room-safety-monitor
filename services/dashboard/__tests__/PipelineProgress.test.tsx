/**
 * Tests for Slice 8d/8e: pipeline progress animation.
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
  const { layerStates } = usePipelineProgress()
  return (
    <div>
      {Object.entries(layerStates).map(([layer, status]) => (
        <span key={layer} data-testid={`state-${layer}`}>{status}</span>
      ))}
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
  it('starts with all layers idle', () => {
    render(<TestHarness layer="L2" status="processing" />)
    expect(screen.getByTestId('state-L1').textContent).toBe('idle')
    expect(screen.getByTestId('state-L2').textContent).toBe('idle')
    expect(screen.getByTestId('state-L3').textContent).toBe('idle')
  })

  it('dispatching processing marks that layer as processing', () => {
    render(<TestHarness layer="L3" status="processing" />)
    act(() => { screen.getByText('L3:processing').click() })
    expect(screen.getByTestId('state-L3').textContent).toBe('processing')
    expect(screen.getByTestId('state-L1').textContent).toBe('idle')
    expect(screen.getByTestId('state-L2').textContent).toBe('idle')
  })

  it('dispatching complete marks that layer as complete', () => {
    render(<TestHarness layer="L4" status="complete" />)
    act(() => { screen.getByText('L4:complete').click() })
    expect(screen.getByTestId('state-L4').textContent).toBe('complete')
  })

  it('per-layer states accumulate: L1 complete then L2 processing both visible', () => {
    function MultiDispatch() {
      const { dispatch } = usePipelineProgress()
      return (
        <div>
          <button onClick={() => dispatch({ layer: 'L1', status: 'complete' })}>L1-done</button>
          <button onClick={() => dispatch({ layer: 'L2', status: 'processing' })}>L2-start</button>
        </div>
      )
    }
    render(
      <PipelineProgressProvider>
        <StateDisplay />
        <MultiDispatch />
      </PipelineProgressProvider>
    )
    act(() => { screen.getByText('L1-done').click() })
    expect(screen.getByTestId('state-L1').textContent).toBe('complete')
    act(() => { screen.getByText('L2-start').click() })
    // L1 stays complete while L2 is processing
    expect(screen.getByTestId('state-L1').textContent).toBe('complete')
    expect(screen.getByTestId('state-L2').textContent).toBe('processing')
  })

  it('L1 complete on a new run resets all previously active layers', () => {
    function MultiDispatch() {
      const { dispatch } = usePipelineProgress()
      return (
        <div>
          <button onClick={() => dispatch({ layer: 'L3', status: 'complete' })}>L3-done</button>
          <button onClick={() => dispatch({ layer: 'L1', status: 'complete' })}>L1-new-run</button>
        </div>
      )
    }
    render(
      <PipelineProgressProvider>
        <StateDisplay />
        <MultiDispatch />
      </PipelineProgressProvider>
    )
    act(() => { screen.getByText('L3-done').click() })
    expect(screen.getByTestId('state-L3').textContent).toBe('complete')
    // New run: L1 fires → resets all layers, then sets L1 complete
    act(() => { screen.getByText('L1-new-run').click() })
    expect(screen.getByTestId('state-L3').textContent).toBe('idle')
    expect(screen.getByTestId('state-L1').textContent).toBe('complete')
  })

  it('unknown layer id is ignored', () => {
    render(<TestHarness layer="UNKNOWN" status="processing" />)
    act(() => { screen.getByText('UNKNOWN:processing').click() })
    expect(screen.getByTestId('state-L1').textContent).toBe('idle')
    expect(screen.queryByTestId('state-UNKNOWN')).toBeNull()
  })

  it('reset() clears all layer states to idle', () => {
    function ResetHarness() {
      const { dispatch, reset } = usePipelineProgress()
      return (
        <div>
          <button onClick={() => dispatch({ layer: 'L2', status: 'complete' })}>set</button>
          <button onClick={reset}>reset</button>
        </div>
      )
    }
    render(
      <PipelineProgressProvider>
        <StateDisplay />
        <ResetHarness />
      </PipelineProgressProvider>
    )
    act(() => { screen.getByText('set').click() })
    expect(screen.getByTestId('state-L2').textContent).toBe('complete')
    act(() => { screen.getByText('reset').click() })
    expect(screen.getByTestId('state-L2').textContent).toBe('idle')
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

  it('shows "Pipeline: complete" when any layer is complete and none processing', () => {
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
