'use client'
import { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { ProvenanceDisclosure } from '@/components/ProvenanceDisclosure'
import { usePipelineProgress, PIPELINE_CASCADE } from '@/context/PipelineProgressContext'
import type { LayerInfo } from '@/lib/api'

const LAYERS: LayerInfo[] = [
  {
    id: 'L1', name: 'Stream Ingest', status: 'not_built',
    description: 'Ingest is a file path passed to process_video. Full RTSP ingest is deferred to Slice 9.',
    components: [
      { name: 'Video source', real: 'cv2.VideoCapture(file_path)', production: 'Multicast RTSP receiver' },
      { name: 'Decode', real: 'OpenCV CPU decode', production: 'NVDEC H.264' },
    ],
  },
  {
    id: 'L2', name: 'Fast CV', status: 'approximated',
    description: 'Real models running in-process on CPU. No Triton/TensorRT; latency target not met.',
    components: [
      { name: 'Detection', real: 'YOLOv8n via ultralytics', production: 'YOLOv8 on TensorRT' },
      { name: 'Pose', real: 'RTMPose via rtmlib/ONNX', production: 'RTMPose on TensorRT' },
      { name: 'Action', real: 'SlowFast R50', production: 'SlowFast on TensorRT' },
    ],
  },
  {
    id: 'gate', name: 'Event Gate', status: 'real',
    description: 'Seven deterministic rules over L2 outputs. Room policies in config/rooms.yaml.',
    components: [
      { name: 'Rules engine', real: 'services/event_gate/engine.py', production: 'Same' },
      { name: 'Room policies', real: 'config/rooms.yaml', production: 'Same' },
    ],
  },
  {
    id: 'L3', name: 'VLM', status: 'real',
    description: 'Real Qwen 2.5 VL 72B via HF Inference Providers. Stub fallback via VLM_MODE env var.',
    components: [
      { name: 'Model', real: 'Qwen2.5-VL-72B-Instruct', production: 'Same or self-hosted' },
      { name: 'KB context', real: 'Top-3 pgvector matches', production: 'Same' },
    ],
  },
  {
    id: 'L4', name: 'Agent', status: 'real',
    description: 'LangGraph 6-node StateGraph with policy engine, confidence fusion, incident FSM.',
    components: [
      { name: 'Framework', real: 'LangGraph StateGraph', production: 'Same' },
      { name: 'Policy', real: 'YAML rules in config/policies/', production: 'Same' },
    ],
  },
  {
    id: 'L5', name: 'Persistence', status: 'partial',
    description: 'Postgres + pgvector are real. Evidence clips save to local filesystem; MinIO deferred.',
    components: [
      { name: 'Incidents DB', real: 'Postgres + SQLAlchemy', production: 'Same' },
      { name: 'KB', real: 'pgvector + sentence-transformers', production: 'Same' },
      { name: 'Clips', real: 'Local filesystem (clips/)', production: 'MinIO' },
    ],
  },
  {
    id: 'L6', name: 'Service Plane', status: 'partial',
    description: 'FastAPI REST + WebSocket are real. Redis pub/sub is documented but not wired.',
    components: [
      { name: 'REST API', real: 'FastAPI', production: 'Same' },
      { name: 'WebSocket', real: 'In-memory ConnectionManager', production: 'Redis pub/sub' },
      { name: 'Dashboard', real: 'Built (Slice 8b/8c) — Next.js 14', production: 'Next.js 14' },
    ],
  },
]

const LAYER_NAME = Object.fromEntries(LAYERS.map((l) => [l.id, l.name]))

type DotClass = 'idle' | 'processing' | 'success'

const DOT: Record<DotClass, string> = {
  idle:       'bg-zinc-600',
  processing: 'bg-blue-500 animate-pulse',
  success:    'bg-emerald-500',
}

function dotClass(layerId: string, activeLayer: string | null, activeStatus: string): DotClass {
  if (!activeLayer || activeStatus === 'idle') return 'idle'
  const activeIdx = PIPELINE_CASCADE.indexOf(activeLayer as typeof PIPELINE_CASCADE[number])
  const thisIdx   = PIPELINE_CASCADE.indexOf(layerId as typeof PIPELINE_CASCADE[number])
  if (activeIdx === -1 || thisIdx === -1) return 'idle'
  if (thisIdx < activeIdx) return 'success'
  if (thisIdx === activeIdx) return activeStatus === 'processing' ? 'processing' : 'success'
  return 'idle'
}

function statusLabel(layer: string | null, status: string): string {
  if (!layer || status === 'idle') return 'Pipeline: idle'
  if (status === 'complete') return 'Pipeline: complete'
  return `Pipeline: processing ${layer} (${LAYER_NAME[layer] ?? layer})`
}

export function LayerStatusBar({ states = {} }: { states?: Record<string, string> }) {
  const [selected, setSelected] = useState<LayerInfo | null>(null)
  const { state } = usePipelineProgress()

  const label = statusLabel(state.layer, state.status)

  return (
    <>
      <div
        className="bg-zinc-950 border-b border-zinc-800/60"
        data-testid="layer-status-bar"
        aria-label="Layer status"
      >
        <div className="mx-auto max-w-content px-6 md:px-8">
          <div className="flex items-center gap-3 py-2">
            <span
              className="text-xs text-zinc-600 shrink-0 transition-colors duration-300"
              data-testid="pipeline-label"
            >
              {label}
            </span>
            <div className="w-px h-3 bg-zinc-800 shrink-0" />
            <div className="flex items-center gap-1">
              {LAYERS.map((layer, i) => {
                const dotCls = dotClass(layer.id, state.layer, state.status)
                const overrideState = states[layer.id]
                const finalDotCls: DotClass =
                  overrideState === 'processing' ? 'processing'
                  : overrideState === 'success' ? 'success'
                  : dotCls
                return (
                  <span key={layer.id} className="flex items-center gap-1">
                    <button
                      onClick={() => setSelected(layer)}
                      className="group/layer relative flex items-center gap-1.5 px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 hover:border-zinc-700 transition-colors duration-150"
                      data-testid={`layer-${layer.id}`}
                      aria-label={`${layer.name} status`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 transition-colors duration-300 ${DOT[finalDotCls]}`}
                      />
                      <span>{layer.id}</span>
                    </button>
                    {i < LAYERS.length - 1 && (
                      <ChevronRight className="w-3 h-3 text-zinc-700 shrink-0" />
                    )}
                  </span>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {selected && (
        <ProvenanceDisclosure layer={selected} onClose={() => setSelected(null)} />
      )}
    </>
  )
}
