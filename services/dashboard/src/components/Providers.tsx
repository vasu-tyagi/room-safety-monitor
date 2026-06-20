'use client'
import { PipelineProgressProvider } from '@/context/PipelineProgressContext'

export function Providers({ children }: { children: React.ReactNode }) {
  return <PipelineProgressProvider>{children}</PipelineProgressProvider>
}
