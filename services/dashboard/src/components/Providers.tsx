'use client'
import { PipelineProgressProvider } from '@/context/PipelineProgressContext'
import { AlertProvider } from '@/context/AlertContext'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <PipelineProgressProvider>
      <AlertProvider>
        {children}
      </AlertProvider>
    </PipelineProgressProvider>
  )
}
