import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import { NavBar } from '@/components/NavBar'
import { LayerStatusBar } from '@/components/LayerStatusBar'
import { Providers } from '@/components/Providers'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  weight: ['400', '500'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Room Safety Monitor',
  description: 'Operator alert dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`dark ${inter.variable} ${mono.variable}`}>
      <body className="bg-zinc-950 text-zinc-100 min-h-screen antialiased font-sans">
        <Providers>
          <NavBar />
          <LayerStatusBar />
          <main className="mx-auto max-w-content px-6 md:px-8 py-8">{children}</main>
        </Providers>
      </body>
    </html>
  )
}
