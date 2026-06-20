'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { Shield, Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'

const NAV_LINKS = [
  { label: 'Live Feed',    href: '/' },
  { label: 'History',      href: '/history' },
  { label: 'Metrics',      href: '/metrics' },
  { label: 'Architecture', href: '/architecture' },
]

export function NavBar() {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

  function isActive(href: string) {
    if (href === '/') return pathname === '/'
    return pathname.startsWith(href)
  }

  return (
    <header className="sticky top-0 z-50 bg-zinc-950 border-b border-zinc-800">
      <div className="mx-auto max-w-content px-6 md:px-8">
        <div className="flex items-stretch h-14 gap-8">

          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 shrink-0 text-zinc-50 hover:text-white transition-colors"
          >
            <Shield className="w-4 h-4 text-blue-500" strokeWidth={2.5} />
            <span className="text-sm font-medium tracking-tight">Room Safety Monitor</span>
          </Link>

          {/* Desktop nav links — flush with header bottom via border-b */}
          <nav className="hidden md:flex items-stretch flex-1 gap-1" aria-label="Main">
            {NAV_LINKS.map(({ label, href }) => (
              <Link
                key={href}
                href={href}
                className={cn(
                  'flex items-center px-3 text-sm font-medium border-b-2 transition-colors duration-150',
                  isActive(href)
                    ? 'border-blue-500 text-zinc-50'
                    : 'border-transparent text-zinc-400 hover:text-zinc-50 hover:border-zinc-600'
                )}
              >
                {label}
              </Link>
            ))}
          </nav>

          {/* System status — desktop right */}
          <div className="hidden md:flex items-center gap-2 ml-auto">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
            <span className="text-xs text-zinc-500">All systems operational</span>
          </div>

          {/* Hamburger — mobile only */}
          <button
            className="ml-auto md:hidden flex items-center p-1.5 rounded text-zinc-400 hover:text-zinc-50 hover:bg-zinc-800 transition-colors"
            aria-label="Toggle menu"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile dropdown */}
        {open && (
          <nav
            className="md:hidden flex flex-col gap-0.5 border-t border-zinc-800 py-2"
            aria-label="Main (mobile)"
          >
            {NAV_LINKS.map(({ label, href }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={cn(
                  'px-3 py-2.5 rounded text-sm font-medium transition-colors',
                  isActive(href)
                    ? 'bg-zinc-800 text-zinc-50'
                    : 'text-zinc-400 hover:text-zinc-50 hover:bg-zinc-800/60'
                )}
              >
                {label}
              </Link>
            ))}
          </nav>
        )}
      </div>
    </header>
  )
}
