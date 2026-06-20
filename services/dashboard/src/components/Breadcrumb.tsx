import Link from 'next/link'

interface Crumb {
  label: string
  href?: string
}

export function Breadcrumb({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-xs text-zinc-500 mb-6">
      {crumbs.map((crumb, i) => (
        <span key={crumb.label} className="flex items-center gap-1.5">
          {i > 0 && (
            <span className="text-zinc-700" aria-hidden="true">/</span>
          )}
          {crumb.href ? (
            <Link href={crumb.href} className="hover:text-zinc-300 transition-colors duration-150">
              {crumb.label}
            </Link>
          ) : (
            <span className="text-zinc-300">{crumb.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}
