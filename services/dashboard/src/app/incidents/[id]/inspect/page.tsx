import { notFound } from 'next/navigation'
import Link from 'next/link'
import { fetchIncident } from '@/lib/api'
import { Breadcrumb } from '@/components/Breadcrumb'
import { InspectorContent } from '@/components/InspectorContent'

export default async function InspectPage({
  params,
}: {
  params: { id: string }
}) {
  let incident
  try {
    incident = await fetchIncident(params.id)
  } catch {
    notFound()
  }

  const shortId = params.id.slice(0, 8)

  return (
    <div>
      <Breadcrumb
        crumbs={[
          { label: 'Live Feed', href: '/' },
          { label: `Incident ${shortId}`, href: `/incidents/${params.id}` },
          { label: 'Inspector' },
        ]}
      />

      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-medium text-zinc-50 capitalize">
            Inspector — {incident.event_type.replace(/_/g, ' ')}
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Per-layer trace for incident{' '}
            <span className="font-mono">{shortId}</span>
          </p>
        </div>
        <Link
          href={`/incidents/${params.id}`}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors mt-1"
        >
          ← Operator view
        </Link>
      </div>

      <InspectorContent incident={incident} />
    </div>
  )
}
