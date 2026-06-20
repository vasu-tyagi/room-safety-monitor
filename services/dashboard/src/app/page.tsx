import { fetchIncidents } from '@/lib/api'
import { AlertFeed } from '@/components/AlertFeed'

export const dynamic = 'force-dynamic'

export default async function HomePage() {
  const from = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
  const initialIncidents = await fetchIncidents({ limit: '100', from }).catch(() => [])
  return <AlertFeed initialIncidents={initialIncidents} />
}
