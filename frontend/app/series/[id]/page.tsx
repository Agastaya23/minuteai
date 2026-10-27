import { SeriesView } from "@/components/series/series-view";

export default async function SeriesDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <SeriesView id={id} />;
}
