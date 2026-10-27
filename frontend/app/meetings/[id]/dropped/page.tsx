import { DroppedView } from "@/components/meeting/dropped-view";

export default async function DroppedPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <DroppedView id={id} />;
}
