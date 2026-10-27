import { MeetingDetail } from "@/components/meeting/meeting-detail";

export default async function MeetingPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ utt?: string }>;
}) {
  const { id } = await params;
  const { utt } = await searchParams;
  const focusUtt = utt !== undefined && /^\d+$/.test(utt) ? Number(utt) : null;
  return <MeetingDetail id={id} focusUtt={focusUtt} />;
}
