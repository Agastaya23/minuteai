"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Filter, Loader2, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo } from "react";
import { toast } from "sonner";

import { StatusBadge } from "@/components/status-badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useMediaQuery } from "@/hooks/use-media-query";
import { useMeeting } from "@/hooks/use-meetings";
import { api } from "@/lib/api";
import { fmtDate, fmtDuration } from "@/lib/format";
import type { Meeting, MeetingStatus, Transcript } from "@/lib/types";
import { cn } from "@/lib/utils";

import { ArtifactsPane } from "./artifacts-pane";
import { AudioPlayer } from "./audio-player";
import { EvidenceProvider, useEvidence } from "./evidence-context";
import { TranscriptPane } from "./transcript-pane";

const STEPS: MeetingStatus[] = ["uploaded", "transcribing", "extracting", "ready"];
const STEP_COPY: Record<string, string> = {
  uploaded: "Audio uploaded",
  transcribing: "Transcribing with word-level timestamps and identifying speakers",
  extracting: "Extracting minutes and checking every claim against the transcript",
  ready: "Ready",
};

function Progress({ meeting }: { meeting: Meeting }) {
  const qc = useQueryClient();
  const retry = useMutation({
    mutationFn: () => api.processMeeting(meeting.meeting_id, false),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["meeting", meeting.meeting_id] }),
    onError: (e: Error) => toast.error(e.message),
  });
  const current = STEPS.indexOf(meeting.status);
  return (
    <div className="mx-auto mt-16 max-w-lg rounded-xl border bg-card p-6">
      {meeting.status === "failed" ? (
        <>
          <p className="font-serif text-2xl">Processing failed</p>
          <p className="mt-2 rounded-md bg-destructive/10 p-3 font-mono text-xs break-words text-destructive">{meeting.error}</p>
          <Button className="mt-4" onClick={() => retry.mutate()} disabled={retry.isPending}>
            <RefreshCw /> Retry
          </Button>
        </>
      ) : (
        <ol className="flex flex-col gap-4">
          {STEPS.map((s, i) => (
            <li key={s} className={cn("flex items-center gap-3 text-sm", i > current && "text-muted-foreground/60")}>
              <span
                className={cn(
                  "flex size-6 items-center justify-center rounded-full border text-xs",
                  i < current && "border-transparent bg-success text-white",
                  i === current && "border-primary text-primary",
                )}
              >
                {i === current && s !== "ready" ? <Loader2 className="size-3.5 animate-spin" /> : i + 1}
              </span>
              {STEP_COPY[s]}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** Deep link support: /meetings/:id?utt=N selects that utterance once data is loaded. */
function FocusOnLoad({ transcript, uttId }: { transcript: Transcript; uttId: number }) {
  const { select } = useEvidence();
  useEffect(() => {
    const u = transcript.utterances.find((x) => x.utt_id === uttId);
    if (!u) return;
    const t = setTimeout(
      () =>
        select(`deeplink:${uttId}`, [
          { utt_id: u.utt_id, start_sec: u.start, end_sec: u.end, speaker_label: u.speaker_label, quote: u.text },
        ]),
      150,
    );
    return () => clearTimeout(t);
  }, [transcript, uttId, select]);
  return null;
}

export function MeetingDetail({ id, focusUtt = null }: { id: string; focusUtt?: number | null }) {
  const meeting = useMeeting(id);
  const desktop = useMediaQuery("(min-width: 1024px)", true);
  const ready = meeting.data?.status === "ready";
  const transcript = useQuery({ queryKey: ["transcript", id], queryFn: () => api.getTranscript(id), enabled: ready });
  const summary = useQuery({ queryKey: ["summary", id], queryFn: () => api.getSummary(id), enabled: ready });
  const dropped = useQuery({ queryKey: ["dropped", id], queryFn: () => api.getDropped(id), enabled: ready });

  const names = useMemo(() => {
    const m = new Map<string, string>();
    transcript.data?.utterances.forEach((u) => u.speaker_name && m.set(u.speaker_label, u.speaker_name));
    return m;
  }, [transcript.data]);

  if (meeting.isLoading) {
    return (
      <div className="mx-auto max-w-[1600px] p-6">
        <Skeleton className="h-10 w-80" />
      </div>
    );
  }
  if (!meeting.data) {
    return <p className="p-10 text-center text-muted-foreground">Meeting not found.</p>;
  }
  const m = meeting.data;

  return (
    <EvidenceProvider>
      <div className="mx-auto flex max-w-[1600px] flex-col px-4 pt-5 sm:px-6 lg:h-[calc(100dvh-3.5rem)]">
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <Link href="/" className={buttonVariants({ variant: "ghost", size: "icon" })} aria-label="Back">
            <ArrowLeft />
          </Link>
          <div className="min-w-0">
            <h1 className="truncate font-serif text-3xl leading-tight">{m.title}</h1>
            <p className="text-xs text-muted-foreground">
              {fmtDate(m.occurred_at, { weekday: "short", day: "numeric", month: "short", year: "numeric" })} ·{" "}
              {fmtDuration(m.duration_sec)}
              {m.participants.length > 0 && <> · {m.participants.join(", ")}</>}
              {m.series_id && (
                <>
                  {" · "}
                  <Link href={`/series/${m.series_id}`} className="underline-offset-2 hover:underline">
                    series {m.series_id}
                  </Link>
                </>
              )}
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <StatusBadge status={m.status} />
            {ready && (
              <Link href={`/meetings/${id}/dropped`} className={buttonVariants({ variant: "outline", size: "sm" })}>
                <Filter /> Filtered {dropped.data ? `(${dropped.data.length})` : ""}
              </Link>
            )}
          </div>
        </div>

        {!ready && <Progress meeting={m} />}

        {ready && (!transcript.data || !summary.data) && (
          <div className="grid gap-4 lg:grid-cols-3">
            <Skeleton className="h-96" />
            <Skeleton className="h-96" />
            <Skeleton className="h-40" />
          </div>
        )}

        {ready && transcript.data && summary.data && (
          <>
            {focusUtt !== null && <FocusOnLoad transcript={transcript.data} uttId={focusUtt} />}
            {desktop ? (
            <div className="grid min-h-0 flex-1 gap-5 pb-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_300px] xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_340px]">
              <section className="flex min-h-0 flex-col rounded-xl border bg-card">
                <h2 className="border-b px-4 py-2.5 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                  Transcript · {transcript.data.utterances.length} utterances
                </h2>
                <div className="min-h-0 flex-1 p-2">
                  <TranscriptPane transcript={transcript.data} />
                </div>
              </section>
              <section className="flex min-h-0 flex-col rounded-xl border bg-card p-3">
                <ArtifactsPane summary={summary.data} names={names} />
              </section>
              <aside className="flex flex-col gap-4">
                <div className="sticky top-20">
                  <AudioPlayer meetingId={id} transcript={transcript.data} />
                  <p className="mt-3 px-1 text-[11px] leading-relaxed text-muted-foreground">
                    Extracted by {summary.data.model_version} · transcribed by {transcript.data.model_version}
                  </p>
                </div>
              </aside>
            </div>
            ) : (
            <div>
              <div className="sticky top-14 z-30 -mx-4 bg-background/95 px-4 pb-3 backdrop-blur">
                <AudioPlayer meetingId={id} transcript={transcript.data} />
              </div>
              <Tabs defaultValue="minutes">
                <TabsList className="w-full">
                  <TabsTrigger value="minutes">Minutes</TabsTrigger>
                  <TabsTrigger value="transcript">Transcript</TabsTrigger>
                </TabsList>
                <TabsContent value="minutes" className="min-h-[60dvh] pb-8">
                  <ArtifactsPane summary={summary.data} names={names} />
                </TabsContent>
                <TabsContent value="transcript" className="h-[70dvh] pb-4">
                  <TranscriptPane transcript={transcript.data} />
                </TabsContent>
              </Tabs>
            </div>
            )}
          </>
        )}
      </div>
    </EvidenceProvider>
  );
}
