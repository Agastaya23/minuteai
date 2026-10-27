"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ShieldX } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { fmtPct, fmtTime, REASON_LABELS } from "@/lib/format";

const CHECKS = ["existence", "quote_fidelity", "timestamp", "speaker", "entity", "non_empty"] as const;
const CHECK_LABEL: Record<string, string> = {
  existence: "Exists",
  quote_fidelity: "Quote",
  timestamp: "Time",
  speaker: "Speaker",
  entity: "Entities",
  non_empty: "Evidence",
};
const TYPE_LABEL: Record<string, string> = {
  abstract: "Summary",
  topic_summary: "Topic summary",
  decision: "Decision",
  action_item: "Action item",
  search_answer: "Search answer",
};

export function DroppedView({ id }: { id: string }) {
  const meeting = useQuery({ queryKey: ["meeting", id], queryFn: () => api.getMeeting(id) });
  const dropped = useQuery({ queryKey: ["dropped", id], queryFn: () => api.getDropped(id) });
  const transcript = useQuery({ queryKey: ["transcript", id], queryFn: () => api.getTranscript(id) });
  const utt = (n: number) => transcript.data?.utterances.find((u) => u.utt_id === n);

  const reasonCounts = new Map<string, number>();
  dropped.data?.forEach((d) => d.failures.forEach((f) => reasonCounts.set(f.reason, (reasonCounts.get(f.reason) ?? 0) + 1)));

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <Link href={`/meetings/${id}`} className={buttonVariants({ variant: "ghost", size: "sm" })}>
        <ArrowLeft /> {meeting.data?.title ?? "Back to meeting"}
      </Link>
      <div className="mt-4 mb-6">
        <h1 className="font-serif text-4xl tracking-tight">Filtered for insufficient evidence</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          The language model proposed these, but the grounding validator couldn&apos;t verify them against the
          transcript, so they were left out of the minutes. Each row shows which of the six checks failed.
        </p>
      </div>

      {reasonCounts.size > 0 && (
        <div className="mb-6 flex flex-wrap gap-2">
          {[...reasonCounts.entries()].map(([r, n]) => (
            <Badge key={r} variant="outline" className="gap-1.5">
              <span className="font-mono">{n}×</span> {REASON_LABELS[r] ?? r}
            </Badge>
          ))}
        </div>
      )}

      {dropped.isLoading && <Skeleton className="h-64" />}
      {dropped.data?.length === 0 && (
        <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
          Nothing was filtered out. Every extracted item passed the grounding checks for this meeting.
        </div>
      )}

      {!!dropped.data?.length && (
        <div className="overflow-x-auto rounded-xl border bg-card">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[40%]">Proposed artifact</TableHead>
                <TableHead>Why it was filtered</TableHead>
                {CHECKS.map((c) => (
                  <TableHead key={c} className="text-center text-xs">
                    {CHECK_LABEL[c]}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {dropped.data.map((d) => {
                const failed = new Set(d.failures.map((f) => f.check));
                return (
                  <TableRow key={`${d.artifact_type}-${d.artifact_id}`} className="align-top">
                    <TableCell className="whitespace-normal">
                      <Badge variant="secondary" className="mb-1">
                        {TYPE_LABEL[d.artifact_type] ?? d.artifact_type}
                      </Badge>
                      <p className="font-medium">
                        <s className="decoration-destructive/50">{d.text}</s>
                      </p>
                      <ul className="mt-2 flex flex-col gap-1 text-xs text-muted-foreground">
                        {d.evidence.map((e, i) => {
                          const u = utt(e.utt_id);
                          return (
                            <li key={i}>
                              cites <span className="font-mono">#{e.utt_id}</span>
                              {u ? ` (${fmtTime(u.start)})` : " (does not exist)"}: “{e.quote}”
                              {u && <span className="block pl-3 text-muted-foreground/80">actually said: “{u.text}”</span>}
                            </li>
                          );
                        })}
                      </ul>
                    </TableCell>
                    <TableCell className="whitespace-normal text-sm">
                      <ul className="flex flex-col gap-1.5">
                        {d.failures.map((f, i) => (
                          <li key={i}>
                            <span className="font-medium">{REASON_LABELS[f.reason] ?? f.reason}</span>
                            <span className="block text-xs text-muted-foreground">{f.detail}</span>
                          </li>
                        ))}
                      </ul>
                      <span className="mt-2 block text-xs text-muted-foreground">
                        grounding confidence {fmtPct(d.confidence)} · {d.mode}
                      </span>
                    </TableCell>
                    {CHECKS.map((c) => (
                      <TableCell key={c} className="text-center">
                        {failed.has(c) ? (
                          <ShieldX className="mx-auto size-4 text-destructive" aria-label="failed" />
                        ) : (
                          <span className="text-success" aria-label="passed">
                            ✓
                          </span>
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
