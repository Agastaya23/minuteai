"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, RefreshCw, Undo2, UserX, X } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { fmtDate, fmtTime } from "@/lib/format";
import type { Commitment, CommitmentGraph, EvidenceSpan, Transition } from "@/lib/types";
import { cn } from "@/lib/utils";

import { STATE_META, STATE_ORDER } from "./states";

function EvidenceList({ spans, titles }: { spans: EvidenceSpan[]; titles: Map<string, string> }) {
  if (!spans.length) return <p className="text-xs text-muted-foreground">No direct quote: this state comes from silence.</p>;
  return (
    <ul className="flex flex-col gap-1.5">
      {spans.map((e, i) => (
        <li key={i} className="text-sm">
          <Link
            href={`/meetings/${e.meeting_id}?utt=${e.utt_id}`}
            className="mr-1.5 rounded bg-secondary px-1.5 py-0.5 font-mono text-[11px] hover:bg-highlight"
          >
            {titles.get(e.meeting_id ?? "") ?? e.meeting_id} · {fmtTime(e.start_sec)}
          </Link>
          <span className="text-muted-foreground">“{e.quote}”</span>
        </li>
      ))}
    </ul>
  );
}

function TransitionDetail({
  c,
  meetingId,
  titles,
  onClose,
}: {
  c: Commitment;
  meetingId: string | null;
  titles: Map<string, string>;
  onClose: () => void;
}) {
  const transitions: Transition[] = meetingId ? c.transitions.filter((t) => t.meeting_id === meetingId) : c.transitions;
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="mb-3 flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="font-serif text-xl leading-snug">{c.description}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {c.owner_name ?? "No clear owner"} · raised in {titles.get(c.origin_meeting_id)}
            {c.due_date && <> · due {fmtDate(c.due_date)}</>}
          </p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close">
          <X />
        </Button>
      </div>
      {transitions.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No change of state in {titles.get(meetingId ?? "")}. It stayed {STATE_META[c.timeline[meetingId ?? ""] ?? c.state].label.toLowerCase()}.
        </p>
      )}
      <ol className="flex flex-col gap-4">
        {transitions.map((t, i) => (
          <li key={i} className="border-l-2 pl-3" style={{ borderColor: "var(--border)" }}>
            <div className="mb-1 flex flex-wrap items-center gap-2 text-sm font-medium">
              {t.from_state && (
                <>
                  <span className={cn("rounded border px-1.5 text-xs", STATE_META[t.from_state].cell)}>
                    {STATE_META[t.from_state].label}
                  </span>
                  <ArrowRight className="size-3.5 text-muted-foreground" />
                </>
              )}
              <span className={cn("rounded border px-1.5 text-xs", STATE_META[t.to_state].cell)}>
                {STATE_META[t.to_state].label}
              </span>
              <span className="text-xs font-normal text-muted-foreground">
                in {titles.get(t.meeting_id)} · {fmtDate(t.occurred_at)}
              </span>
            </div>
            <p className="mb-2 text-sm">{t.reason}</p>
            <EvidenceList spans={t.evidence} titles={titles} />
          </li>
        ))}
      </ol>
    </div>
  );
}

function Timeline({
  graph,
  titles,
  onPick,
  picked,
}: {
  graph: CommitmentGraph;
  titles: Map<string, string>;
  onPick: (c: Commitment, meetingId: string | null) => void;
  picked: { id: string; meeting: string | null } | null;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className="border-b">
            <th className="w-[38%] px-4 py-3 text-left text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Commitment
            </th>
            {graph.meetings.map((m) => (
              <th key={m.meeting_id} className="px-2 py-3 text-left align-bottom">
                <Link href={`/meetings/${m.meeting_id}`} className="block font-medium hover:underline">
                  {m.title}
                </Link>
                <span className="text-xs font-normal text-muted-foreground">{fmtDate(m.occurred_at, { day: "numeric", month: "short" })}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {graph.commitments.map((c) => (
            <tr key={c.commitment_id} className="border-b last:border-0">
              <td className="px-4 py-2.5">
                <button className="text-left" onClick={() => onPick(c, null)}>
                  <span className="font-medium hover:underline">{c.description}</span>
                  <span className="mt-0.5 flex items-center gap-1 text-xs text-muted-foreground">
                    {c.owner_name ?? (
                      <span className="inline-flex items-center gap-1 text-warning">
                        <UserX className="size-3" /> Unassigned
                      </span>
                    )}
                    {c.due_date && <> · due {fmtDate(c.due_date, { day: "numeric", month: "short" })}</>}
                  </span>
                </button>
              </td>
              {graph.meetings.map((m) => {
                const state = c.timeline[m.meeting_id];
                const changed = c.transitions.some((t) => t.meeting_id === m.meeting_id);
                const isPicked = picked?.id === c.commitment_id && picked.meeting === m.meeting_id;
                return (
                  <td key={m.meeting_id} className="px-2 py-2.5">
                    {state ? (
                      <button
                        onClick={() => onPick(c, m.meeting_id)}
                        title={`${STATE_META[state].label} after ${titles.get(m.meeting_id)}`}
                        className={cn(
                          "flex h-8 w-full items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-shadow",
                          STATE_META[state].cell,
                          !changed && "opacity-55",
                          isPicked && "ring-2 ring-ring ring-offset-1 ring-offset-background",
                        )}
                      >
                        {changed && <span className={cn("size-1.5 rounded-full", STATE_META[state].dot)} />}
                        {STATE_META[state].label}
                      </button>
                    ) : (
                      <span className="block h-8 rounded-md border border-dashed border-border/60" />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Ledger({ graph, titles }: { graph: CommitmentGraph; titles: Map<string, string> }) {
  const people = graph.people.filter((p) => p.method !== "unresolved");
  const [personId, setPersonId] = useState<string | null>(people[0]?.person_id ?? null);
  const ledger = useQuery({
    queryKey: ["ledger", personId],
    queryFn: () => api.getLedger(personId!),
    enabled: !!personId,
  });
  const unresolvedCount = graph.people.length - people.length;
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {people.map((p) => (
          <button
            key={p.person_id}
            onClick={() => setPersonId(p.person_id)}
            className={cn(
              "rounded-full border px-3 py-1 text-sm",
              personId === p.person_id ? "border-primary bg-primary text-primary-foreground" : "hover:bg-muted",
            )}
            title={`Identified by ${p.method.replace("_", " ")} · ${Math.round(p.confidence * 100)}%`}
          >
            {p.display_name}
          </button>
        ))}
        {unresolvedCount > 0 && (
          <span className="text-xs text-muted-foreground">
            + {unresolvedCount} unidentified speaker{unresolvedCount > 1 ? "s" : ""}
          </span>
        )}
      </div>
      {ledger.data && (
        <>
          <div className="mb-3 flex flex-wrap gap-3 text-xs">
            {STATE_ORDER.filter((s) => ledger.data.counts[s]).map((s) => (
              <span key={s} className="inline-flex items-center gap-1.5">
                <span className={cn("size-2 rounded-full", STATE_META[s].dot)} />
                {ledger.data.counts[s]} {STATE_META[s].label.toLowerCase()}
              </span>
            ))}
          </div>
          <ul className="flex flex-col divide-y">
            {ledger.data.entries.map(({ commitment: c }) => (
              <li key={c.commitment_id} className="flex items-center gap-3 py-2 text-sm">
                <span className={cn("rounded border px-1.5 py-0.5 text-[11px] font-medium", STATE_META[c.state].cell)}>
                  {STATE_META[c.state].label}
                </span>
                <span className="flex-1">{c.description}</span>
                <span className="text-xs text-muted-foreground">{titles.get(c.origin_meeting_id)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export function SeriesView({ id }: { id: string }) {
  const qc = useQueryClient();
  const graph = useQuery({ queryKey: ["graph", id], queryFn: () => api.getCommitments(id) });
  const rebuild = useMutation({
    mutationFn: () => api.rebuildSeries(id),
    onSuccess: (g) => {
      qc.setQueryData(["graph", id], g);
      qc.invalidateQueries({ queryKey: ["ledger"] });
      toast.success("Accountability graph rebuilt");
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const [picked, setPicked] = useState<{ id: string; meeting: string | null } | null>(null);
  const titles = useMemo(() => new Map(graph.data?.meetings.map((m) => [m.meeting_id, m.title]) ?? []), [graph.data]);
  const pickedCommitment = graph.data?.commitments.find((c) => c.commitment_id === picked?.id);

  const counts = useMemo(() => {
    const m = new Map<string, number>();
    graph.data?.commitments.forEach((c) => m.set(c.state, (m.get(c.state) ?? 0) + 1));
    return m;
  }, [graph.data]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <Link href="/series" className={buttonVariants({ variant: "ghost", size: "sm" })}>
        <ArrowLeft /> All series
      </Link>
      <div className="mt-3 mb-6 flex flex-wrap items-end gap-4">
        <div>
          <h1 className="font-serif text-4xl tracking-tight sm:text-5xl">{id}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            What was promised, and what became of it. Click any cell to see the moment that changed it.
          </p>
        </div>
        <Button variant="outline" size="sm" className="ml-auto" onClick={() => rebuild.mutate()} disabled={rebuild.isPending}>
          <RefreshCw className={cn(rebuild.isPending && "animate-spin")} /> Rebuild
        </Button>
      </div>

      {graph.isLoading && <Skeleton className="h-72 rounded-xl" />}
      {graph.error && <p className="text-sm text-destructive">{(graph.error as Error).message}</p>}

      {graph.data && (
        <div className="flex flex-col gap-8">
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            {STATE_ORDER.map((s) => (
              <span key={s} className="inline-flex items-center gap-1.5">
                <span className={cn("size-2 rounded-full", STATE_META[s].dot)} /> {STATE_META[s].label}
                <span className="font-mono text-foreground">{counts.get(s) ?? 0}</span>
              </span>
            ))}
          </div>

          {graph.data.commitments.length === 0 ? (
            <p className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
              No commitments yet. They appear once meetings in this series finish processing.
            </p>
          ) : (
            <Timeline
              graph={graph.data}
              titles={titles}
              picked={picked}
              onPick={(c, meeting) => setPicked({ id: c.commitment_id, meeting })}
            />
          )}

          {pickedCommitment && (
            <TransitionDetail c={pickedCommitment} meetingId={picked!.meeting} titles={titles} onClose={() => setPicked(null)} />
          )}

          <div className="grid gap-8 lg:grid-cols-2">
            <section>
              <h2 className="mb-3 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Per-person ledger</h2>
              <Ledger graph={graph.data} titles={titles} />
            </section>
            <section>
              <h2 className="mb-3 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Decision reversals</h2>
              {graph.data.reversals.length === 0 ? (
                <p className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">No decision has been overturned in this series.</p>
              ) : (
                <ul className="flex flex-col gap-3">
                  {graph.data.reversals.map((r) => (
                    <li key={r.reversal_id} className="rounded-xl border bg-card p-4">
                      <div className="grid gap-3 sm:grid-cols-[1fr_auto_1fr] sm:items-start">
                        <div>
                          <p className="text-xs text-muted-foreground">{titles.get(r.original.meeting_id)}</p>
                          <p className="font-medium">{r.original.statement}</p>
                          <div className="mt-2">
                            <EvidenceList spans={r.original.evidence} titles={titles} />
                          </div>
                        </div>
                        <Undo2 className="mx-auto size-4 text-rose-500 sm:mt-6" />
                        <div>
                          <p className="text-xs text-muted-foreground">{titles.get(r.reversing.meeting_id)}</p>
                          <p className="font-medium">{r.reversing.statement}</p>
                          <div className="mt-2">
                            <EvidenceList spans={r.reversing.evidence} titles={titles} />
                          </div>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
