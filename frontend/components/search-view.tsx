"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Loader2, Search as SearchIcon, ShieldAlert, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, api } from "@/lib/api";
import { fmtDate, fmtTime, REASON_LABELS } from "@/lib/format";
import type { SearchFilters, SearchResponse, SearchSource } from "@/lib/types";
import { cn } from "@/lib/utils";

const TYPES = [
  { value: "utterances", label: "Discussion" },
  { value: "decision", label: "Decisions" },
  { value: "action_item", label: "Action items" },
] as const;

const EXAMPLES = [
  "Why did we choose Postgres?",
  "What is the fraud team's latency budget?",
  "Who owns the staging database?",
  "What happened to the Thursday demos?",
];

function SourceCard({ s, cited }: { s: SearchSource; cited: boolean }) {
  return (
    <Link
      href={`/meetings/${s.meeting_id}?utt=${s.start_utt}`}
      id={`source-${s.n}`}
      className={cn(
        "group flex flex-col gap-2 rounded-xl border bg-card p-4 transition-colors hover:border-primary/40",
        cited && "border-highlight-strong/70 bg-highlight/15",
      )}
    >
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="flex size-5 items-center justify-center rounded bg-secondary font-mono text-[11px] text-foreground">
          {s.n}
        </span>
        <span className="truncate font-medium text-foreground">{s.meeting_title}</span>
        <span>· {fmtDate(s.occurred_at)}</span>
        <ArrowUpRight className="ml-auto size-4 opacity-0 transition-opacity group-hover:opacity-100" />
      </div>
      <p className="line-clamp-4 text-sm leading-relaxed">{s.text}</p>
      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
        <Badge variant="secondary" className="text-[10px]">
          {s.artifact_type === "utterances" ? "discussion" : s.artifact_type.replace("_", " ")}
        </Badge>
        <span className="font-mono">
          #{s.start_utt}
          {s.end_utt !== s.start_utt && `–${s.end_utt}`} · {fmtTime(s.start_sec)}
        </span>
      </div>
    </Link>
  );
}

function Answer({ r }: { r: SearchResponse }) {
  const titleOf = (mid?: string | null) => r.sources.find((s) => s.meeting_id === mid)?.meeting_title ?? mid;
  return (
    <div className="rounded-xl border bg-card p-5">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium">
        {r.answerable && r.validated ? (
          <span className="inline-flex items-center gap-1 text-success">
            <ShieldCheck className="size-4" /> Answer verified against the transcripts
          </span>
        ) : r.answerable ? (
          <span className="inline-flex items-center gap-1 text-warning">
            <ShieldAlert className="size-4" /> Partly verified: {r.flags.map((f) => REASON_LABELS[f] ?? f).join("; ")}
          </span>
        ) : (
          <span className="text-muted-foreground">No supported answer in your meetings</span>
        )}
      </div>
      <p className="font-serif text-2xl leading-snug">{r.answer}</p>
      {r.evidence.length > 0 && (
        <ul className="mt-4 flex flex-col gap-1.5 border-t pt-3 text-sm">
          {r.evidence.map((e, i) => (
            <li key={i} className="flex items-baseline gap-2">
              <Link
                href={`/meetings/${e.meeting_id}?utt=${e.utt_id}`}
                className="shrink-0 rounded bg-secondary px-1.5 font-mono text-xs hover:bg-highlight"
              >
                {titleOf(e.meeting_id)} · {fmtTime(e.start_sec)}
              </Link>
              <span className="text-muted-foreground">“{e.quote}”</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SearchView() {
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const series = useQuery({ queryKey: ["series"], queryFn: api.listSeries, retry: false });
  const search = useMutation({ mutationFn: (q: string) => api.search(q, filters) });

  const run = (q: string) => {
    if (q.trim().length < 2) return;
    setQuery(q);
    search.mutate(q.trim());
  };

  const toggleType = (t: (typeof TYPES)[number]["value"]) => {
    const cur = new Set(filters.artifact_types ?? []);
    if (cur.has(t)) cur.delete(t);
    else cur.add(t);
    setFilters({ ...filters, artifact_types: cur.size ? [...cur] : null });
  };

  const cited = (s: SearchSource) =>
    !!search.data?.evidence.some((e) => e.meeting_id === s.meeting_id && e.utt_id >= s.start_utt && e.utt_id <= s.end_utt);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-12">
      <h1 className="font-serif text-4xl tracking-tight sm:text-5xl">Ask your meetings</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Hybrid keyword and semantic retrieval across every meeting. Answers cite the meeting and the moment, and
        they go through the same grounding validator as the minutes.
      </p>

      <form
        className="mt-6 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          run(query);
        }}
      >
        <div className="relative flex-1">
          <SearchIcon className="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What did we decide about…"
            className="h-11 pl-9 text-base"
            autoFocus
          />
        </div>
        <Button type="submit" className="h-11 px-5" disabled={search.isPending}>
          {search.isPending ? <Loader2 className="animate-spin" /> : "Ask"}
        </Button>
      </form>

      <div className="mt-3 flex flex-wrap items-end gap-x-4 gap-y-3 text-sm">
        <div className="flex gap-1">
          {TYPES.map((t) => (
            <button
              key={t.value}
              type="button"
              onClick={() => toggleType(t.value)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs text-muted-foreground transition-colors",
                filters.artifact_types?.includes(t.value) && "border-primary bg-accent text-foreground",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="grid gap-1">
          <Label className="text-xs text-muted-foreground">Series</Label>
          <select
            className="h-8 rounded-md border bg-background px-2 text-xs"
            value={filters.series_id ?? ""}
            onChange={(e) => setFilters({ ...filters, series_id: e.target.value || null })}
          >
            <option value="">All meetings</option>
            {series.data?.map((s) => (
              <option key={s.series_id} value={s.series_id}>
                {s.series_id}
              </option>
            ))}
          </select>
        </div>
        <div className="grid gap-1">
          <Label className="text-xs text-muted-foreground">Participant</Label>
          <Input
            className="h-8 w-32 text-xs"
            placeholder="Anyone"
            value={filters.participant ?? ""}
            onChange={(e) => setFilters({ ...filters, participant: e.target.value || null })}
          />
        </div>
        <div className="grid gap-1">
          <Label className="text-xs text-muted-foreground">From</Label>
          <Input
            type="date"
            className="h-8 text-xs"
            value={filters.date_from ?? ""}
            onChange={(e) => setFilters({ ...filters, date_from: e.target.value || null })}
          />
        </div>
        <div className="grid gap-1">
          <Label className="text-xs text-muted-foreground">To</Label>
          <Input
            type="date"
            className="h-8 text-xs"
            value={filters.date_to ?? ""}
            onChange={(e) => setFilters({ ...filters, date_to: e.target.value || null })}
          />
        </div>
      </div>

      {!search.data && !search.isPending && !search.error && (
        <div className="mt-10 flex flex-wrap gap-2">
          {EXAMPLES.map((q) => (
            <button
              key={q}
              onClick={() => run(q)}
              className="rounded-full border bg-card px-3 py-1.5 text-sm text-muted-foreground hover:border-primary/40 hover:text-foreground"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {search.error && (
        <p className="mt-8 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          {search.error instanceof ApiError && search.error.status === 501
            ? "Search isn't available on this server yet."
            : (search.error as Error).message}
        </p>
      )}

      {search.isPending && (
        <div className="mt-8 flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Retrieving and checking the evidence…
        </div>
      )}

      {search.data && (
        <div className="mt-8 flex flex-col gap-6">
          <Answer r={search.data} />
          {search.data.sources.length > 0 && (
            <div>
              <h2 className="mb-3 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                Retrieved sources
              </h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {search.data.sources.map((s) => (
                  <SourceCard key={s.n} s={s} cited={cited(s)} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
