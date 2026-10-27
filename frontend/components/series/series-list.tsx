"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Repeat } from "lucide-react";
import Link from "next/link";

import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { fmtDate } from "@/lib/format";

export function SeriesList() {
  const series = useQuery({ queryKey: ["series"], queryFn: api.listSeries });
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-12">
      <h1 className="font-serif text-4xl tracking-tight sm:text-5xl">Recurring meetings</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Across a series, MinuteAI links commitments from meeting to meeting and tracks whether each one was fulfilled,
        slipped, quietly dropped or reversed. Every change of state cites the moment that caused it.
      </p>
      <div className="mt-8 flex flex-col gap-3">
        {series.isLoading && <Skeleton className="h-20" />}
        {series.data?.length === 0 && (
          <p className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
            No series yet. Give meetings a series name when you upload them.
          </p>
        )}
        {series.data?.map((s) => (
          <Link
            key={s.series_id}
            href={`/series/${s.series_id}`}
            className="group flex items-center gap-4 rounded-xl border bg-card p-4 hover:border-primary/40"
          >
            <Repeat className="size-5 text-muted-foreground" />
            <div>
              <p className="font-serif text-xl">{s.series_id}</p>
              <p className="text-xs text-muted-foreground">
                {s.meetings} meeting{s.meetings === 1 ? "" : "s"} · last {fmtDate(s.last_occurred_at)}
              </p>
            </div>
            <ArrowRight className="ml-auto size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
          </Link>
        ))}
      </div>
    </div>
  );
}
