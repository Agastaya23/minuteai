"use client";

import { AlertTriangle } from "lucide-react";

import { MeetingCard } from "@/components/meeting-card";
import { Skeleton } from "@/components/ui/skeleton";
import { UploadDialog } from "@/components/upload-dialog";
import { useMeetings } from "@/hooks/use-meetings";

export default function MeetingsPage() {
  const { data: meetings, isLoading, error } = useMeetings();

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-12">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-serif text-4xl tracking-tight sm:text-5xl">Meetings</h1>
          <p className="mt-2 max-w-xl text-sm text-muted-foreground">
            Minutes extracted from your recordings. Every sentence, decision and action item links back to
            the moment it was said, and anything that can&apos;t be verified gets filtered out.
          </p>
        </div>
        <UploadDialog />
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 size-4 text-destructive" />
          <div>
            <p className="font-medium">Can&apos;t reach the API.</p>
            <p className="text-muted-foreground">
              Start it with <code className="font-mono">make dev-backend</code> ({(error as Error).message}).
            </p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-36 rounded-xl" />
          ))}
        </div>
      )}

      {meetings && meetings.length === 0 && (
        <div className="rounded-xl border border-dashed p-12 text-center">
          <p className="font-serif text-2xl">No meetings yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload a recording, or run <code className="font-mono">make seed</code> to load three sample meetings.
          </p>
        </div>
      )}

      {meetings && meetings.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {meetings.map((m) => (
            <MeetingCard key={m.meeting_id} meeting={m} />
          ))}
        </div>
      )}
    </div>
  );
}
