import { CalendarDays, Clock, Repeat, Users } from "lucide-react";
import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";
import { fmtDate, fmtDuration } from "@/lib/format";
import type { Meeting } from "@/lib/types";

export function MeetingCard({ meeting }: { meeting: Meeting }) {
  return (
    <Link
      href={`/meetings/${meeting.meeting_id}`}
      className="group flex flex-col gap-3 rounded-xl border bg-card p-4 transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-[0_6px_24px_-12px_oklch(0.3_0.08_268/0.35)]"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-serif text-xl leading-tight group-hover:text-primary">{meeting.title}</h3>
        <StatusBadge status={meeting.status} />
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <CalendarDays className="size-3.5" />
          <dd>{fmtDate(meeting.occurred_at)}</dd>
        </div>
        <div className="flex items-center gap-1.5">
          <Clock className="size-3.5" />
          <dd>{fmtDuration(meeting.duration_sec)}</dd>
        </div>
        <div className="col-span-2 flex items-center gap-1.5">
          <Users className="size-3.5 shrink-0" />
          <dd className="truncate">{meeting.participants.length ? meeting.participants.join(", ") : "Participants pending"}</dd>
        </div>
        {meeting.series_id && (
          <div className="col-span-2 flex items-center gap-1.5">
            <Repeat className="size-3.5" />
            <dd className="truncate">{meeting.series_id}</dd>
          </div>
        )}
      </dl>
      {meeting.status === "failed" && meeting.error && (
        <p className="line-clamp-2 rounded-md bg-destructive/10 px-2 py-1 text-xs text-destructive">{meeting.error}</p>
      )}
    </Link>
  );
}
