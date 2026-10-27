import { Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { MeetingStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLES: Record<MeetingStatus, string> = {
  uploaded: "bg-secondary text-secondary-foreground",
  transcribing: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200",
  extracting: "bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-200",
  ready: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
};

const LABELS: Record<MeetingStatus, string> = {
  uploaded: "Uploaded",
  transcribing: "Transcribing",
  extracting: "Extracting",
  ready: "Ready",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: MeetingStatus }) {
  const busy = status === "transcribing" || status === "extracting";
  return (
    <Badge className={cn("gap-1 border-transparent font-medium", STYLES[status])}>
      {busy && <Loader2 className="size-3 animate-spin" />}
      {LABELS[status]}
    </Badge>
  );
}
