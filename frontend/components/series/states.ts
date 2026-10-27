import type { CommitmentState } from "@/lib/types";

export const STATE_META: Record<CommitmentState, { label: string; cell: string; dot: string }> = {
  PROPOSED: {
    label: "Proposed",
    cell: "bg-secondary text-secondary-foreground border-dashed border-muted-foreground/40",
    dot: "bg-muted-foreground/50",
  },
  COMMITTED: {
    label: "Committed",
    cell: "bg-sky-100 text-sky-900 border-sky-200 dark:bg-sky-950 dark:text-sky-100 dark:border-sky-900",
    dot: "bg-sky-500",
  },
  FULFILLED: {
    label: "Fulfilled",
    cell: "bg-emerald-100 text-emerald-900 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-100 dark:border-emerald-900",
    dot: "bg-emerald-500",
  },
  SLIPPED: {
    label: "Slipped",
    cell: "bg-amber-100 text-amber-900 border-amber-200 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-900",
    dot: "bg-amber-500",
  },
  SILENTLY_DROPPED: {
    label: "Silently dropped",
    cell: "bg-zinc-200 text-zinc-700 border-zinc-300 line-through decoration-zinc-500/60 dark:bg-zinc-800 dark:text-zinc-300 dark:border-zinc-700",
    dot: "bg-zinc-400",
  },
  REVERSED: {
    label: "Reversed",
    cell: "bg-rose-100 text-rose-900 border-rose-200 dark:bg-rose-950 dark:text-rose-100 dark:border-rose-900",
    dot: "bg-rose-500",
  },
};

export const STATE_ORDER: CommitmentState[] = [
  "PROPOSED",
  "COMMITTED",
  "SLIPPED",
  "FULFILLED",
  "SILENTLY_DROPPED",
  "REVERSED",
];
