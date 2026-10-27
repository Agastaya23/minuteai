"use client";

import { AlertTriangle, CalendarClock, CheckCircle2, CircleDashed, Quote, ShieldCheck, UserCheck, UserX } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtDate, fmtPct, fmtTime, REASON_LABELS } from "@/lib/format";
import type { ActionItem, Decision, EvidenceSpan, GroundedSentence, MeetingSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

import { useEvidence, useEvidenceTarget } from "./evidence-context";

function Flags({ flags, validated }: { flags: string[]; validated: boolean }) {
  if (validated) {
    return (
      <Tooltip>
        <TooltipTrigger render={<span className="inline-flex text-success" />}>
          <ShieldCheck className="size-3.5" />
        </TooltipTrigger>
        <TooltipContent>Passed all six grounding checks</TooltipContent>
      </Tooltip>
    );
  }
  if (!flags.length) return null;
  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex items-center gap-1 text-[11px] font-medium text-warning" />}>
        <AlertTriangle className="size-3.5" /> partly verified
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        Kept, but flagged: {flags.map((f) => REASON_LABELS[f] ?? f).join("; ")}
      </TooltipContent>
    </Tooltip>
  );
}

function Cites({ spans }: { spans: EvidenceSpan[] }) {
  return (
    <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
      {spans.map((s, i) => (
        <span
          key={`${s.utt_id}-${i}`}
          className="rounded bg-secondary px-1 font-mono text-[10px] text-muted-foreground tabular-nums"
          title={`“${s.quote}”`}
        >
          #{s.utt_id} {fmtTime(s.start_sec)}
        </span>
      ))}
    </span>
  );
}

function ArtifactShell({
  id,
  spans,
  children,
  className,
}: {
  id: string;
  spans: EvidenceSpan[];
  children: React.ReactNode;
  className?: string;
}) {
  const { isActive, isSelected, props } = useEvidenceTarget(id, spans);
  return (
    <div
      {...props}
      className={cn(
        "cursor-pointer rounded-lg border border-transparent px-3 py-2.5 transition-colors outline-none hover:bg-highlight/30 focus-visible:ring-2 focus-visible:ring-ring",
        isActive && "border-highlight-strong/60 bg-highlight/35",
        isSelected && "border-highlight-strong",
        className,
      )}
    >
      {children}
    </div>
  );
}

function SentenceItem({ s, id }: { s: GroundedSentence; id: string }) {
  return (
    <ArtifactShell id={id} spans={s.evidence}>
      <p className="text-[15px] leading-relaxed">
        {s.text} <Flags flags={s.flags} validated={s.validated} />
      </p>
      <Cites spans={s.evidence} />
    </ArtifactShell>
  );
}

const DECISION_STYLE: Record<Decision["status"], string> = {
  agreed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  proposed: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200",
  deferred: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
  reversed: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200",
};

function DecisionItem({ d, names }: { d: Decision; names: Map<string, string> }) {
  return (
    <ArtifactShell id={`d:${d.decision_id}`} spans={d.evidence}>
      <div className="flex items-start gap-2">
        <Badge className={cn("mt-0.5 border-transparent capitalize", DECISION_STYLE[d.status])}>{d.status}</Badge>
        <div className="min-w-0 flex-1">
          <p className="font-medium leading-snug">
            {d.statement} <Flags flags={d.flags} validated={d.validated} />
          </p>
          {d.rationale && <p className="mt-0.5 text-sm text-muted-foreground">Because {d.rationale.replace(/^./, (c) => c.toLowerCase())}</p>}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {d.participants.length > 0 && <span>{d.participants.map((p) => names.get(p) ?? p).join(", ")}</span>}
            <span>confidence {fmtPct(d.confidence)}</span>
            <Cites spans={d.evidence} />
          </div>
        </div>
      </div>
    </ArtifactShell>
  );
}

const METHOD_LABEL: Record<ActionItem["owner_resolution_method"], string> = {
  self_commitment: "committed themselves",
  assigned_by_other: "assigned by someone else",
  explicit_name: "named explicitly",
  unresolved: "no clear owner",
};

function Owner({ a }: { a: ActionItem }) {
  if (!a.owner) {
    return (
      <Tooltip>
        <TooltipTrigger
          render={
            <span className="inline-flex items-center gap-1 rounded-full border border-dashed border-warning/70 bg-warning/10 px-2 py-0.5 text-xs font-medium text-warning" />
          }
        >
          <UserX className="size-3.5" /> Unassigned
        </TooltipTrigger>
        <TooltipContent>Nobody was clearly assigned in the meeting, so MinuteAI didn&apos;t guess.</TooltipContent>
      </Tooltip>
    );
  }
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span className="inline-flex items-center gap-1 rounded-full bg-primary px-2 py-0.5 text-xs font-medium text-primary-foreground" />
        }
      >
        <UserCheck className="size-3.5" /> {a.owner}
        <span className="opacity-70">{fmtPct(a.owner_confidence)}</span>
      </TooltipTrigger>
      <TooltipContent>
        Owner {METHOD_LABEL[a.owner_resolution_method]} · confidence {fmtPct(a.owner_confidence)}
      </TooltipContent>
    </Tooltip>
  );
}

function ActionItemRow({ a }: { a: ActionItem }) {
  return (
    <ArtifactShell id={`a:${a.item_id}`} spans={a.evidence}>
      <div className="flex items-start gap-2.5">
        <CircleDashed className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="font-medium leading-snug">
            {a.description} <Flags flags={a.flags} validated={a.validated} />
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Owner a={a} />
            {a.due_raw && (
              <span className="inline-flex items-center gap-1">
                <CalendarClock className="size-3.5" />
                {a.due_date ? fmtDate(a.due_date, { day: "numeric", month: "short" }) : ""}
                <span className="italic">“{a.due_raw}”</span>
              </span>
            )}
            <Cites spans={a.evidence} />
          </div>
        </div>
      </div>
    </ArtifactShell>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="px-3 py-6 text-sm text-muted-foreground">{children}</p>;
}

export function ArtifactsPane({ summary, names }: { summary: MeetingSummary; names: Map<string, string> }) {
  const { scrollToUtt } = useEvidence();
  const unassigned = summary.action_items.filter((a) => !a.owner).length;

  return (
    <Tabs defaultValue="summary" className="flex h-full flex-col">
      <TabsList variant="line" className="w-full justify-start border-b">
        <TabsTrigger value="summary" className="flex-none">Summary</TabsTrigger>
        <TabsTrigger value="decisions" className="flex-none">
          Decisions <span className="text-muted-foreground">{summary.decisions.length}</span>
        </TabsTrigger>
        <TabsTrigger value="actions" className="flex-none">
          Action items <span className="text-muted-foreground">{summary.action_items.length}</span>
        </TabsTrigger>
      </TabsList>

      <div className="min-h-0 flex-1 overflow-y-auto pt-2">
        <TabsContent value="summary" className="flex flex-col gap-1">
          {summary.abstract.length ? (
            summary.abstract.map((s, i) => <SentenceItem key={i} s={s} id={`s:${i}`} />)
          ) : (
            <Empty>No summary sentences survived grounding.</Empty>
          )}
          {summary.topics.length > 0 && (
            <div className="mt-6">
              <h3 className="mb-2 px-3 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Topics</h3>
              <ol className="flex flex-col gap-1">
                {summary.topics.map((t, i) => (
                  <li key={t.topic_id}>
                    <button
                      className="flex w-full items-baseline gap-3 rounded-lg px-3 py-2 text-left hover:bg-muted/60"
                      onClick={() => {
                        scrollToUtt(t.start_utt);
                      }}
                    >
                      <span className="font-mono text-xs text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
                      <span className="flex-1 font-medium">{t.title}</span>
                      <span className="font-mono text-xs text-muted-foreground">
                        #{t.start_utt}–{t.end_utt}
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </TabsContent>

        <TabsContent value="decisions" className="flex flex-col gap-1">
          {summary.decisions.length ? (
            summary.decisions.map((d) => <DecisionItem key={d.decision_id} d={d} names={names} />)
          ) : (
            <Empty>No decisions were recorded.</Empty>
          )}
        </TabsContent>

        <TabsContent value="actions" className="flex flex-col gap-1">
          {unassigned > 0 && (
            <p className="mx-3 mb-1 flex items-center gap-1.5 text-xs text-warning">
              <UserX className="size-3.5" /> {unassigned} item{unassigned > 1 ? "s have" : " has"} no clear owner
            </p>
          )}
          {summary.action_items.length ? (
            summary.action_items.map((a) => <ActionItemRow key={a.item_id} a={a} />)
          ) : (
            <Empty>No action items were recorded.</Empty>
          )}
        </TabsContent>
      </div>
      <p className="mt-2 flex items-center gap-1.5 border-t px-3 pt-2 text-[11px] text-muted-foreground">
        <Quote className="size-3" /> Hover any line to see its evidence; click to jump there.
        <span className="ml-auto inline-flex items-center gap-1">
          <CheckCircle2 className="size-3" />
          {summary.grounding_mode ?? "BALANCED"} grounding
        </span>
      </p>
    </Tabs>
  );
}
