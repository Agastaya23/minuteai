"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useMemo, useRef } from "react";

import { fmtTime, speakerColor } from "@/lib/format";
import type { EvidenceSpan, Transcript, Utterance } from "@/lib/types";
import { cn } from "@/lib/utils";

import { useEvidence } from "./evidence-context";

function markQuote(text: string, quotes: string[]): React.ReactNode {
  const lower = text.toLowerCase();
  const ranges: [number, number][] = [];
  for (const q of quotes) {
    const needle = q.toLowerCase().replace(/[“”"]/g, "").trim();
    if (!needle) continue;
    const i = lower.indexOf(needle);
    if (i >= 0) ranges.push([i, i + needle.length]);
  }
  if (!ranges.length) return text;
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([...r]);
  }
  const out: React.ReactNode[] = [];
  let pos = 0;
  merged.forEach(([a, b], k) => {
    if (a > pos) out.push(text.slice(pos, a));
    out.push(
      <mark key={k} className="evidence-mark text-inherit">
        {text.slice(a, b)}
      </mark>,
    );
    pos = b;
  });
  if (pos < text.length) out.push(text.slice(pos));
  return out;
}

function UtteranceRow({
  u,
  active,
  quotes,
  playing,
  onClick,
}: {
  u: Utterance;
  active: boolean;
  quotes: string[];
  playing: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "group relative cursor-pointer rounded-lg border-l-2 border-transparent px-3 py-2 transition-colors hover:bg-muted/60",
        active && "border-l-highlight-strong bg-highlight/40 hover:bg-highlight/50",
        playing && !active && "border-l-primary/60 bg-accent/50",
      )}
    >
      <div className="mb-0.5 flex items-baseline gap-2 text-xs">
        <span className={cn("font-semibold", speakerColor(u.speaker_label))}>
          {u.speaker_name ?? u.speaker_label}
        </span>
        {u.speaker_name && <span className="text-muted-foreground/70">{u.speaker_label}</span>}
        <span className="ml-auto font-mono text-muted-foreground tabular-nums">
          #{u.utt_id} · {fmtTime(u.start)}
        </span>
      </div>
      <p className="text-[15px] leading-relaxed">{active ? markQuote(u.text, quotes) : u.text}</p>
    </div>
  );
}

export function TranscriptPane({ transcript }: { transcript: Transcript }) {
  const { active, activeUtts, registerScroll, seek, currentTime } = useEvidence();
  const parentRef = useRef<HTMLDivElement>(null);
  const utts = transcript.utterances;
  const indexById = useMemo(() => new Map(utts.map((u, i) => [u.utt_id, i])), [utts]);

  const virtualizer = useVirtualizer({
    count: utts.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 76,
    overscan: 8,
  });

  useEffect(() => {
    registerScroll((uttId: number) => {
      const idx = indexById.get(uttId);
      if (idx !== undefined) virtualizer.scrollToIndex(idx, { align: "center", behavior: "smooth" });
    });
  }, [registerScroll, indexById, virtualizer]);

  const quotesByUtt = useMemo(() => {
    const m = new Map<number, string[]>();
    (active?.spans ?? []).forEach((s: EvidenceSpan) => m.set(s.utt_id, [...(m.get(s.utt_id) ?? []), s.quote]));
    return m;
  }, [active]);

  const playingIdx = useMemo(() => {
    let lo = 0;
    let hi = utts.length - 1;
    let ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (utts[mid].start <= currentTime) {
        ans = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return ans >= 0 && currentTime <= utts[ans].end + 0.5 ? ans : -1;
  }, [utts, currentTime]);

  return (
    <div ref={parentRef} className="h-full overflow-y-auto overscroll-contain pr-1" aria-label="Transcript">
      <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
        {virtualizer.getVirtualItems().map((row) => {
          const u = utts[row.index];
          return (
            <div
              key={row.key}
              data-index={row.index}
              ref={virtualizer.measureElement}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", transform: `translateY(${row.start}px)` }}
              className="pb-1"
            >
              <UtteranceRow
                u={u}
                active={activeUtts.has(u.utt_id)}
                quotes={quotesByUtt.get(u.utt_id) ?? []}
                playing={row.index === playingIdx}
                onClick={() => seek(u.start, true)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
