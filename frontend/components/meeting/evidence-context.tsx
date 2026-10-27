"use client";

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

import type { EvidenceSpan } from "@/lib/types";

interface Focus {
  key: string;
  spans: EvidenceSpan[];
}

interface EvidenceState {
  hovered: Focus | null;
  selected: Focus | null;
  /** utt_ids currently highlighted (hovered takes precedence over selected) */
  active: Focus | null;
  activeUtts: Set<number>;
  hover: (key: string, spans: EvidenceSpan[]) => void;
  unhover: (key: string) => void;
  select: (key: string, spans: EvidenceSpan[]) => void;
  clear: () => void;
  currentTime: number;
  setCurrentTime: (t: number) => void;
  seek: (sec: number, play?: boolean) => void;
  scrollToUtt: (uttId: number) => void;
  registerSeek: (fn: (sec: number, play?: boolean) => void) => void;
  registerScroll: (fn: (uttId: number) => void) => void;
}

const Ctx = createContext<EvidenceState | null>(null);

export function EvidenceProvider({ children }: { children: React.ReactNode }) {
  const [hovered, setHovered] = useState<Focus | null>(null);
  const [selected, setSelected] = useState<Focus | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const seekRef = useRef<(sec: number, play?: boolean) => void>(() => {});
  const scrollRef = useRef<(uttId: number) => void>(() => {});

  const hover = useCallback((key: string, spans: EvidenceSpan[]) => setHovered({ key, spans }), []);
  const unhover = useCallback((key: string) => setHovered((h) => (h?.key === key ? null : h)), []);
  const seek = useCallback((sec: number, play?: boolean) => seekRef.current(sec, play), []);
  const scrollToUtt = useCallback((id: number) => scrollRef.current(id), []);

  const select = useCallback(
    (key: string, spans: EvidenceSpan[]) => {
      setSelected({ key, spans });
      if (spans.length) {
        const first = [...spans].sort((a, b) => a.start_sec - b.start_sec)[0];
        scrollToUtt(first.utt_id);
        seek(first.start_sec, true);
      }
    },
    [scrollToUtt, seek],
  );

  const value = useMemo<EvidenceState>(() => {
    const active = hovered ?? selected;
    return {
      hovered,
      selected,
      active,
      activeUtts: new Set(active?.spans.map((s) => s.utt_id) ?? []),
      hover,
      unhover,
      select,
      clear: () => setSelected(null),
      currentTime,
      setCurrentTime,
      seek,
      scrollToUtt,
      registerSeek: (fn) => {
        seekRef.current = fn;
      },
      registerScroll: (fn) => {
        scrollRef.current = fn;
      },
    };
  }, [hovered, selected, hover, unhover, select, currentTime, seek, scrollToUtt]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useEvidence(): EvidenceState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useEvidence must be used inside EvidenceProvider");
  return v;
}

/** Props that make any element an evidence-linked artifact. */
export function useEvidenceTarget(key: string, spans: EvidenceSpan[]) {
  const { hover, unhover, select, selected, hovered } = useEvidence();
  return {
    isActive: hovered?.key === key || (!hovered && selected?.key === key),
    isSelected: selected?.key === key,
    props: {
      onMouseEnter: () => hover(key, spans),
      onMouseLeave: () => unhover(key),
      onFocus: () => hover(key, spans),
      onBlur: () => unhover(key),
      onClick: () => select(key, spans),
      onKeyDown: (e: React.KeyboardEvent) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select(key, spans);
        }
      },
      role: "button" as const,
      tabIndex: 0,
    },
  };
}
