export function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function fmtDuration(sec: number): string {
  if (!sec) return "—";
  const m = Math.round(sec / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function fmtDate(iso: string | null | undefined, opts?: Intl.DateTimeFormatOptions): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, opts ?? { day: "numeric", month: "short", year: "numeric" });
}

export function fmtPct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export const IN_FLIGHT = new Set(["uploaded", "transcribing", "extracting"]);

export const REASON_LABELS: Record<string, string> = {
  UTT_NOT_FOUND: "Cited utterance does not exist",
  QUOTE_NOT_FOUND: "Quote not found in cited utterance",
  TIMESTAMP_MISMATCH: "Timestamp outside the utterance",
  SPEAKER_MISMATCH: "Attributed speaker did not say it",
  ENTITY_NOT_SUPPORTED: "Name, number or date not in the evidence",
  NO_EVIDENCE: "No evidence cited",
};

export const SPEAKER_COLORS = [
  "text-sky-700 dark:text-sky-300",
  "text-violet-700 dark:text-violet-300",
  "text-emerald-700 dark:text-emerald-300",
  "text-rose-700 dark:text-rose-300",
  "text-amber-700 dark:text-amber-300",
  "text-teal-700 dark:text-teal-300",
];

export function speakerColor(label: string): string {
  const n = parseInt(label.replace(/\D/g, ""), 10);
  return SPEAKER_COLORS[(Number.isNaN(n) ? 0 : n) % SPEAKER_COLORS.length];
}
