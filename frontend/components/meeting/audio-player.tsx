"use client";

import { useQuery } from "@tanstack/react-query";
import { Pause, Play, RotateCcw, RotateCw, VolumeX } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { fmtTime } from "@/lib/format";
import type { Transcript } from "@/lib/types";
import { cn } from "@/lib/utils";

import { useEvidence } from "./evidence-context";

const RATES = [1, 1.25, 1.5, 2];

export function AudioPlayer({ meetingId, transcript }: { meetingId: string; transcript: Transcript }) {
  const { registerSeek, active, setCurrentTime, currentTime } = useEvidence();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const { data: audio, isError } = useQuery({
    queryKey: ["audio", meetingId],
    queryFn: () => api.getAudioUrl(meetingId),
    staleTime: 50 * 60_000,
    retry: false,
  });
  const hasAudio = Boolean(audio?.url) && !isError;
  const duration = transcript.duration_sec || transcript.utterances.at(-1)?.end || 1;

  const seek = useCallback(
    (sec: number, play?: boolean) => {
      const t = Math.max(0, Math.min(sec, duration));
      setCurrentTime(t);
      const el = audioRef.current;
      if (!el) return;
      el.currentTime = t;
      if (play) el.play().catch(() => {});
    },
    [duration, setCurrentTime],
  );

  useEffect(() => registerSeek(seek), [registerSeek, seek]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = rate;
  }, [rate]);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) el.play().catch(() => {});
    else el.pause();
  };

  const onTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - rect.left) / rect.width) * duration);
  };

  const pct = (s: number) => `${Math.min(100, Math.max(0, (s / duration) * 100))}%`;

  return (
    <div className="flex flex-col gap-4 rounded-xl border bg-card p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">Recording</h2>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </span>
      </div>

      {hasAudio && (
        <audio
          ref={audioRef}
          src={audio!.url}
          preload="metadata"
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
      )}

      {/* Timeline: speaker turns, evidence spans, playhead */}
      <div
        className="relative h-14 cursor-pointer overflow-hidden rounded-md bg-muted"
        onClick={onTimelineClick}
        role="slider"
        aria-label="Seek"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={currentTime}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") seek(currentTime + 5);
          if (e.key === "ArrowLeft") seek(currentTime - 5);
        }}
      >
        {transcript.utterances.map((u) => (
          <div
            key={u.utt_id}
            className="absolute bottom-0 h-3 bg-foreground/15"
            style={{ left: pct(u.start), width: `max(1px, ${pct(u.end - u.start)})` }}
          />
        ))}
        {active?.spans.map((s, i) => (
          <div
            key={`${s.utt_id}-${i}`}
            className="absolute inset-y-0 bg-highlight-strong/80 ring-1 ring-highlight-strong"
            style={{ left: pct(s.start_sec), width: `max(3px, ${pct(s.end_sec - s.start_sec)})` }}
            title={`#${s.utt_id} · ${fmtTime(s.start_sec)}`}
          />
        ))}
        <div className="absolute inset-y-0 w-0.5 bg-primary" style={{ left: pct(currentTime) }} />
      </div>

      <div className="flex items-center gap-2">
        <Button variant="outline" size="icon" disabled={!hasAudio} onClick={() => seek(currentTime - 10)} aria-label="Back 10 seconds">
          <RotateCcw />
        </Button>
        <Button size="icon-lg" className="size-10 rounded-full" disabled={!hasAudio} onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause /> : <Play />}
        </Button>
        <Button variant="outline" size="icon" disabled={!hasAudio} onClick={() => seek(currentTime + 10)} aria-label="Forward 10 seconds">
          <RotateCw />
        </Button>
        <div className="ml-auto flex gap-1">
          {RATES.map((r) => (
            <button
              key={r}
              onClick={() => setRate(r)}
              disabled={!hasAudio}
              className={cn(
                "rounded px-1.5 py-0.5 font-mono text-xs text-muted-foreground disabled:opacity-40",
                rate === r && "bg-secondary text-foreground",
              )}
            >
              {r}×
            </button>
          ))}
        </div>
      </div>

      {!hasAudio && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <VolumeX className="size-3.5" /> No audio for this meeting. The timeline still shows where the evidence is.
        </p>
      )}
      {active && active.spans.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Evidence at{" "}
          {active.spans
            .map((s) => fmtTime(s.start_sec))
            .filter((v, i, a) => a.indexOf(v) === i)
            .join(", ")}
        </p>
      )}
    </div>
  );
}
