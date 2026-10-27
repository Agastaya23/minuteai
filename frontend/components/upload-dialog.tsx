"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FileAudio, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { api, uploadToS3 } from "@/lib/api";
import { cn } from "@/lib/utils";

const ACCEPT = "audio/*,video/mp4,video/webm,.m4a,.mp3,.wav,.ogg,.flac,.webm";

function guessType(file: File): string {
  if (file.type) return file.type;
  const ext = file.name.split(".").pop()?.toLowerCase();
  return (
    { m4a: "audio/mp4", mp3: "audio/mpeg", wav: "audio/wav", ogg: "audio/ogg", flac: "audio/flac", webm: "audio/webm" }[
      ext ?? ""
    ] ?? "audio/mpeg"
  );
}

export function UploadDialog() {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [series, setSeries] = useState("");
  const [participants, setParticipants] = useState("");
  const [occurredAt, setOccurredAt] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const router = useRouter();

  const reset = () => {
    setFile(null);
    setTitle("");
    setSeries("");
    setParticipants("");
    setOccurredAt("");
    setProgress(null);
  };

  const pick = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      setFile(f);
      if (!title) setTitle(f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "));
    },
    [title],
  );

  const upload = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Choose an audio file");
      const contentType = guessType(file);
      const { meeting, upload_url } = await api.createMeeting({
        title: title.trim() || file.name,
        filename: file.name,
        content_type: contentType,
        series_name: series.trim() || null,
        occurred_at: occurredAt ? new Date(occurredAt).toISOString() : null,
        participants: participants
          .split(",")
          .map((p) => p.trim())
          .filter(Boolean),
      });
      setProgress(0);
      await uploadToS3(upload_url, file, contentType, setProgress);
      await api.processMeeting(meeting.meeting_id);
      return meeting;
    },
    onSuccess: (meeting) => {
      toast.success("Uploaded — processing has started");
      qc.invalidateQueries({ queryKey: ["meetings"] });
      setOpen(false);
      reset();
      router.push(`/meetings/${meeting.meeting_id}`);
    },
    onError: (e: Error) => {
      setProgress(null);
      toast.error(e.message);
    },
  });

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!upload.isPending) setOpen(o);
      }}
    >
      <DialogTrigger render={<Button />}>
        <Upload /> Upload meeting
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="font-serif text-2xl font-normal">New meeting</DialogTitle>
          <DialogDescription>
            Audio goes straight to S3, then MinuteAI transcribes, extracts and verifies the minutes.
          </DialogDescription>
        </DialogHeader>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pick(e.dataTransfer.files?.[0]);
          }}
          className={cn(
            "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors",
            dragging ? "border-primary bg-accent" : "border-border hover:bg-muted/60",
          )}
        >
          <FileAudio className="size-7 text-muted-foreground" />
          {file ? (
            <span className="text-sm font-medium">
              {file.name} <span className="text-muted-foreground">· {(file.size / 1e6).toFixed(1)} MB</span>
            </span>
          ) : (
            <span className="text-sm text-muted-foreground">Drop an audio file here, or click to choose</span>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </button>

        <div className="grid gap-3">
          <div className="grid gap-1.5">
            <Label htmlFor="title">Title</Label>
            <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Sprint planning" />
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <Label htmlFor="series">Series (optional)</Label>
              <Input
                id="series"
                value={series}
                onChange={(e) => setSeries(e.target.value)}
                placeholder="Payments weekly"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="when">Held on (optional)</Label>
              <Input id="when" type="datetime-local" value={occurredAt} onChange={(e) => setOccurredAt(e.target.value)} />
            </div>
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="participants">Participants (optional, comma-separated)</Label>
            <Input
              id="participants"
              value={participants}
              onChange={(e) => setParticipants(e.target.value)}
              placeholder="Priya, Rahul, Ananya"
            />
          </div>
        </div>

        {progress !== null && (
          <div className="grid gap-1">
            <Progress value={Math.round(progress * 100)} />
            <span className="text-xs text-muted-foreground">
              {progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : "Starting processing…"}
            </span>
          </div>
        )}

        <DialogFooter>
          <Button disabled={!file || upload.isPending} onClick={() => upload.mutate()}>
            {upload.isPending ? "Uploading…" : "Upload and process"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
