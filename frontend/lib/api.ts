import type {
  CommitmentGraph,
  DecisionReversal,
  DroppedArtifact,
  Meeting,
  MeetingStatus,
  MeetingSummary,
  PersonLedger,
  SearchFilters,
  SearchResponse,
  SeriesSummary,
  Transcript,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export interface CreateMeetingInput {
  title: string;
  filename: string;
  content_type: string;
  series_name?: string | null;
  occurred_at?: string | null;
  participants?: string[];
}

export const api = {
  listMeetings: (params: { series_id?: string; status?: MeetingStatus } = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v) as [string, string][],
    ).toString();
    return request<{ meetings: Meeting[] }>(`/meetings${q ? `?${q}` : ""}`).then((r) => r.meetings);
  },
  getMeeting: (id: string) => request<Meeting>(`/meetings/${id}`),
  createMeeting: (body: CreateMeetingInput) =>
    request<{ meeting: Meeting; upload_url: string; audio_s3_key: string }>("/meetings", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  processMeeting: (id: string, force = false) =>
    request<{ meeting_id: string; status: MeetingStatus }>(`/meetings/${id}/process`, {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
  deleteMeeting: (id: string) => request<void>(`/meetings/${id}`, { method: "DELETE" }),
  getTranscript: (id: string) => request<Transcript>(`/meetings/${id}/transcript`),
  getSummary: (id: string) => request<MeetingSummary>(`/meetings/${id}/summary`),
  getDropped: (id: string) => request<DroppedArtifact[]>(`/meetings/${id}/dropped`),
  getAudioUrl: (id: string) => request<{ url: string; expires_in: number }>(`/meetings/${id}/audio`),
  search: (query: string, filters: SearchFilters = {}, top_k = 8) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify({ query, filters, top_k }),
    }),
  listSeries: () => request<SeriesSummary[]>("/series"),
  getCommitments: (seriesId: string) => request<CommitmentGraph>(`/series/${seriesId}/commitments`),
  rebuildSeries: (seriesId: string) =>
    request<CommitmentGraph>(`/series/${seriesId}/rebuild`, { method: "POST" }),
  getReversals: (seriesId: string) => request<DecisionReversal[]>(`/series/${seriesId}/reversals`),
  getLedger: (personId: string) => request<PersonLedger>(`/people/${personId}/ledger`),
};

/** PUT the file straight to S3 with the presigned URL, reporting progress. */
export function uploadToS3(
  url: string,
  file: File,
  contentType: string,
  onProgress: (fraction: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("Content-Type", contentType);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new ApiError(xhr.status, `S3 upload failed (${xhr.status})`));
    xhr.onerror = () => reject(new ApiError(0, "S3 upload failed — check the bucket CORS policy"));
    xhr.send(file);
  });
}
