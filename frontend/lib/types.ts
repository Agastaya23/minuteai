// Mirrors backend/minuteai/models — the API contract.

export type MeetingStatus = "uploaded" | "transcribing" | "extracting" | "ready" | "failed";

export interface Word {
  text: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Utterance {
  utt_id: number;
  speaker_label: string;
  speaker_name: string | null;
  text: string;
  start: number;
  end: number;
  words: Word[];
}

export interface Transcript {
  meeting_id: string;
  utterances: Utterance[];
  language: string;
  duration_sec: number;
  backend: string;
  model_version: string;
}

export interface EvidenceSpan {
  utt_id: number;
  start_sec: number;
  end_sec: number;
  speaker_label: string;
  quote: string;
  meeting_id?: string | null;
}

export interface GroundedSentence {
  text: string;
  evidence: EvidenceSpan[];
  validated: boolean;
  flags: string[];
}

export interface Decision {
  decision_id: string;
  statement: string;
  rationale: string | null;
  participants: string[];
  evidence: EvidenceSpan[];
  confidence: number;
  status: "proposed" | "agreed" | "deferred" | "reversed";
  validated: boolean;
  flags: string[];
}

export type OwnerResolution = "explicit_name" | "self_commitment" | "assigned_by_other" | "unresolved";

export interface ActionItem {
  item_id: string;
  description: string;
  owner: string | null;
  owner_speaker_label: string | null;
  owner_confidence: number;
  owner_resolution_method: OwnerResolution;
  due_date: string | null;
  due_raw: string | null;
  evidence: EvidenceSpan[];
  confidence: number;
  validated: boolean;
  flags: string[];
}

export interface Topic {
  topic_id: string;
  title: string;
  start_utt: number;
  end_utt: number;
  summary: GroundedSentence[];
}

export interface MeetingSummary {
  meeting_id: string;
  abstract: GroundedSentence[];
  topics: Topic[];
  decisions: Decision[];
  action_items: ActionItem[];
  participants: string[];
  generated_at: string;
  model_version: string;
  grounding_mode: string | null;
}

export interface Meeting {
  meeting_id: string;
  title: string;
  series_id: string | null;
  occurred_at: string;
  duration_sec: number;
  audio_s3_key: string | null;
  participants: string[];
  status: MeetingStatus;
  created_at: string;
  error: string | null;
}

export interface CheckFailure {
  check: string;
  reason: string;
  detail: string;
}

export interface DroppedArtifact {
  meeting_id: string;
  artifact_type: string;
  artifact_id: string;
  text: string;
  evidence: EvidenceSpan[];
  failures: CheckFailure[];
  confidence: number;
  mode: string;
  dropped_at: string;
}

export interface SearchFilters {
  date_from?: string | null;
  date_to?: string | null;
  series_id?: string | null;
  participant?: string | null;
  artifact_types?: ("utterances" | "decision" | "action_item")[] | null;
}

export interface SearchSource {
  n: number;
  meeting_id: string;
  meeting_title: string;
  occurred_at: string | null;
  artifact_type: string;
  artifact_id: string | null;
  start_utt: number;
  end_utt: number;
  start_sec: number;
  end_sec: number;
  text: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  answerable: boolean;
  answer: string;
  evidence: EvidenceSpan[];
  validated: boolean;
  flags: string[];
  sources: SearchSource[];
}

export type CommitmentState =
  | "PROPOSED"
  | "COMMITTED"
  | "FULFILLED"
  | "SLIPPED"
  | "SILENTLY_DROPPED"
  | "REVERSED";

export interface Transition {
  commitment_id: string;
  from_state: CommitmentState | null;
  to_state: CommitmentState;
  meeting_id: string;
  occurred_at: string;
  reason: string;
  evidence: EvidenceSpan[];
}

export interface CommitmentMention {
  meeting_id: string;
  item_id: string;
  occurred_at: string;
  description: string;
  owner: string | null;
  due_date: string | null;
  due_raw: string | null;
  evidence: EvidenceSpan[];
  similarity: number;
}

export interface Commitment {
  commitment_id: string;
  series_id: string;
  description: string;
  owner_id: string | null;
  owner_name: string | null;
  origin_meeting_id: string;
  created_at: string;
  due_date: string | null;
  state: CommitmentState;
  mentions: CommitmentMention[];
  transitions: Transition[];
  timeline: Record<string, CommitmentState>;
}

export interface PersonIdentity {
  person_id: string;
  display_name: string | null;
  aliases: string[];
  speaker_labels: Record<string, string>;
  method: string;
  confidence: number;
}

export interface DecisionRef {
  meeting_id: string;
  decision_id: string;
  statement: string;
  occurred_at: string;
  evidence: EvidenceSpan[];
}

export interface DecisionReversal {
  reversal_id: string;
  series_id: string;
  original: DecisionRef;
  reversing: DecisionRef;
  similarity: number;
  reason: string;
}

export interface SeriesMeetingRef {
  meeting_id: string;
  title: string;
  occurred_at: string;
}

export interface CommitmentGraph {
  series_id: string;
  meetings: SeriesMeetingRef[];
  commitments: Commitment[];
  people: PersonIdentity[];
  reversals: DecisionReversal[];
}

export interface PersonLedger {
  person_id: string;
  display_name: string | null;
  entries: { commitment: Commitment; series_id: string }[];
  counts: Record<string, number>;
}

export interface SeriesSummary {
  series_id: string;
  meetings: number;
  last_occurred_at: string | null;
}
