"""Prompt templates. Every extraction prompt follows the same rules:

* utterances are rendered as ``[utt_id] SPEAKER_XX (Name) (mm:ss): text``;
* every returned object must include ``evidence`` with utt_ids + verbatim quotes;
* return an empty list rather than inventing content;
* action items with no clearly identifiable owner get ``owner: null`` and
  ``owner_resolution_method: "unresolved"``.

Templates use str.format placeholders; literal braces are doubled.
"""

from __future__ import annotations

_EVIDENCE_RULES = """\
Rules:
- Use ONLY the transcript below. Do not use outside knowledge.
- Every object you return MUST have an "evidence" array. Each entry gives the utt_id (the
  number in square brackets) of an utterance that supports the claim, and "quote": a short
  verbatim excerpt copied exactly from that utterance's text (not paraphrased).
- Cite every utterance needed to support the claim, and only those.
- Every name, number, date and technical term in your text must appear in the cited
  utterances.
- If nothing qualifies, return an empty list. Never invent content to fill the list."""

SUMMARY_PROMPT = """\
Summarize this portion of a meeting transcript as 2-6 concise, factual sentences covering
what was discussed and concluded.

{rules}

Return JSON: {{"sentences": [{{"text": str, "evidence": [{{"utt_id": int, "quote": str}}]}}]}}

Transcript:
{transcript}
"""
"""Returns LLMSummaryResponse: grounded abstract sentences for one chunk."""

DECISIONS_PROMPT = """\
List the decisions made in this portion of a meeting. A decision is a choice the group
settled on, proposed, explicitly deferred, or reversed - not a task or an open question.

{rules}
- "participants": speaker labels (e.g. "SPEAKER_01") of people who proposed or agreed to it.
  Cite at least one utterance spoken by each participant.
- "status": "agreed" if settled, "proposed" if suggested but not settled, "deferred" if
  explicitly postponed, "reversed" if it overturns an earlier decision.
- "confidence": 0.0-1.0, how clearly the transcript shows this decision.

Return JSON: {{"decisions": [{{"statement": str, "rationale": str|null, "participants": [str],
"status": str, "confidence": float, "evidence": [{{"utt_id": int, "quote": str}}]}}]}}

Transcript:
{transcript}
"""
"""Returns LLMDecisionsResponse: decisions for one chunk."""

ACTION_ITEMS_PROMPT = """\
List the action items (concrete tasks someone is expected to do after the meeting) in this
portion of a meeting.

{rules}
- Owner: set "owner" to the person's name and "owner_speaker_label" to their speaker label
  ONLY if the transcript clearly identifies who will do it:
    "self_commitment"   - the speaker commits themselves ("I will ...")
    "assigned_by_other" - someone assigns it and the assignee is identified ("Ananya, can you")
    "explicit_name"     - the owner is named explicitly in some other way
  If no owner is clearly identifiable (e.g. "someone should ..."), set "owner": null,
  "owner_speaker_label": null and "owner_resolution_method": "unresolved". Do NOT guess.
- When an owner is set, cite the utterance where the owner commits or is assigned, and the
  owner's own acceptance if there is one.
- "due_raw": the deadline phrase copied verbatim (e.g. "by Friday"), or null.
- "confidence" / "owner_confidence": 0.0-1.0.

Return JSON: {{"action_items": [{{"description": str, "owner": str|null,
"owner_speaker_label": str|null, "owner_resolution_method": str, "owner_confidence": float,
"due_raw": str|null, "confidence": float, "evidence": [{{"utt_id": int, "quote": str}}]}}]}}

Transcript:
{transcript}
"""
"""Returns LLMActionItemsResponse: action items for one chunk."""

TOPIC_TITLES_PROMPT = """\
A meeting transcript has been split into numbered segments. Give each segment a short topic
title (3-8 words) describing what it is about. Use only words supported by the segment.
Return one entry per segment, using the same segment numbers.

Return JSON: {{"topics": [{{"segment": int, "title": str}}]}}

{segments}
"""
"""Returns LLMTopicTitlesResponse. Segment boundaries are computed deterministically."""

SEARCH_ANSWER_PROMPT = """\
Answer the question using ONLY the numbered meeting excerpts below.

- If the excerpts do not contain the answer, set "answerable": false, "answer" to a short
  statement that the meetings do not cover this, and "evidence": [].
- Otherwise answer in 1-3 sentences. Every claim must be supported by an excerpt: cite it in
  "evidence" with "utt_id" set to the excerpt number in square brackets and "quote" a
  verbatim excerpt from it.

Question: {query}

Excerpts:
{excerpts}

Return JSON: {{"answerable": bool, "answer": str, "evidence": [{{"utt_id": int, "quote": str}}]}}
"""
"""Returns LLMAnswer for natural-language search."""

BASELINE_SINGLE_PROMPT = """\
Read this meeting transcript and write the meeting minutes: a short summary, the decisions
made, and the action items.

Return JSON: {{"summary": [str], "decisions": [str], "action_items": [str]}}

Transcript:
{transcript}
"""
"""Returns LLMBaselineResponse. Ablation config A: one prompt, no chunking, no evidence."""

COMMITMENT_STATUS_PROMPT = """\
Earlier, this commitment was recorded:
  "{commitment}" (owner: {owner}; due: {due})

Here is a later meeting's transcript. Determine what this meeting says about that commitment:
- "fulfilled": someone explicitly states it is done/completed/merged/shipped.
- "slipped": it is re-raised with a new deadline or an explicit delay.
- "reversed": a decision explicitly cancels or contradicts it.
- "in_progress": it is mentioned as ongoing, with no new deadline.
- "not_mentioned": the meeting does not refer to it.
{rules}
- "new_due_raw": verbatim new deadline phrase if slipped, else null.

Return JSON: {{"status": str, "new_due_raw": str|null,
"evidence": [{{"utt_id": int, "quote": str}}]}}

Transcript:
{transcript}
"""
"""Returns LLMCommitmentStatus (optional LLM judge for the accountability graph)."""


def render(template: str, **kwargs) -> str:
    return template.format(rules=_EVIDENCE_RULES, **kwargs)
