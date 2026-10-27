"""GraphService — build, persist and query the accountability graph for a series."""

from __future__ import annotations

import logging
from collections import Counter

from minuteai.extract.llm_client import LLMClient
from minuteai.graph.decisions import detect_reversals
from minuteai.graph.entity_resolution import MeetingBundle, PeopleIndex, resolve_people
from minuteai.graph.lifecycle import LifecycleTracker
from minuteai.graph.linker import link_commitments
from minuteai.graph.similarity import Similarity
from minuteai.models.graph import CommitmentGraph, LedgerEntry, PersonLedger, SeriesMeetingRef
from minuteai.search.embedder import Embedder, get_embedder
from minuteai.storage.repository import MeetingRepository

log = logging.getLogger(__name__)


class GraphService:
    def __init__(
        self,
        repo: MeetingRepository,
        embedder: Embedder | None = None,
        judge: LLMClient | None = None,
        silence_meetings: int = 2,
    ) -> None:
        self.repo = repo
        self.embedder = embedder or get_embedder()
        self.judge = judge
        self.silence_meetings = silence_meetings

    def bundles(self, series_id: str) -> list[MeetingBundle]:
        out = []
        for m in self.repo.list_series(series_id):
            if m.status != "ready":
                continue
            t = self.repo.get_transcript(m.meeting_id)
            if t is None:
                continue
            out.append(MeetingBundle(m, t, self.repo.get_summary(m.meeting_id)))
        return sorted(out, key=lambda b: b.meeting.occurred_at)

    def build(self, series_id: str, persist: bool = True) -> CommitmentGraph:
        bundles = self.bundles(series_id)
        sim = Similarity(self.embedder)
        people = resolve_people(bundles)
        tracker = LifecycleTracker(sim, self.silence_meetings, judge=self.judge)
        commitments = link_commitments(series_id, bundles, PeopleIndex(people), sim, tracker)
        graph = CommitmentGraph(
            series_id=series_id,
            meetings=[SeriesMeetingRef(meeting_id=b.meeting_id, title=b.meeting.title,
                                       occurred_at=b.meeting.occurred_at) for b in bundles],
            commitments=commitments,
            people=sorted(people.values(), key=lambda p: (p.method == "unresolved", p.person_id)),
            reversals=detect_reversals(series_id, bundles, sim),
        )
        if persist:
            self.repo.put_graph(graph)
        log.info("graph %s: %d meetings, %d commitments, %d reversals", series_id,
                 len(bundles), len(commitments), len(graph.reversals))
        return graph

    def get(self, series_id: str) -> CommitmentGraph:
        return self.repo.get_graph(series_id) or self.build(series_id)

    def rebuild_for_meeting(self, meeting_id: str) -> None:
        meeting = self.repo.get_meeting(meeting_id)
        if meeting and meeting.series_id:
            self.build(meeting.series_id)

    def ledger(self, person_id: str) -> PersonLedger:
        entries: list[LedgerEntry] = []
        name = None
        for series_id, person in self.repo.list_person_series(person_id):
            name = name or person.display_name
            graph = self.repo.get_graph(series_id)
            for c in graph.commitments if graph else []:
                if c.owner_id == person_id:
                    entries.append(LedgerEntry(commitment=c, series_id=series_id))
        entries.sort(key=lambda e: e.commitment.created_at, reverse=True)
        counts = Counter(e.commitment.state for e in entries)
        return PersonLedger(person_id=person_id, display_name=name, entries=entries,
                            counts=dict(counts))
