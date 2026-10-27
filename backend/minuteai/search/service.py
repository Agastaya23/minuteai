"""SearchService — owns the global index, keeps it in sync, answers queries."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from minuteai.extract.llm_client import LLMClient
from minuteai.models.search import SearchFilters
from minuteai.models.transcript import Transcript
from minuteai.search.answerer import Answer, Answerer
from minuteai.search.embedder import Embedder, get_embedder
from minuteai.search.hybrid import HybridRetriever
from minuteai.search.index import SearchIndex
from minuteai.storage.repository import MeetingRepository

log = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        repo: MeetingRepository,
        embedder: Embedder | None = None,
        llm_factory: Callable[[], LLMClient] = LLMClient,
    ) -> None:
        self.repo = repo
        self.embedder = embedder or get_embedder()
        self.llm_factory = llm_factory
        self.index = SearchIndex(self.embedder, s3=repo.s3)
        self.retriever = HybridRetriever(self.index)
        self._loaded = False
        self._load_lock = threading.Lock()
        self._transcripts: dict[str, Transcript] = {}

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            if not self.index.load():
                self.rebuild()
            self._loaded = True

    def rebuild(self) -> int:
        """Re-index every ready meeting from storage."""
        self.index = SearchIndex(self.embedder, s3=self.repo.s3)
        self.retriever = HybridRetriever(self.index)
        for m in self.repo.list_by_status("ready"):
            self._add(m.meeting_id)
        self.index.save()
        log.info("search index rebuilt: %d entries", len(self.index))
        return len(self.index)

    def _add(self, meeting_id: str) -> int:
        meeting = self.repo.get_meeting(meeting_id)
        transcript = self.repo.get_transcript(meeting_id)
        if meeting is None or transcript is None:
            return 0
        self._transcripts[meeting_id] = transcript
        return self.index.add_meeting(meeting, transcript, self.repo.get_summary(meeting_id))

    def index_meeting(self, meeting_id: str) -> int:
        self.ensure_loaded()
        n = self._add(meeting_id)
        self.index.save()
        return n

    def remove_meeting(self, meeting_id: str) -> None:
        self.ensure_loaded()
        self.index.remove_meeting(meeting_id)
        self._transcripts.pop(meeting_id, None)
        self.index.save()

    def _transcript(self, meeting_id: str) -> Transcript | None:
        if meeting_id not in self._transcripts:
            t = self.repo.get_transcript(meeting_id)
            if t is None:
                return None
            self._transcripts[meeting_id] = t
        return self._transcripts[meeting_id]

    def answer(self, query: str, filters: SearchFilters | None = None, k: int = 8) -> Answer:
        self.ensure_loaded()
        answerer = Answerer(self.retriever, self.llm_factory(), self._transcript)
        return answerer.answer(query, filters, k)
