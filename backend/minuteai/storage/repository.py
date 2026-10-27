"""MeetingRepository — the only DynamoDB access point. Returns pydantic models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TypeVar

from boto3.dynamodb.conditions import Key
from pydantic import BaseModel

from minuteai.config import get_settings
from minuteai.models.artifacts import ActionItem, Decision, MeetingSummary
from minuteai.models.graph import (
    Commitment,
    CommitmentGraph,
    DecisionReversal,
    PersonIdentity,
    Transition,
)
from minuteai.models.grounding import DroppedArtifact
from minuteai.models.meeting import Meeting, MeetingStatus, utcnow
from minuteai.models.transcript import Transcript
from minuteai.storage.s3_store import S3Store
from minuteai.storage.schema import (
    GSI1,
    commitment_pk,
    dynamodb_resource,
    meeting_pk,
    owner_gsi_pk,
    series_pk,
    status_gsi_pk,
)

M = TypeVar("M", bound=BaseModel)

# DynamoDB's hard item limit is 400 KB; leave headroom for keys/attributes.
MAX_INLINE_BYTES = 350_000


class NotFoundError(KeyError):
    pass


class MeetingRepository:
    def __init__(self, s3: S3Store | None = None, table_name: str | None = None) -> None:
        self.table = dynamodb_resource().Table(table_name or get_settings().ddb_table)
        self.s3 = s3 or S3Store()

    # -- helpers -----------------------------------------------------------
    def _put(self, pk: str, sk: str, model: BaseModel | str, **extra: str) -> None:
        data = model if isinstance(model, str) else model.model_dump_json()
        item = {"PK": pk, "SK": sk, "data": data, **{k: v for k, v in extra.items() if v}}
        self.table.put_item(Item=item)

    def _get_raw(self, pk: str, sk: str) -> dict | None:
        return self.table.get_item(Key={"PK": pk, "SK": sk}).get("Item")

    def _get(self, pk: str, sk: str, model: type[M]) -> M | None:
        item = self._get_raw(pk, sk)
        return model.model_validate_json(item["data"]) if item else None

    def _query_prefix(self, pk: str, prefix: str, model: type[M]) -> list[M]:
        items = self._paginate(
            KeyConditionExpression=Key("PK").eq(pk) & Key("SK").begins_with(prefix)
        )
        return [model.model_validate_json(i["data"]) for i in items]

    def _query_gsi(self, gsi_pk: str, newest_first: bool = True) -> list[dict]:
        return self._paginate(
            IndexName=GSI1,
            KeyConditionExpression=Key("GSI1PK").eq(gsi_pk),
            ScanIndexForward=not newest_first,
        )

    def _paginate(self, **kwargs) -> list[dict]:
        items: list[dict] = []
        while True:
            resp = self.table.query(**kwargs)
            items.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                return items
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def _delete_prefix(self, pk: str, prefix: str) -> None:
        condition = Key("PK").eq(pk)
        if prefix:
            condition = condition & Key("SK").begins_with(prefix)
        items = self._paginate(KeyConditionExpression=condition, ProjectionExpression="PK, SK")
        with self.table.batch_writer() as batch:
            for i in items:
                batch.delete_item(Key={"PK": i["PK"], "SK": i["SK"]})

    def ping(self) -> None:
        self.table.load()

    # -- meetings (patterns 1, 6, 8) --------------------------------------
    def put_meeting(self, meeting: Meeting) -> Meeting:
        pk = meeting_pk(meeting.meeting_id)
        self._put(
            pk, "META", meeting,
            GSI1PK=status_gsi_pk(meeting.status), GSI1SK=meeting.created_at.isoformat(),
        )
        if meeting.series_id:
            self._put(
                pk, "SERIES", meeting.meeting_id,
                GSI1PK=series_pk(meeting.series_id), GSI1SK=meeting.occurred_at.isoformat(),
            )
        else:
            self.table.delete_item(Key={"PK": pk, "SK": "SERIES"})
        return meeting

    def get_meeting(self, meeting_id: str) -> Meeting | None:
        return self._get(meeting_pk(meeting_id), "META", Meeting)

    def require_meeting(self, meeting_id: str) -> Meeting:
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            raise NotFoundError(meeting_id)
        return meeting

    def update_status(
        self, meeting_id: str, status: MeetingStatus, error: str | None = None, **fields
    ) -> Meeting:
        meeting = self.require_meeting(meeting_id)
        updated = meeting.model_copy(update={"status": status, "error": error, **fields})
        return self.put_meeting(updated)

    def list_meetings(self) -> list[Meeting]:
        statuses: Iterable[MeetingStatus] = (
            "uploaded", "transcribing", "extracting", "ready", "failed"
        )
        meetings = [m for s in statuses for m in self.list_by_status(s)]
        return sorted(meetings, key=lambda m: m.occurred_at, reverse=True)

    def list_by_status(self, status: MeetingStatus) -> list[Meeting]:
        return [
            Meeting.model_validate_json(i["data"]) for i in self._query_gsi(status_gsi_pk(status))
        ]

    def list_series(self, series_id: str) -> list[Meeting]:
        """Meetings in a series, newest first."""
        ids = [i["data"] for i in self._query_gsi(series_pk(series_id))]
        meetings = [self.get_meeting(mid) for mid in ids]
        return [m for m in meetings if m is not None]

    def list_series_ids(self) -> list[str]:
        return sorted({m.series_id for m in self.list_meetings() if m.series_id})

    def delete_meeting(self, meeting_id: str) -> None:
        self._delete_prefix(meeting_pk(meeting_id), "")

    # -- transcript (pattern 2) — body lives in S3 ------------------------
    def put_transcript(self, transcript: Transcript) -> None:
        key = self.s3.put_transcript(transcript)
        pointer = {
            "s3_key": key,
            "utterances": len(transcript.utterances),
            "duration_sec": transcript.duration_sec,
            "backend": transcript.backend,
            "model_version": transcript.model_version,
        }
        self._put(meeting_pk(transcript.meeting_id), "TRANSCRIPT", json.dumps(pointer))

    def get_transcript(self, meeting_id: str) -> Transcript | None:
        if self._get_raw(meeting_pk(meeting_id), "TRANSCRIPT") is None:
            return None
        return self.s3.get_transcript(meeting_id)

    # -- summary + per-item rows (patterns 3, 4, 5, 7) --------------------
    def put_summary(self, summary: MeetingSummary) -> None:
        mid = summary.meeting_id
        pk = meeting_pk(mid)
        body = summary.model_dump_json()
        if len(body.encode()) > MAX_INLINE_BYTES:
            key = f"summaries/{mid}.json"
            self.s3.put_text(key, body)
            self.table.put_item(Item={"PK": pk, "SK": "SUMMARY", "s3_key": key})
        else:
            self._put(pk, "SUMMARY", body)

        meeting = self.get_meeting(mid)
        occurred = (meeting.occurred_at if meeting else summary.generated_at).isoformat()
        self._delete_prefix(pk, "ACTION#")
        self._delete_prefix(pk, "DECISION#")
        with self.table.batch_writer() as batch:
            for item in summary.action_items:
                row = {"PK": pk, "SK": f"ACTION#{item.item_id}", "data": item.model_dump_json()}
                if item.owner:
                    row["GSI1PK"] = owner_gsi_pk(item.owner)
                    row["GSI1SK"] = f"{occurred}#{mid}#{item.item_id}"
                    row["meeting_id"] = mid
                batch.put_item(Item=row)
            for d in summary.decisions:
                batch.put_item(
                    Item={"PK": pk, "SK": f"DECISION#{d.decision_id}", "data": d.model_dump_json()}
                )

    def get_summary(self, meeting_id: str) -> MeetingSummary | None:
        item = self._get_raw(meeting_pk(meeting_id), "SUMMARY")
        if item is None:
            return None
        body = self.s3.get_text(item["s3_key"]) if "s3_key" in item else item["data"]
        return MeetingSummary.model_validate_json(body)

    def list_action_items(self, meeting_id: str) -> list[ActionItem]:
        return self._query_prefix(meeting_pk(meeting_id), "ACTION#", ActionItem)

    def list_decisions(self, meeting_id: str) -> list[Decision]:
        return self._query_prefix(meeting_pk(meeting_id), "DECISION#", Decision)

    def list_by_owner(self, owner: str) -> list[tuple[str, ActionItem]]:
        """(meeting_id, item) for every action item owned by `owner`, newest first."""
        return [
            (i["meeting_id"], ActionItem.model_validate_json(i["data"]))
            for i in self._query_gsi(owner_gsi_pk(owner))
        ]

    # -- dropped artifacts -------------------------------------------------
    def put_dropped(self, meeting_id: str, dropped: list[DroppedArtifact]) -> None:
        body = json.dumps([d.model_dump(mode="json") for d in dropped])
        self._put(meeting_pk(meeting_id), "DROPPED", body)

    def get_dropped(self, meeting_id: str) -> list[DroppedArtifact]:
        item = self._get_raw(meeting_pk(meeting_id), "DROPPED")
        if item is None:
            return []
        return [DroppedArtifact.model_validate(d) for d in json.loads(item["data"])]

    # -- accountability graph ---------------------------------------------
    def put_graph(self, graph: CommitmentGraph) -> None:
        spk = series_pk(graph.series_id)
        for prefix in ("COMMITMENT#", "REVERSAL#", "PERSON#"):
            self._delete_prefix(spk, prefix)
        with self.table.batch_writer() as batch:
            for c in graph.commitments:
                stored = c.model_copy(update={"transitions": []})
                batch.put_item(
                    Item={"PK": spk, "SK": f"COMMITMENT#{c.commitment_id}",
                          "data": stored.model_dump_json()}
                )
            for r in graph.reversals:
                batch.put_item(
                    Item={"PK": spk, "SK": f"REVERSAL#{r.reversal_id}",
                          "data": r.model_dump_json()}
                )
            for p in graph.people:
                batch.put_item(
                    Item={"PK": spk, "SK": f"PERSON#{p.person_id}", "data": p.model_dump_json(),
                          "GSI1PK": f"PERSON#{p.person_id}", "GSI1SK": spk}
                )
        for c in graph.commitments:
            cpk = commitment_pk(c.commitment_id)
            self._delete_prefix(cpk, "TRANSITION#")
            with self.table.batch_writer() as batch:
                for n, t in enumerate(c.transitions):
                    batch.put_item(
                        Item={"PK": cpk, "SK": f"TRANSITION#{t.occurred_at.isoformat()}#{n:03d}",
                              "data": t.model_dump_json()}
                    )
        self._put(spk, "GRAPHMETA", graph.model_copy(
            update={"commitments": [], "reversals": [], "people": []}
        ), GSI1PK="GRAPHS", GSI1SK=utcnow().isoformat())

    def get_graph(self, series_id: str) -> CommitmentGraph | None:
        spk = series_pk(series_id)
        meta = self._get(spk, "GRAPHMETA", CommitmentGraph)
        if meta is None:
            return None
        commitments = [
            c.model_copy(update={"transitions": self.list_transitions(c.commitment_id)})
            for c in self._query_prefix(spk, "COMMITMENT#", Commitment)
        ]
        return meta.model_copy(update={
            "commitments": commitments,
            "reversals": self._query_prefix(spk, "REVERSAL#", DecisionReversal),
            "people": self._query_prefix(spk, "PERSON#", PersonIdentity),
        })

    def list_transitions(self, commitment_id: str) -> list[Transition]:
        return self._query_prefix(commitment_pk(commitment_id), "TRANSITION#", Transition)

    def list_person_series(self, person_id: str) -> list[tuple[str, PersonIdentity]]:
        return [
            (i["PK"].removeprefix("SERIES#"), PersonIdentity.model_validate_json(i["data"]))
            for i in self._query_gsi(f"PERSON#{person_id}")
        ]
