"""Single-table DynamoDB design for MinuteAI.

Table ``minuteai`` — composite key (PK, SK) plus one GSI (GSI1PK, GSI1SK).
Every item stores its pydantic model as a JSON string in ``data`` so the
repository never hands raw attribute maps to callers.

| Item              | PK                     | SK                        | GSI1PK                    | GSI1SK                        | Notes |
|-------------------|------------------------|---------------------------|---------------------------|-------------------------------|-------|
| Meeting           | MEETING#<id>           | META                      | STATUS#<status>           | <created_at ISO>              | pattern 1, 8 |
| Series link       | MEETING#<id>           | SERIES                    | SERIES#<series_id>        | <occurred_at ISO>             | pattern 6; data = meeting_id |
| Transcript ptr    | MEETING#<id>           | TRANSCRIPT                | —                         | —                             | pattern 2; body in S3 transcripts/<id>.json |
| Summary           | MEETING#<id>           | SUMMARY                   | —                         | —                             | pattern 3; offloaded to S3 if > 350 KB |
| Action item       | MEETING#<id>           | ACTION#<item_id>          | OWNER#<normalized owner>  | <occurred_at ISO>#<item_id>   | pattern 4, 7; GSI only if owner resolved |
| Decision          | MEETING#<id>           | DECISION#<decision_id>    | —                         | —                             | pattern 5 |
| Dropped artifacts | MEETING#<id>           | DROPPED                   | —                         | —                             | grounding-filter rejects |
| Commitment        | SERIES#<series_id>     | COMMITMENT#<id>           | —                         | —                             | accountability graph |
| Transition        | COMMITMENT#<id>        | TRANSITION#<ts ISO>       | —                         | —                             | lifecycle evidence |
| Reversal          | SERIES#<series_id>     | REVERSAL#<id>             | —                         | —                             | decision reversals |
| Person            | SERIES#<series_id>     | PERSON#<person_id>        | PERSON#<person_id>        | SERIES#<series_id>            | resolved identities |

Access patterns
1. Meeting by id                → GetItem PK=MEETING#<id>, SK=META
2. Transcript for meeting       → GetItem PK=MEETING#<id>, SK=TRANSCRIPT (+ S3 fetch)
3. Summary for meeting          → GetItem PK=MEETING#<id>, SK=SUMMARY
4. Action items for meeting     → Query PK=MEETING#<id>, SK begins_with ACTION#
5. Decisions for meeting        → Query PK=MEETING#<id>, SK begins_with DECISION#
6. Meetings in series, newest   → Query GSI1 GSI1PK=SERIES#<id>, ScanIndexForward=False
7. Action items by owner        → Query GSI1 GSI1PK=OWNER#<normalized_name>
8. Meetings by status           → Query GSI1 GSI1PK=STATUS#<status>
"""

from __future__ import annotations

import re

import boto3
from botocore.exceptions import ClientError

from minuteai.config import get_settings

GSI1 = "GSI1"


def meeting_pk(meeting_id: str) -> str:
    return f"MEETING#{meeting_id}"


def series_pk(series_id: str) -> str:
    return f"SERIES#{series_id}"


def owner_gsi_pk(owner: str) -> str:
    return f"OWNER#{normalize_name(owner)}"


def status_gsi_pk(status: str) -> str:
    return f"STATUS#{status}"


def commitment_pk(commitment_id: str) -> str:
    return f"COMMITMENT#{commitment_id}"


def normalize_name(name: str) -> str:
    """Owner names are matched case- and punctuation-insensitively."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def dynamodb_resource():
    s = get_settings()
    return boto3.resource("dynamodb", region_name=s.aws_region, endpoint_url=s.ddb_endpoint_url)


def create_table(table_name: str | None = None) -> bool:
    """Create the table + GSI1 if missing. Returns True if it was created."""
    s = get_settings()
    name = table_name or s.ddb_table
    ddb = dynamodb_resource()
    try:
        ddb.meta.client.describe_table(TableName=name)
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    table = ddb.create_table(
        TableName=name,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": a, "AttributeType": "S"} for a in ("PK", "SK", "GSI1PK", "GSI1SK")
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": GSI1,
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )
    table.wait_until_exists()
    return True
