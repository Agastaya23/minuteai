"""Idempotently create the MinuteAI DynamoDB table (+ GSI1) and S3 bucket.

Targets DynamoDB Local when DDB_ENDPOINT_URL is set, real AWS otherwise.
    cd backend && uv run python ../scripts/create_table.py
"""

from __future__ import annotations

import argparse

from minuteai.config import get_settings
from minuteai.storage.s3_store import S3Store
from minuteai.storage.schema import create_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bucket", action="store_true", help="only create the table")
    args = parser.parse_args()

    s = get_settings()
    target = s.ddb_endpoint_url or f"AWS {s.aws_region}"
    created = create_table()
    print(f"table {s.ddb_table!r} on {target}: {'created' if created else 'already exists'}")
    if not args.skip_bucket:
        S3Store().ensure_bucket()
        print(f"bucket {s.s3_bucket!r}: ready")


if __name__ == "__main__":
    main()
