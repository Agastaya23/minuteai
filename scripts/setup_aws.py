"""Provision MinuteAI's AWS resources and print the IAM policy the backend needs.

    cd backend && uv run python ../scripts/setup_aws.py --bucket my-minuteai --region ap-south-1
    cd backend && uv run python ../scripts/setup_aws.py --print-policy      # policy only

Creates (idempotently):
  * the S3 bucket, with public access blocked and a CORS rule allowing the
    frontend origin(s) to PUT directly with presigned URLs and GET for playback
  * the DynamoDB table with GSI1 (on-demand billing)
Then prints the least-privilege IAM policy JSON to attach to the backend role.
"""

from __future__ import annotations

import argparse
import json
import os

from botocore.exceptions import ClientError

from minuteai.config import get_settings


def iam_policy(bucket: str, table: str, region: str, account: str = "*",
               transcribe: bool = True) -> dict:
    table_arn = f"arn:aws:dynamodb:{region}:{account}:table/{table}"
    statements = [
        {
            "Sid": "MeetingAudioTranscriptsIndexes",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            "Resource": f"arn:aws:s3:::{bucket}/*",
        },
        {
            "Sid": "BucketLevel",
            "Effect": "Allow",
            "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
            "Resource": f"arn:aws:s3:::{bucket}",
        },
        {
            "Sid": "SingleTable",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem",
                "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:BatchWriteItem",
                "dynamodb:BatchGetItem", "dynamodb:DescribeTable",
            ],
            "Resource": [table_arn, f"{table_arn}/index/*"],
        },
    ]
    if transcribe:
        statements.append({
            "Sid": "OptionalAwsTranscribeBackend",
            "Effect": "Allow",
            "Action": ["transcribe:StartTranscriptionJob", "transcribe:GetTranscriptionJob"],
            "Resource": "*",
        })
    return {"Version": "2012-10-17", "Statement": statements}


def setup_bucket(s3, bucket: str, region: str, origins: list[str]) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket {bucket}: exists")
    except ClientError:
        kwargs: dict = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        print(f"bucket {bucket}: created")
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [{
        "AllowedOrigins": origins,
        "AllowedMethods": ["PUT", "GET", "HEAD"],
        "AllowedHeaders": ["*"],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3600,
    }]})
    print(f"bucket {bucket}: CORS allows {', '.join(origins)}")


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bucket", default=s.s3_bucket)
    p.add_argument("--table", default=s.ddb_table)
    p.add_argument("--region", default=s.aws_region)
    p.add_argument("--origin", action="append", dest="origins",
                   help="frontend origin allowed to upload (repeatable)")
    p.add_argument("--print-policy", action="store_true", help="only print the IAM policy")
    p.add_argument("--no-transcribe", action="store_true", help="omit AWS Transcribe actions")
    args = p.parse_args()
    origins = args.origins or ["http://localhost:3000"]

    account = "*"
    if not args.print_policy:
        import boto3

        # setup targets real AWS: ignore local endpoint overrides
        os.environ.pop("DDB_ENDPOINT_URL", None)
        os.environ["DDB_TABLE"], os.environ["AWS_REGION"] = args.table, args.region
        get_settings.cache_clear()
        get_settings().ddb_endpoint_url = None
        account = boto3.client("sts", region_name=args.region).get_caller_identity()["Account"]
        setup_bucket(boto3.client("s3", region_name=args.region), args.bucket, args.region,
                     origins)
        from minuteai.storage.schema import create_table

        created = create_table(args.table)
        print(f"table {args.table}: {'created' if created else 'exists'}")

    print("\nAttach this IAM policy to the backend's role/user:\n")
    print(json.dumps(iam_policy(args.bucket, args.table, args.region, account,
                                not args.no_transcribe), indent=2))


if __name__ == "__main__":
    main()
