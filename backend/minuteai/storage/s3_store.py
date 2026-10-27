"""S3 access: audio, transcript JSON, search indexes, presigned URLs."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from minuteai.config import get_settings
from minuteai.models.transcript import Transcript


def transcript_key(meeting_id: str) -> str:
    return f"transcripts/{meeting_id}.json"


def audio_key(meeting_id: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower() or ".bin"
    return f"audio/{meeting_id}/original{suffix}"


class S3Store:
    def __init__(self, bucket: str | None = None) -> None:
        s = get_settings()
        self.bucket = bucket or s.s3_bucket
        self.region = s.aws_region
        self.client = boto3.client(
            "s3",
            region_name=s.aws_region,
            endpoint_url=s.s3_endpoint_url,
            config=Config(signature_version="s3v4"),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            kwargs: dict = {"Bucket": self.bucket}
            if self.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self.client.create_bucket(**kwargs)

    def ping(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)

    # -- generic -----------------------------------------------------------
    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put_bytes(self, key: str, body: bytes, content_type: str = "application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def put_text(self, key: str, text: str, content_type: str = "application/json") -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type)

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")

    def download_file(self, key: str, dest: Path) -> Path:
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def upload_file(self, src: Path, key: str) -> None:
        self.client.upload_file(str(src), self.bucket, key)

    # -- audio -------------------------------------------------------------
    def put_audio(self, key: str, path: Path) -> None:
        self.upload_file(path, key)

    def get_audio(self, key: str, dest: Path) -> Path:
        return self.download_file(key, dest)

    # -- transcripts -------------------------------------------------------
    def put_transcript(self, transcript: Transcript) -> str:
        key = transcript_key(transcript.meeting_id)
        self.put_text(key, transcript.model_dump_json())
        return key

    def get_transcript(self, meeting_id: str) -> Transcript | None:
        key = transcript_key(meeting_id)
        if not self.exists(key):
            return None
        return Transcript.model_validate_json(self.get_text(key))

    # -- presigned URLs ----------------------------------------------------
    def presigned_upload_url(self, key: str, content_type: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires,
        )

    def presigned_download_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires
        )
