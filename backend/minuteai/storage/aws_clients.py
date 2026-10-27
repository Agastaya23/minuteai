"""Factories for non-storage AWS service clients, so boto3 stays inside storage/."""

from __future__ import annotations

import boto3

from minuteai.config import get_settings


def transcribe_client():
    return boto3.client("transcribe", region_name=get_settings().aws_region)
