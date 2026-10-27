"""Post-processing hooks run after a meeting reaches `ready`."""

from __future__ import annotations

from collections.abc import Callable

from minuteai.api.processing import Services


def default_hooks(services: Services) -> list[Callable[[str], None]]:
    return []
