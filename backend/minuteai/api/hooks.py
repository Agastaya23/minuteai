"""Post-processing hooks run after a meeting reaches `ready`."""

from __future__ import annotations

from collections.abc import Callable

from minuteai.api.processing import Services


def default_hooks(services: Services) -> list[Callable[[str], None]]:
    hooks: list[Callable[[str], None]] = []
    if services.search is not None:
        hooks.append(services.search.index_meeting)
    if services.graph is not None:
        hooks.append(services.graph.rebuild_for_meeting)
    return hooks
