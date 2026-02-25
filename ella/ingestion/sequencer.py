"""Sort a list of MessageUnit objects by message_id to preserve original order."""
from __future__ import annotations

from ella.agents.protocol import MessageUnit


def sort_by_message_id(units: list[MessageUnit]) -> list[MessageUnit]:
    return sorted(units, key=lambda u: u.message_id)
