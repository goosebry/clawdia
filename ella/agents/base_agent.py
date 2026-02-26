from __future__ import annotations

from abc import ABC, abstractmethod

from ella.agents.protocol import HandoffMessage, UserTask


class BaseAgent(ABC):
    """All agents implement handle() which receives a message and returns one or more handoffs."""

    @abstractmethod
    async def handle(self, message: UserTask | HandoffMessage) -> list[HandoffMessage]:
        ...
