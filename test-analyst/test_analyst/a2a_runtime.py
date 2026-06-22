from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import A2AHandoff, AgentCard
from .utils import jsonify


class A2AAgentRegistry:
    """Keep lightweight A2A-ready agent cards and handoff records."""

    def __init__(self, agent_cards: list[AgentCard]) -> None:
        self._cards = {card.agent_id: card for card in agent_cards}

    @property
    def cards(self) -> list[AgentCard]:
        return list(self._cards.values())

    def handoff(
        self,
        *,
        from_agent: str,
        to_agent: str,
        task_type: str,
        summary: str,
        payload_preview: dict[str, Any] | None = None,
    ) -> A2AHandoff:
        return A2AHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            summary=summary,
            created_at=datetime.utcnow().isoformat(),
            payload_preview=jsonify(payload_preview or {}),
        )
