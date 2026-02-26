"""Distillation (CODE) skill — Reflects on recent conversations and updates User profile.

This background skill runs periodically. It pulls the last N exchanges from `ella_conversations`,
asks the LLM to identify new facts, preferences, or emotional states about the user,
and updates `~/Ella/User.md` (the Tiago Forte "Areas" mapping) accordingly.
"""
import logging
from typing import Any

from ella.skills.base import BaseSkill
from ella.memory.knowledge import get_knowledge_store
from ella.memory.identity import ELLA_DIR
from ella.llm.gemini_client import get_gemini_client

logger = logging.getLogger(__name__)


class DistillKnowledgeSkill(BaseSkill):
    """Reflects on recent conversations to build Gareth's long-term profile."""

    name = "distill_user_knowledge"
    description = "Automatically reflects on recent chat history to learn facts about the user and update their profile."
    version = "1.0"
    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, chat_id: int, args: dict[str, Any]) -> str:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            return "Qdrant client not available for distillation."

        store = get_knowledge_store()
        
        # 1. Fetch the last 20 conversation turns for this chat
        # Since qdrant scroll returns points ordered by internal ID (not time),
        # and we want recent conversation, we should ideally use query_points
        # sorted by timestamp, or fetch a larger batch and sort in memory.
        try:
            # We fetch a larger batch since scroll order is undefined
            response = await store._client.scroll(
                collection_name="ella_conversations",
                scroll_filter=Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]),
                limit=100,
                with_payload=True,
            )
            points = response[0]
            if not points:
                return "No recent conversation history to distill."
                
            # Sort by timestamp to get recency, then take the last 30
            points.sort(key=lambda p: (p.payload or {}).get("timestamp", ""))
            points = points[-30:]
            
            history_text = "\n".join([(p.payload or {}).get("text", "") for p in points])
        except Exception as e:
            logger.error("Failed to fetch history for distillation: %s", e)
            return f"Failed to fetch history: {e}"

        if not history_text.strip():
            return "History was empty."

        # 2. Prepare current profile
        user_file = ELLA_DIR / "User.md"
        current_content = user_file.read_text(encoding="utf-8") if user_file.exists() else ""

        # 3. Ask Gemini to rewrite/merge facts
        prompt = (
            "You are a background reflection process managing Gareth's 'Second Brain' profile.\n"
            "Review the recent chat history and extract any NEW, concrete facts about Gareth "
            "(preferences, goals, emotional state, current projects, etc.).\n"
            "If there is NOTHING new, print 'NONE' and stop immediately.\n\n"
            "If there ARE new facts, rewrite his entire 'Learned Traits' section below. "
            "Merge the new facts seamlessly into the existing points. Remove outdated/conflicting info. "
            "Keep it strictly formatted as a concise markdown bulleted list.\n\n"
            f"=== Current Profile ===\n{current_content}\n\n"
            f"=== Recent Chat History ===\n{history_text}\n\n"
            "Output 'NONE' or the complete, updated profile:"
        )
        
        client = get_gemini_client()
        result = await client.generate_text(prompt)
        
        if not result or result.strip() == "NONE":
            return "No new facts distilled."

        # 4. Save the optimized, rolling rewrite
        try:
            user_file.write_text(result.strip() + "\n", encoding="utf-8")
            logger.info("Distilled and rewrote User.md.")
            return "Successfully optimized and updated User.md."
            
        except Exception as e:
            logger.error("Failed to write to User.md during distillation: %s", e)
            return f"Distillation failed on write: {e}"
