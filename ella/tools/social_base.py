"""Shared schema for social media tool results.

All social tools (social_rednote, social_x, social_facebook, …) return a
list of SocialPost objects serialised as JSON, so ResearchSkill can handle
them uniformly regardless of which platform produced the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SocialPost:
    """A single social media post with its engagement metrics and comments."""
    platform: str           # "rednote" | "x" | "facebook"
    post_id: str
    url: str
    title: str              # first line / headline of the post
    body: str               # full post text
    author: str
    published_at: str       # ISO 8601 timestamp (best-effort; empty if unavailable)
    likes: int = 0
    collects: int = 0       # saves/bookmarks — Rednote 收藏; 0 for platforms without this
    comments_count: int = 0
    shares: int = 0
    comments: list[str] = field(default_factory=list)  # full text of ALL comments

    @property
    def engagement_score(self) -> int:
        """Credibility signal: sum of all engagement signals.

        High engagement (especially collects on Rednote) indicates the
        community found the content credible and worth saving.
        """
        return self.likes + self.collects + self.comments_count + self.shares
