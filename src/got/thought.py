# src/got/thought.py

"""
Представление одной "мысли" в графе размышлений.
"""
import uuid
from typing import List, Optional, Any


class Thought:
    def __init__(self, content: str, parent_ids: List[str] = None, metadata: dict = None, score: float = 0.0):
        self.id = str(uuid.uuid4())
        self.content = content
        self.parent_ids = parent_ids or []
        self.metadata = metadata or {}
        self.score = score

    def __repr__(self):
        return f"Thought(id={self.id[:8]}..., score={self.score})"