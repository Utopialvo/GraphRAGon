# src/got/graph_state.py

"""
Состояние графа мыслей: хранение всех узлов и порядка их обхода.
"""
from typing import List, Dict, Optional
from .thought import Thought


class GraphReasoningState:
    def __init__(self):
        self.thoughts: Dict[str, Thought] = {}
        self.current_ids: List[str] = []

    def add_thought(self, thought: Thought) -> None:
        self.thoughts[thought.id] = thought

    def get_thought(self, thought_id: str) -> Optional[Thought]:
        return self.thoughts.get(thought_id)

    def get_thoughts(self, thought_ids: List[str]) -> List[Thought]:
        return [self.thoughts[tid] for tid in thought_ids if tid in self.thoughts]

    def get_current_thoughts(self) -> List[Thought]:
        return [self.thoughts[tid] for tid in self.current_ids]

    def set_current(self, ids: List[str]) -> None:
        self.current_ids = ids