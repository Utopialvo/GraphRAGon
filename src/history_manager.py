# src/history_manager.py

"""
Модуль для хранения истории запросов и ответов в RAM.
"""

from collections import deque
from typing import List, Tuple, Optional


class HistoryManager:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size)

    def add_entry(self, question: str, answer: str) -> None:
        self.history.append((question, answer))

    def get_history(self, last_n: Optional[int] = None) -> List[Tuple[str, str]]:
        if last_n is None:
            return list(self.history)
        return list(self.history)[-last_n:]

    def get_history_as_text(self, last_n: Optional[int] = None) -> str:
        entries = self.get_history(last_n)
        if not entries:
            return "История диалога пуста."
        text = "Предыдущие вопросы и ответы:\n"
        for i, (q, a) in enumerate(entries, 1):
            text += f"{i}. Вопрос: {q}\n   Ответ: {a}\n"
        return text

    def clear(self) -> None:
        self.history.clear()