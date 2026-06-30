# GraphRAGon/src/history_manager.py

"""
Хранение истории запросов и ответов в оперативной памяти.
Реализовано на основе deque с ограничением размера.
Элементы истории — словари с ключами question и answer.
"""
from collections import deque
from typing import List, Optional, Dict


class HistoryManager:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size)

    def add_entry(self, question: str, answer: str) -> None:
        entry = {"question": question, "answer": answer}
        self.history.append(entry)

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        if last_n is None:
            return list(self.history)
        return list(self.history)[-last_n:]

    def clear(self) -> None:
        self.history.clear()