# src/got/parser.py

"""
Парсер ответов LLM для извлечения числовых оценок.
"""
import re
import logging

class Parser:
    def parse_score(self, response: str) -> float:
        # Ищем первое число с плавающей точкой или целое
        match = re.search(r'\b(\d+(\.\d+)?)\b', response)
        if match:
            score = float(match.group(1))
            return max(0, min(10, score))
        logging.warning(f"Не удалось извлечь оценку из ответа: {response[:100]}")
        return 0.0