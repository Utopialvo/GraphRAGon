# src/relation_extractor.py

"""
Извлечение отношений между сущностями с помощью LLM.
Также использует базовый класс BaseExtractor.
"""
import json
import logging
from typing import List
from pydantic import BaseModel, ValidationError
from llm_client import LLMClient, LLMConfig
from utils import safe_parse_json
from llm_entity_extract import BaseExtractor


class Relation(BaseModel):
    head: str
    relation: str
    tail: str


class RelationExtractor(BaseExtractor):
    """Извлекает отношения между сущностями, опираясь на текст и список найденных сущностей."""

    def __init__(self, llm_config: LLMConfig, max_retries: int = 2, retry_delay: float = 1.0):
        super().__init__(llm_config, max_retries, retry_delay)
        self.prompt_template = """
Даны сущности: {entities}
На основе текста найди все отношения между ними. Отношение должно быть кратким глаголом или фразой (например, "рубил", "растопила", "пошёл на").
Верни ТОЛЬКО JSON-список объектов с полями "head", "relation", "tail", где head и tail — это тексты сущностей.
Пример: [{{"head": "Иван", "relation": "рубил", "tail": "дрова"}}]
Если отношений нет, верни [].
Текст: {text}
"""
        self.fallback_prompt_template = """
Извлеки отношения между сущностями из текста. 
Сущности: {entities}
Текст: {text}
Верни JSON-массив объектов с ключами "head", "relation", "tail". 
Пример: [{{"head": "Иван", "relation": "рубил", "tail": "дрова"}}]
Если отношений нет, верни [].
Ответ (только JSON):
"""

    def extract(self, text: str, entities: List) -> List[Relation]:
        entity_texts = [e.text if hasattr(e, 'text') else e.get('text', '') for e in entities]
        entities_str = ", ".join(entity_texts)

        prompt = self.prompt_template.format(entities=entities_str, text=text)
        fallback = self.fallback_prompt_template.format(entities=entities_str, text=text)
        items = self._extract_with_fallback(prompt, fallback)

        relations = []
        for item in items:
            try:
                relations.append(Relation(**item))
            except (TypeError, ValidationError) as e:
                logging.warning(f"Пропускаем некорректное отношение: {item}, ошибка: {e}")
        return relations