# src/relation_extractor.py

"""
Модуль извлечения отношений между сущностями с помощью LLM.
Парсинг JSON вынесен в utils.safe_parse_json.
"""

import json
import logging
from typing import List
from pydantic import BaseModel, ValidationError
from llm_client import LLMClient, LLMConfig
from utils import safe_parse_json


class Relation(BaseModel):
    head: str
    relation: str
    tail: str


class RelationExtractor:
    """Извлекает отношения между сущностями на основе текста и списка сущностей."""

    def __init__(self, llm_config: LLMConfig):
        self.client = LLMClient(llm_config)
        self.prompt_template = """
Ты — система извлечения отношений между сущностями.
Даны сущности: {entities}
На основе текста найди все отношения между ними. Отношение должно быть кратким глаголом или фразой (например, "рубил", "растопила", "пошёл на").
Верни ТОЛЬКО JSON-список объектов с полями "head", "relation", "tail", где head и tail — это тексты сущностей.
Пример: [{{"head": "Иван", "relation": "рубил", "tail": "дрова"}}]
Если отношений нет, верни пустой список: []
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
        response = self.client.chat(prompt, response_format={"type": "json_object"})
        data = safe_parse_json(response)

        if not data or not isinstance(data, list):
            logging.warning("Не удалось распарсить отношения из основного промпта, пробуем fallback.")
            fallback_prompt = self.fallback_prompt_template.format(entities=entities_str, text=text)
            response = self.client.chat(fallback_prompt, response_format={"type": "json_object"})
            data = safe_parse_json(response)

        relations = []
        if isinstance(data, list):
            for item in data:
                try:
                    rel = Relation(**item)
                    relations.append(rel)
                except (TypeError, ValidationError) as e:
                    logging.warning(f"Пропускаем некорректное отношение: {item}, ошибка: {e}")
        return relations