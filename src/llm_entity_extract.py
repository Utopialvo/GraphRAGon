# src/llm_entity_extract.py

"""
Модуль извлечения сущностей из текста с помощью LLM.
Парсинг JSON вынесен в utils.safe_parse_json.
"""

import json
import logging
from typing import List
from pydantic import BaseModel, ValidationError
from llm_client import LLMClient, LLMConfig
from utils import safe_parse_json


class Entity(BaseModel):
    text: str
    type: str


class LLMEntityExtractor:
    """Извлекает именованные сущности из текста через LLM."""

    def __init__(self, llm_config: LLMConfig):
        self.client = LLMClient(llm_config)
        self.prompt_template = """
Ты — система извлечения сущностей. Извлеки все именованные сущности (люди, места, предметы, действия, временные отрезки) из текста.
Верни ТОЛЬКО JSON-список объектов с полями "text" (сама сущность) и "type" (тип, например "PERSON", "LOCATION", "OBJECT", "ACTION", "TIME").
Пример: [{{"text": "Иван", "type": "PERSON"}}, {{"text": "дрова", "type": "OBJECT"}}]
Если сущностей нет, верни [].
Текст: {text}
"""
        self.fallback_prompt = """
Извлеки сущности из текста. Верни JSON-массив объектов с ключами "text" и "type".
Пример: [{{"text": "Иван", "type": "PERSON"}}]
Если сущностей нет, верни [].
Текст: {text}
Ответ (только JSON):
"""

    def extract(self, text: str) -> List[Entity]:
        prompt = self.prompt_template.format(text=text)
        response = self.client.chat(prompt, response_format={"type": "json_object"})
        data = safe_parse_json(response)
        if not data or not isinstance(data, list):
            logging.warning("Не удалось распарсить сущности из основного промпта, пробуем fallback.")
            fallback = self.fallback_prompt.format(text=text)
            response = self.client.chat(fallback, response_format={"type": "json_object"})
            data = safe_parse_json(response)

        entities = []
        if isinstance(data, list):
            for item in data:
                try:
                    ent = Entity(**item)
                    entities.append(ent)
                except (TypeError, ValidationError) as e:
                    logging.warning(f"Пропускаем некорректную сущность: {item}, ошибка: {e}")
        return entities