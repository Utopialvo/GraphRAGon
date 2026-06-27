# src/llm_entity_extract.py

"""
Извлечение именованных сущностей из текста с помощью LLM.
Использует базовый класс BaseExtractor для общей логики парсинга JSON.
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


class BaseExtractor:
    """
    Общие методы для экстракторов: отправка промпта, парсинг JSON с fallback-промптом.
    """
    def __init__(self, llm_config: LLMConfig):
        self.client = LLMClient(llm_config)

    def _extract_with_fallback(self, prompt: str, fallback_prompt: str) -> List[dict]:
        """
        Отправляет основной промпт, если результат невалидный — пробует запасной.
        Возвращает список словарей.
        """
        response = self.client.chat(prompt, response_format={"type": "json_object"})
        data = safe_parse_json(response)
        if data is None or not isinstance(data, list):
            logging.warning("Основной промпт не дал списка, пробуем fallback.")
            response = self.client.chat(fallback_prompt, response_format={"type": "json_object"})
            data = safe_parse_json(response)
        return data if isinstance(data, list) else []


class LLMEntityExtractor(BaseExtractor):
    """Извлекает сущности из текста с помощью LLM."""

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.prompt_template = """
Извлеки все именованные сущности (люди, места, предметы, действия, временные отрезки) из текста.
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
        fallback = self.fallback_prompt.format(text=text)
        items = self._extract_with_fallback(prompt, fallback)
        entities = []
        for item in items:
            try:
                entities.append(Entity(**item))
            except (TypeError, ValidationError) as e:
                logging.warning(f"Пропускаем некорректную сущность: {item}, ошибка: {e}")
        return entities