# src/llm_entity_extract.py

"""
Извлечение именованных сущностей из текста с помощью LLM.
Использует базовый класс BaseExtractor для общей логики парсинга JSON.
"""
import json
import logging
import time
from typing import List
from pydantic import BaseModel, ValidationError
from llm_client import LLMClient, LLMConfig
from utils import safe_parse_json


class Entity(BaseModel):
    text: str
    type: str


class BaseExtractor:
    """
    Общие методы для экстракторов: отправка промпта, парсинг JSON с повторными попытками.
    """
    def __init__(self, llm_config: LLMConfig, max_retries: int = 2, retry_delay: float = 1.0):
        self.client = LLMClient(llm_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _extract_with_fallback(self, prompt: str, fallback_prompt: str) -> List[dict]:
        """
        Пытается получить список словарей, используя основной промпт.
        При неудаче делает до max_retries повторных попыток с основным промптом,
        а затем одну попытку с fallback-промптом.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(prompt, response_format={"type": "json_object"})
                data = safe_parse_json(response)
                if data is not None and isinstance(data, list):
                    return data
                logging.warning(f"Попытка {attempt+1}/{self.max_retries}: не удалось извлечь список JSON. "
                                f"Ответ: {response[:200]}...")
            except Exception as e:
                logging.warning(f"Попытка {attempt+1}/{self.max_retries} не удалась: {e}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))

        logging.info("Основные попытки не дали результата, пробуем fallback-промпт.")
        try:
            response = self.client.chat(fallback_prompt, response_format={"type": "json_object"})
            data = safe_parse_json(response)
            if isinstance(data, list):
                return data
        except Exception as e:
            logging.error(f"Fallback-промпт также не удался: {e}")

        return []


class LLMEntityExtractor(BaseExtractor):
    """Извлекает сущности из текста с помощью LLM."""

    def __init__(self, llm_config: LLMConfig, max_retries: int = 2, retry_delay: float = 1.0):
        super().__init__(llm_config, max_retries, retry_delay)
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