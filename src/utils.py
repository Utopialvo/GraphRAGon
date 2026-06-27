# src/utils.py

"""
Безопасный разбор JSON, который возвращает LLM.
Модель иногда добавляет лишний текст до или после JSON‑структуры.
Функции здесь стараются вытащить чистый JSON из любого подобного ответа.
"""
import json
import re
import logging
from typing import Optional, Any


def safe_parse_json(response: str) -> Optional[Any]:
    """
    Извлекает первый JSON-объект или массив из строки.
    Поддерживает блоки с тройными обратными кавычками ```json ... ```,
    а также ищет первую открывающую скобку { или [.
    Если ничего не получается, возвращает None.
    """
    if not response or not response.strip():
        return None

    # Попытка найти блок в формате Markdown
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1).strip()
    else:
        # Ищем первую фигурную или квадратную скобку
        start = response.find('[')
        if start == -1:
            start = response.find('{')
        if start != -1:
            end = response.rfind(']')
            if end == -1:
                end = response.rfind('}')
            if end != -1 and end > start:
                json_str = response[start:end + 1]
            else:
                json_str = response[start:]
        else:
            json_str = response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.debug(f"Не удалось разобрать JSON из ответа: {response[:200]}")
        return None