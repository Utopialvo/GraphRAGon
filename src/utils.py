# src/utils.py

"""
Утилиты для безопасного разбора JSON из ответов LLM.
LLM иногда возвращает текст с лишними пояснениями, поэтому приходится выковыривать JSON.
"""

import json
import re
import logging
from typing import Optional, List, Any


def safe_parse_json(response: str) -> Optional[Any]:
    """
    Извлекает JSON-объект или массив из строки ответа модели.
    Поддерживает блоки ```json ... ``` и поиск первой/последней скобки.
    Возвращает распарсенный объект или None.
    """
    if not response or not response.strip():
        return None

    # Поиск блока ```json ... ```
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1)
    else:
        # Ищем первую { или [
        start = response.find('[')
        if start == -1:
            start = response.find('{')
        if start != -1:
            end = response.rfind(']')
            if end == -1:
                end = response.rfind('}')
            if end != -1 and end > start:
                json_str = response[start:end+1]
            else:
                json_str = response[start:]
        else:
            json_str = response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.debug(f"Не удалось разобрать JSON из строки: {response[:200]}")
        return None