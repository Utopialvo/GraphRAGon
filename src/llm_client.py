# src/llm_client.py

"""
Клиент для общения с локальной LLM через OpenAI-совместимый API.
Кэширует ответы в SQLite, умеет чистить старые записи.
"""

import json
import hashlib
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from openai import OpenAI


class LLMConfig:
    def __init__(self, model_name: str, base_url: str, api_key: str = "dummy", temperature: float = 0.0):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature


class LLMClient:
    def __init__(self, config: LLMConfig, cache_db: str = "llm_cache.db",
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.config = config
        self.cache_db = cache_db
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._init_cache()

    def _init_cache(self) -> None:
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT,
                timestamp DATETIME
            )
        """)
        # совместимость со старыми таблицами без timestamp
        cursor = conn.execute("PRAGMA table_info(cache)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'timestamp' not in columns:
            logging.info("Добавляем колонку timestamp в таблицу cache")
            conn.execute("ALTER TABLE cache ADD COLUMN timestamp DATETIME")
        conn.commit()
        conn.close()

    def _cache_key(self, prompt: str, model: str) -> str:
        return hashlib.md5(f"{prompt}_{model}".encode()).hexdigest()

    def cleanup_cache(self, max_age_hours: int = 1) -> int:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        conn = sqlite3.connect(self.cache_db)
        cur = conn.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        logging.info(f"Очистка кэша: удалено {deleted} записей старше {max_age_hours} ч.")
        return deleted

    def chat(self, prompt: str, response_format: dict = None) -> str:
        key = self._cache_key(prompt, self.config.model_name)
        conn = sqlite3.connect(self.cache_db)
        cur = conn.execute("SELECT response FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            conn.close()
            return row[0]
        conn.close()

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                if response_format and response_format.get("type") == "json_object":
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature
                    )
                content = response.choices[0].message.content

                conn = sqlite3.connect(self.cache_db)
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (key, content)
                )
                conn.commit()
                conn.close()
                return content

            except Exception as e:
                last_exception = e
                logging.warning(f"Попытка {attempt+1}/{self.max_retries} не удалась: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logging.error(f"Все попытки исчерпаны: {e}")
                    raise last_exception