# src/embedding_client.py

"""
Клиент для вычисления текстовых эмбеддингов с использованием sentence-transformers.
Результаты кэшируются в SQLite.
"""
import hashlib
import sqlite3
import json
from typing import Optional


class EmbeddingClient:
    """Генерирует векторные представления текстов с локальным кэшированием."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_db: str = "llm_cache.db",
        batch_size: int = 32,
        precision: str = "float32",
    ):
        self.model_name = model_name
        self.cache_db = cache_db
        self.batch_size = batch_size
        self.precision = precision
        self._model = None
        self._dim = None
        self._init_cache()

    def _init_cache(self) -> None:
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                embedding TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model
        return self._dim

    def embed(
        self,
        text: str,
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list:
        """Возвращает нормализованный эмбеддинг для одного текста."""
        key = hashlib.md5(f"{text}_{prompt_name}_{prompt}".encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db)
        cur = conn.execute("SELECT embedding FROM embedding_cache WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            conn.close()
            return json.loads(row[0])

        emb = self.model.encode(
            text,
            prompt_name=prompt_name,
            prompt=prompt,
            normalize_embeddings=True,
            precision=self.precision,
        ).tolist()

        emb_json = json.dumps(emb)
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (key, embedding) VALUES (?, ?)",
            (key, emb_json)
        )
        conn.commit()
        conn.close()
        return emb