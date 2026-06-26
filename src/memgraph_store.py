# src/memgraph_store.py

"""
Модуль для работы с Memgraph как трёхуровневой памятью (Schema, Fact, Passage).
Исправления:
- add_relation: корректное обновление списка based_on (добавление элемента)
- get_relations_with_context: корректная обработка списка based_on через UNWIND
- get_all_relations_with_passage: возвращает based_on как список
- Потокобезопасность через блокировку всех публичных методов.
- add_entity теперь использует MERGE для связи MENTIONED_IN (без дублей).
- clear() удаляет векторный индекс и пересоздаёт его.
"""

import logging
import uuid
import threading
from typing import List, Dict, Any, Optional

from gqlalchemy import Memgraph


class MemgraphStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        embedding_client=None,
        index_capacity: int = 1000,
        auto_create_entities: bool = True,
        update_type_if_unknown: bool = True,
    ):
        self.db = Memgraph(host=host, port=port)
        self.embedding_client = embedding_client
        self.index_capacity = index_capacity
        self.auto_create_entities = auto_create_entities
        self.update_type_if_unknown = update_type_if_unknown
        self._lock = threading.Lock()

        try:
            cursor = self.db.execute_and_fetch("RETURN 1")
            if cursor is None:
                raise ConnectionError("Memgraph вернул None при проверочном запросе")
            logging.info(f"Подключение к Memgraph {host}:{port} установлено")
        except Exception as e:
            logging.error(f"Не удалось подключиться к Memgraph: {e}")
            raise

        self._init_index()

    # ---------- Транзакции ----------
    def begin(self) -> None:
        with self._lock:
            self.db.execute("BEGIN")

    def commit(self) -> None:
        with self._lock:
            self.db.execute("COMMIT")

    def rollback(self) -> None:
        with self._lock:
            self.db.execute("ROLLBACK")

    def snapshot(self) -> None:
        with self._lock:
            self.db.execute("CREATE SNAPSHOT")
            logging.info("Снэпшот Memgraph создан.")

    # ---------- Индексы ----------
    def _init_index(self) -> None:
        if not self.embedding_client:
            return
        dim = self.embedding_client.dim
        index_name = "entity_embeddings"

        try:
            check_query = "SHOW VECTOR INDEXES"
            cursor = self.db.execute_and_fetch(check_query)
            if cursor:
                for row in cursor:
                    if row.get('name') == index_name:
                        existing_dim = row.get('dimension')
                        if existing_dim is not None and existing_dim != dim:
                            logging.warning(
                                f"Индекс {index_name} имеет размерность {existing_dim}, "
                                f"требуется {dim}. Пересоздаём..."
                            )
                            self.db.execute(f"DROP VECTOR INDEX {index_name}")
                        else:
                            logging.info(f"Векторный индекс {index_name} уже существует с размерностью {dim}")
                            return
        except Exception as e:
            logging.debug(f"Не удалось проверить индекс: {e}")

        try:
            self.db.execute(
                f"""
                CREATE VECTOR INDEX {index_name} ON :Entity(embedding)
                WITH CONFIG {{"dimension": {dim}, "capacity": {self.index_capacity}}}
                """
            )
            logging.info(f"Векторный индекс {index_name} успешно создан с размерностью {dim}.")
        except Exception as e:
            if "already exists" in str(e).lower():
                logging.info("Векторный индекс уже существует.")
                return
            logging.error(f"Ошибка при создании векторного индекса: {e}")
            raise

    # ---------- Вспомогательные методы ----------
    def _ensure_schema(self, schema_name: str, schema_type: str = "entity_type") -> None:
        query = """
        MERGE (s:Schema {name: $name})
        SET s.type = $schema_type
        """
        with self._lock:
            self.db.execute(query, {"name": schema_name, "schema_type": schema_type})

    def _entity_exists(self, name: str) -> bool:
        query = "MATCH (e:Entity {name: $name}) RETURN e LIMIT 1"
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": name})
        if cursor is None:
            return False
        rows = list(cursor)
        return len(rows) > 0

    def _get_existing_entity(self, name: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e.name AS name, e.type AS type, e.embedding AS embedding
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": name})
        if cursor is None:
            return None
        rows = list(cursor)
        return rows[0] if rows else None

    def get_passage_text_by_id(self, passage_id: str) -> Optional[str]:
        """Возвращает текст пассажа по его ID. Ожидает одиночный ID."""
        query = "MATCH (p:Passage {id: $pid}) RETURN p.text AS text"
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"pid": passage_id})
        if cursor is None:
            return None
        rows = list(cursor)
        return rows[0]["text"] if rows else None

    # ---------- Passage ----------
    def add_passage(self, text: str, passage_id: Optional[str] = None) -> str:
        if passage_id is None:
            passage_id = str(uuid.uuid4())
        query = """
        MERGE (p:Passage {id: $id})
        SET p.text = $text
        """
        with self._lock:
            self.db.execute(query, {"id": passage_id, "text": text})
        return passage_id

    # ---------- Schema ----------
    def add_schema(self, schema_name: str, schema_type: str = "entity_type") -> None:
        self._ensure_schema(schema_name, schema_type)

    # ---------- Entity ----------
    def add_entity(self, name: str, entity_type: str, passage_id: str) -> None:
        self._ensure_schema(entity_type, "entity_type")
        existing = self._get_existing_entity(name)
        with self._lock:
            if existing is None:
                emb = self.embedding_client.embed(name) if self.embedding_client else None
                query = """
                CREATE (e:Entity {name: $name, type: $type, embedding: $embedding})
                WITH e
                MATCH (p:Passage {id: $passage_id})
                CREATE (e)-[:MENTIONED_IN]->(p)
                WITH e
                MATCH (s:Schema {name: $type})
                CREATE (e)-[:INSTANCE_OF]->(s)
                """
                self.db.execute(
                    query,
                    {
                        "name": name,
                        "type": entity_type,
                        "embedding": emb,
                        "passage_id": passage_id,
                    },
                )
            else:
                if self.update_type_if_unknown and existing["type"] == "UNKNOWN" and entity_type != "UNKNOWN":
                    self.db.execute(
                        "MATCH (e:Entity {name: $name}) SET e.type = $new_type",
                        {"name": name, "new_type": entity_type}
                    )
                # MERGE вместо CREATE, чтобы не плодить дублирующие рёбра
                link_query = """
                MATCH (e:Entity {name: $name}), (p:Passage {id: $passage_id})
                MERGE (e)-[:MENTIONED_IN]->(p)
                """
                self.db.execute(link_query, {"name": name, "passage_id": passage_id})

    # ---------- Relation ----------
    def add_relation(self, head: str, relation: str, tail: str, passage_id: str) -> None:
        """Добавляет отношение. passage_id – одиночный идентификатор пассажа."""
        self._ensure_schema(relation, "relation_type")
        if self.auto_create_entities:
            if not self._entity_exists(head):
                self.add_entity(head, "UNKNOWN", passage_id)
            if not self._entity_exists(tail):
                self.add_entity(tail, "UNKNOWN", passage_id)
        else:
            if not self._entity_exists(head):
                raise ValueError(f"Головная сущность '{head}' не найдена")
            if not self._entity_exists(tail):
                raise ValueError(f"Хвостовая сущность '{tail}' не найдена")

        query = """
        MATCH (h:Entity {name: $head}), (t:Entity {name: $tail})
        MERGE (h)-[r:RELATION {type: $relation}]->(t)
        ON CREATE SET r.based_on = [$passage_id]
        ON MATCH SET r.based_on = CASE
            WHEN $passage_id NOT IN r.based_on THEN r.based_on + [$passage_id]
            ELSE r.based_on
        END
        """
        with self._lock:
            self.db.execute(
                query,
                {
                    "head": head,
                    "tail": tail,
                    "relation": relation,
                    "passage_id": passage_id,
                },
            )

    def delete_relation(self, head: str, relation: str, tail: str) -> None:
        query = """
        MATCH (h:Entity {name: $head})-[r:RELATION {type: $rel}]->(t:Entity {name: $tail})
        DELETE r
        """
        with self._lock:
            self.db.execute(query, {"head": head, "tail": tail, "rel": relation})

    # ---------- Большие тексты ----------
    def add_large_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        if chunk_size <= overlap:
            raise ValueError("overlap должен быть меньше chunk_size")
        words = text.split()
        total_words = len(words)
        step = chunk_size - overlap
        passage_ids = []
        for start in range(0, total_words, step):
            end = min(start + chunk_size, total_words)
            chunk = " ".join(words[start:end])
            if not chunk.strip():
                continue
            passage_id = self.add_passage(chunk)
            passage_ids.append(passage_id)
        return passage_ids

    # ---------- Поиск и извлечение данных ----------
    def vector_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.embedding_client:
            raise ValueError("Не задан embedding_client для векторного поиска")
        q_emb = self.embedding_client.embed(query_text)
        query = """
        CALL vector_search.search($index_name, $top_k, $embedding)
        YIELD node, distance
        RETURN node.name AS name, node.type AS type, distance AS score
        """
        with self._lock:
            try:
                cursor = self.db.execute_and_fetch(
                    query,
                    {
                        "index_name": "entity_embeddings",
                        "embedding": q_emb,
                        "top_k": top_k,
                    },
                )
                return list(cursor) if cursor is not None else []
            except Exception as e:
                if "Index not found" in str(e) or "does not exist" in str(e):
                    raise RuntimeError(
                        "Векторный индекс 'entity_embeddings' не найден. "
                        "Убедитесь, что индекс создан (вызов _init_index)."
                    ) from e
                raise

    def get_passage_for_entity(self, entity_name: str) -> Optional[str]:
        query = """
        MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(p:Passage)
        RETURN p.text AS text LIMIT 1
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": entity_name})
        if cursor is None:
            return None
        rows = list(cursor)
        return rows[0]["text"] if rows else None

    def get_passages_for_entity(self, entity_name: str, limit: int = 3) -> List[str]:
        query = """
        MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(p:Passage)
        RETURN p.text AS text LIMIT $limit
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": entity_name, "limit": limit})
        if cursor is None:
            return []
        return [row["text"] for row in cursor]

    def get_relations_with_context(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Возвращает все отношения указанной сущности вместе с текстом каждого связанного пассажа.
        Если based_on содержит несколько ID, возвращается по строке на каждый пассаж.
        """
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATION]-(other:Entity)
        UNWIND r.based_on AS bid
        MATCH (p:Passage {id: bid})
        RETURN e.name AS source, r.type AS relation, other.name AS target, p.text AS passage_text
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": entity_name})
        return list(cursor) if cursor is not None else []

    def get_all_relations(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (h:Entity)-[r:RELATION]->(t:Entity)
        RETURN DISTINCT h.name AS head, r.type AS relation, t.name AS tail
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query)
        return list(cursor) if cursor is not None else []

    def get_all_relations_with_passage(self) -> List[Dict[str, Any]]:
        """
        Возвращает все триплеты с полем based_on (список идентификаторов пассажей).
        """
        query = """
        MATCH (h:Entity)-[r:RELATION]->(t:Entity)
        RETURN h.name AS head, r.type AS relation, t.name AS tail, r.based_on AS based_on
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query)
        return list(cursor) if cursor is not None else []

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        return self._get_existing_entity(name)

    def clear(self) -> None:
        with self._lock:
            self.db.execute("MATCH (n) DETACH DELETE n")
            if self.embedding_client:
                try:
                    self.db.execute("DROP VECTOR INDEX entity_embeddings")
                except Exception as e:
                    logging.debug(f"Не удалось удалить векторный индекс: {e}")
                # сразу создаём индекс заново, чтобы граф оставался в рабочем состоянии
                self._init_index()