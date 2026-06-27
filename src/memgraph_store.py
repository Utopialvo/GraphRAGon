# src/memgraph_store.py

"""
Работа с графовой базой Memgraph.
Хранит сущности, отношения, пассажи и векторный индекс.
Все публичные методы потокобезопасны (используется RLock).
Добавлен временный кэш эмбеддингов для ускорения массовой загрузки.
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
        self._lock = threading.RLock()
        self._embedding_cache: Dict[str, list] = None

        try:
            cursor = self.db.execute_and_fetch("RETURN 1")
            if cursor is None:
                raise ConnectionError("Memgraph вернул None при проверочном запросе")
            logging.info(f"Подключение к Memgraph {host}:{port} установлено")
        except Exception as e:
            logging.error(f"Не удалось подключиться к Memgraph: {e}")
            raise

        self._init_index()

    def enable_batch_cache(self) -> None:
        """Включает кэширование эмбеддингов на время массовой загрузки."""
        self._embedding_cache = {}

    def disable_batch_cache(self) -> None:
        """Очищает кэш эмбеддингов и отключает его."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
            self._embedding_cache = None

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
            cursor = self.db.execute_and_fetch("SHOW VECTOR INDEXES")
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

    # ---------- Entity ----------
    def add_entity(self, name: str, entity_type: str, passage_id: str) -> None:
        self._ensure_schema(entity_type, "entity_type")
        existing = self._get_existing_entity(name)

        with self._lock:
            if existing is None:
                if self._embedding_cache is not None and name in self._embedding_cache:
                    emb = self._embedding_cache[name]
                else:
                    emb = self.embedding_client.embed(name) if self.embedding_client else None
                    if self._embedding_cache is not None:
                        self._embedding_cache[name] = emb
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
                link_query = """
                MATCH (e:Entity {name: $name}), (p:Passage {id: $passage_id})
                MERGE (e)-[:MENTIONED_IN]->(p)
                """
                self.db.execute(link_query, {"name": name, "passage_id": passage_id})

    # ---------- Relation ----------
    def add_relation(self, head: str, relation: str, tail: str, passage_id: str) -> None:
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

        create_query = """
        MATCH (h:Entity {name: $head}), (t:Entity {name: $tail})
        MERGE (h)-[r:RELATION {type: $relation}]->(t)
        ON CREATE SET r.based_on = [$passage_id]
        """
        with self._lock:
            self.db.execute(create_query, {
                "head": head,
                "tail": tail,
                "relation": relation,
                "passage_id": passage_id,
            })

        # Добавляем passage_id в список, если его там ещё нет
        add_passage_query = """
        MATCH (h:Entity {name: $head})-[r:RELATION {type: $relation}]->(t:Entity {name: $tail})
        WHERE NOT $passage_id IN r.based_on
        SET r.based_on = r.based_on + [$passage_id]
        """
        with self._lock:
            self.db.execute(add_passage_query, {
                "head": head,
                "tail": tail,
                "relation": relation,
                "passage_id": passage_id,
            })

    def delete_relation(self, head: str, relation: str, tail: str) -> None:
        query = """
        MATCH (h:Entity {name: $head})-[r:RELATION {type: $rel}]->(t:Entity {name: $tail})
        DELETE r
        """
        with self._lock:
            self.db.execute(query, {"head": head, "tail": tail, "rel": relation})

    # ---------- Random Walk ----------
    def random_walk(self, start_entity: str, depth: int = 2, breadth: int = 3) -> List[Dict[str, Any]]:
        query = f"""
        MATCH path = (start:Entity {{name: $name}})-[:RELATION*..{depth}]-(neighbor)
        WHERE start <> neighbor
        UNWIND relationships(path) AS r
        WITH start, r, neighbor, rand() AS rand
        ORDER BY rand
        LIMIT $limit
        RETURN DISTINCT start.name AS head, r.type AS relation, neighbor.name AS tail
        """
        with self._lock:
            cursor = self.db.execute_and_fetch(query, {"name": start_entity, "limit": breadth})
        return list(cursor) if cursor is not None else []

    # ---------- Поиск и извлечение данных ----------
    def vector_search(self, query_text: str = None, top_k: int = 5, embedding: list = None) -> List[Dict[str, Any]]:
        if embedding is None:
            if not self.embedding_client:
                raise ValueError("Не задан embedding_client для векторного поиска")
            if not query_text:
                raise ValueError("Нужно указать query_text или embedding")
            embedding = self.embedding_client.embed(query_text)

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
                        "embedding": embedding,
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
                finally:
                    self._init_index()