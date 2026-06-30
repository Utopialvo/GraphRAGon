# GraphRAGon/src/memgraph_store.py
"""
Работа с графовой базой Memgraph через официальный Neo4j-драйвер.
Потокобезопасность обеспечивается блокировкой операций с драйвером.
Поддержка транзакций и временного кэша эмбеддингов.
"""
import logging
import threading
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase

class MemgraphStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        embedding_model=None,
        index_capacity: int = 1000,
        auto_create_entities: bool = True,
        update_type_if_unknown: bool = True,
        database: str = "memgraph",
    ):
        self.embedding_model = embedding_model
        self.index_capacity = index_capacity
        self.auto_create_entities = auto_create_entities
        self.update_type_if_unknown = update_type_if_unknown
        self._lock = threading.RLock()
        self._batch_cache_enabled = False
        self._embedding_cache: Dict[str, list] = {}

        url = f"bolt://{host}:{port}"
        try:
            self.driver = GraphDatabase.driver(url, auth=None)
            # Проверка соединения
            with self.driver.session(database=database) as session:
                session.run("RETURN 1").single()
            logging.info(f"Подключение к Memgraph {host}:{port} установлено")
        except Exception as e:
            logging.error(f"Не удалось подключиться к Memgraph: {e}")
            raise

        self._init_indexes()

    def enable_batch_cache(self) -> None:
        self._batch_cache_enabled = True
        self._embedding_cache.clear()

    def disable_batch_cache(self) -> None:
        self._batch_cache_enabled = False
        self._embedding_cache.clear()

    @contextmanager
    def transaction(self):
        """
        Контекстный менеджер для выполнения набора операций в одной транзакции.
        При выходе без исключений транзакция коммитится, иначе откатывается.
        """
        with self._lock:
            with self.driver.session() as session:
                tx = session.begin_transaction()
                try:
                    yield tx
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise

    # ---------- Индексы ----------
    def _init_indexes(self) -> None:
        if not self.embedding_model:
            return
        # Получаем размерность эмбеддинга через тестовый вызов
        test_embedding = self.embedding_model.embed_query("test")
        dim = len(test_embedding)
        self._create_vector_index("entity_embeddings", "Entity", "embedding", dim)
        self._create_vector_index("passage_embeddings", "Passage", "embedding", dim)

    def _create_vector_index(self, index_name: str, label: str, property: str, dim: int) -> None:
        with self._lock:
            with self.driver.session() as session:
                try:
                    session.run(f"DROP VECTOR INDEX {index_name}")
                except Exception as e:
                    logging.debug(f"Индекс {index_name} не существовал: {e}")
                query = f"""
                CREATE VECTOR INDEX {index_name} ON :{label}({property})
                WITH CONFIG {{"dimension": {dim}, "capacity": {self.index_capacity}}}
                """
                session.run(query)
                logging.info(f"Векторный индекс {index_name} создан с размерностью {dim}.")

    # ---------- Вспомогательные ----------
    def _get_embedding(self, text: str) -> list:
        if not self.embedding_model:
            return None
        if self._batch_cache_enabled:
            if text in self._embedding_cache:
                return self._embedding_cache[text]
            emb = self.embedding_model.embed_query(text)
            self._embedding_cache[text] = emb
            return emb
        return self.embedding_model.embed_query(text)

    def _execute(self, query: str, params: dict = None, tx=None):
        """Выполняет запрос либо в переданной транзакции tx, либо в новой автокоммит-сессии."""
        if tx:
            return tx.run(query, params or {})
        else:
            with self._lock:
                with self.driver.session() as session:
                    return list(session.run(query, params or {}))

    def _execute_read(self, query: str, params: dict = None):
        """Выполняет read-запрос с блокировкой."""
        with self._lock:
            with self.driver.session() as session:
                return list(session.run(query, params or {}))

    def _ensure_schema(self, schema_name: str, schema_type: str = "entity_type", tx=None) -> None:
        query = """
        MERGE (s:Schema {name: $name})
        SET s.type = $schema_type
        """
        self._execute(query, {"name": schema_name, "schema_type": schema_type}, tx)

    def _entity_exists(self, name: str, tx=None) -> bool:
        query = "MATCH (e:Entity {name: $name}) RETURN e LIMIT 1"
        res = self._execute(query, {"name": name}, tx)
        if tx:
            return res.single() is not None
        return len(res) > 0

    def _get_existing_entity(self, name: str, tx=None) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e.name AS name, e.type AS type, e.embedding AS embedding
        """
        res = self._execute(query, {"name": name}, tx)
        if tx:
            record = res.single()
            return dict(record) if record else None
        return res[0] if res else None

    # ---------- Passage ----------
    def add_passage(self, text: str, passage_id: Optional[str] = None, tx=None) -> str:
        if passage_id is None:
            import uuid
            passage_id = str(uuid.uuid4())
        emb = self._get_embedding(text) if self.embedding_model else None
        query = """
        MERGE (p:Passage {id: $id})
        SET p.text = $text, p.embedding = $embedding
        """
        self._execute(query, {"id": passage_id, "text": text, "embedding": emb}, tx)
        return passage_id

    # ---------- Entity ----------
    def add_entity(self, name: str, entity_type: str, passage_id: str, tx=None) -> None:
        self._ensure_schema(entity_type, "entity_type", tx)
        existing = self._get_existing_entity(name, tx)

        if existing is None:
            emb = self._get_embedding(name) if self.embedding_model else None
            query = """
            CREATE (e:Entity {name: $name, type: $type, embedding: $embedding})
            WITH e
            MATCH (p:Passage {id: $passage_id})
            CREATE (e)-[:MENTIONED_IN]->(p)
            WITH e
            MATCH (s:Schema {name: $type})
            CREATE (e)-[:INSTANCE_OF]->(s)
            """
            self._execute(query, {
                "name": name,
                "type": entity_type,
                "embedding": emb,
                "passage_id": passage_id,
            }, tx)
        else:
            if self.update_type_if_unknown and existing["type"] == "UNKNOWN" and entity_type != "UNKNOWN":
                self._execute(
                    "MATCH (e:Entity {name: $name}) SET e.type = $new_type",
                    {"name": name, "new_type": entity_type},
                    tx
                )
            link_query = """
            MATCH (e:Entity {name: $name}), (p:Passage {id: $passage_id})
            MERGE (e)-[:MENTIONED_IN]->(p)
            """
            self._execute(link_query, {"name": name, "passage_id": passage_id}, tx)

    # ---------- Relation ----------
    def add_relation(self, head: str, relation: str, tail: str, passage_id: str, tx=None) -> None:
        self._ensure_schema(relation, "relation_type", tx)
        if self.auto_create_entities:
            if not self._entity_exists(head, tx):
                self.add_entity(head, "UNKNOWN", passage_id, tx)
            if not self._entity_exists(tail, tx):
                self.add_entity(tail, "UNKNOWN", passage_id, tx)
        else:
            if not self._entity_exists(head, tx):
                raise ValueError(f"Головная сущность '{head}' не найдена")
            if not self._entity_exists(tail, tx):
                raise ValueError(f"Хвостовая сущность '{tail}' не найдена")

        create_query = """
        MATCH (h:Entity {name: $head}), (t:Entity {name: $tail})
        MERGE (h)-[r:RELATION {type: $relation}]->(t)
        ON CREATE SET r.based_on = [$passage_id]
        ON MATCH SET r.based_on = CASE WHEN NOT $passage_id IN r.based_on THEN r.based_on + [$passage_id] ELSE r.based_on END
        """
        self._execute(create_query, {
            "head": head,
            "tail": tail,
            "relation": relation,
            "passage_id": passage_id,
        }, tx)

    # ---------- Остальные методы ----------
    def delete_relation(self, head: str, relation: str, tail: str) -> None:
        query = """
        MATCH (h:Entity {name: $head})-[r:RELATION {type: $rel}]->(t:Entity {name: $tail})
        DELETE r
        """
        self._execute(query, {"head": head, "tail": tail, "rel": relation})

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
        return self._execute_read(query, {"name": start_entity, "limit": breadth})

    def vector_search(self, query_text: str = None, top_k: int = 5, embedding: list = None) -> List[Dict[str, Any]]:
        if embedding is None:
            if not self.embedding_model:
                raise ValueError("Не задан embedding_model для векторного поиска")
            if not query_text:
                raise ValueError("Нужно указать query_text или embedding")
            embedding = self._get_embedding(query_text)

        query = """
        CALL vector_search.search($index_name, $top_k, $embedding)
        YIELD node, distance
        RETURN node.name AS name, node.type AS type, distance AS score
        """
        try:
            return self._execute_read(query, {
                "index_name": "entity_embeddings",
                "embedding": embedding,
                "top_k": top_k,
            })
        except Exception as e:
            if "Index not found" in str(e) or "does not exist" in str(e):
                raise RuntimeError("Векторный индекс 'entity_embeddings' не найден.") from e
            raise

    def vector_search_passages(self, query_text: str = None, top_k: int = 5, embedding: list = None) -> List[Dict[str, Any]]:
        if embedding is None:
            if not self.embedding_model:
                raise ValueError("Не задан embedding_model для векторного поиска")
            if not query_text:
                raise ValueError("Нужно указать query_text или embedding")
            embedding = self._get_embedding(query_text)

        query = """
        CALL vector_search.search($index_name, $top_k, $embedding)
        YIELD node, distance
        RETURN node.text AS text, distance AS score
        """
        try:
            return self._execute_read(query, {
                "index_name": "passage_embeddings",
                "embedding": embedding,
                "top_k": top_k,
            })
        except Exception as e:
            if "Index not found" in str(e) or "does not exist" in str(e):
                raise RuntimeError("Векторный индекс 'passage_embeddings' не найден.") from e
            raise

    def get_passages_for_entity(self, entity_name: str, limit: int = 3) -> List[str]:
        query = """
        MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(p:Passage)
        RETURN p.text AS text LIMIT $limit
        """
        res = self._execute_read(query, {"name": entity_name, "limit": limit})
        return [row["text"] for row in res]

    def get_relations_with_context(self, entity_name: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATION]-(other:Entity)
        UNWIND r.based_on AS bid
        MATCH (p:Passage {id: bid})
        RETURN e.name AS source, r.type AS relation, other.name AS target, p.text AS passage_text
        """
        return self._execute_read(query, {"name": entity_name})

    def get_all_relations_with_passage(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (h:Entity)-[r:RELATION]->(t:Entity)
        RETURN h.name AS head, r.type AS relation, t.name AS tail, r.based_on AS based_on
        """
        return self._execute_read(query)

    def get_passage_text_by_id(self, passage_id: str) -> Optional[str]:
        query = "MATCH (p:Passage {id: $id}) RETURN p.text AS text"
        res = self._execute_read(query, {"id": passage_id})
        return res[0]["text"] if res else None

    def clear(self) -> None:
        with self._lock:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

    def snapshot(self) -> None:
        with self._lock:
            with self.driver.session() as session:
                session.run("CREATE SNAPSHOT")
                logging.info("Снэпшот Memgraph создан.")