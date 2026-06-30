# GraphRAGon/src/graph_rag.py
"""
Главный модуль GraphRAG.
Объединяет извлечение знаний, построение графа, разрешение конфликтов,
векторный поиск и Graph of Thoughts для ответа на вопросы.
"""
import logging
import time
import atexit
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
import logging

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from memgraph_store import MemgraphStore
from llm_entity_extract import LLMEntityExtractor
from relation_extractor import RelationExtractor
from conflict_resolver import ConflictResolver
from history_manager import HistoryManager
from got_graph import GoTGraph

_instances = []

def _atexit_shutdown():
    for instance in _instances:
        if not instance._closed:
            try:
                instance.store.snapshot()
                logging.info(f"Graceful shutdown для экземпляра {id(instance)}: снэпшот создан.")
            except Exception as e:
                logging.error(f"Ошибка graceful shutdown: {e}")
atexit.register(_atexit_shutdown)


@dataclass
class GraphRAGConfig:
    memgraph_host: str = "localhost"
    memgraph_port: int = 7687
    llm_model: str = "/models/Qwen3.5-9B-Q4_0.gguf"
    llm_base_url: str = "http://llm-vulcan:8080/v1"
    llm_api_key: str = "dummy"
    llm_temperature: float = 0.0
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_db: str = "llm_cache.db"
    chunk_size: int = 500
    overlap: int = 100
    history_max_size: int = 10
    top_k: int = 5
    max_context_length: int = 3000
    auto_cleanup_cache: bool = True
    cache_cleanup_hours: int = 1
    max_relations_per_pair: int = 10
    embedding_batch_size: int = 32
    embedding_precision: str = "float32"
    random_walk_depth: int = 2
    random_walk_breadth: int = 3
    got_enabled: bool = True
    got_num_thoughts: int = 3
    got_generation_temperature: float = 0.7
    got_refine_temperature: float = 0.5
    got_merge_temperature: float = 0.3
    got_num_refinements: int = 1
    got_merge_enabled: bool = True
    got_score_enabled: bool = True
    got_use_random_roles: bool = True
    got_iterations: int = 1
    got_select_top_k: int = 3
    got_use_select: bool = True
    got_roles: List[str] = field(default_factory=lambda: ["Аналитик", "Критик", "Стратег", "Скептик", "Учёный"])

    @classmethod
    def from_yaml(cls, path: str) -> "GraphRAGConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class GraphRAG:
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self._validate_config()
        self._closed = False
        _instances.append(self)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Инициализация LLM с кэшированием через современный API
        set_llm_cache(SQLiteCache(database_path=config.cache_db))
        self.llm = ChatOpenAI(
            model=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            max_retries=3
        )

        # Эмбеддинги
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'precision': config.embedding_precision}
        )

        # Текстовый сплиттер
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Хранилище графа
        self.store = MemgraphStore(
            host=config.memgraph_host,
            port=config.memgraph_port,
            embedding_model=self.embedding_model,
            database="memgraph"
        )

        # Извлечение сущностей и отношений
        self.entity_extractor = LLMEntityExtractor(self.llm)
        self.relation_extractor = RelationExtractor(self.llm)

        # Разрешение конфликтов
        self.conflict_resolver = ConflictResolver(
            store=self.store,
            llm=self.llm,
            max_relations_per_pair=config.max_relations_per_pair,
        )

        # История
        self.history = HistoryManager(max_size=config.history_max_size)

        # Graph of Thoughts
        if config.got_enabled:
            self.got_graph = GoTGraph(
                llm=self.llm,
                num_thoughts=config.got_num_thoughts,
                generation_temperature=config.got_generation_temperature,
                refine_temperature=config.got_refine_temperature,
                merge_temperature=config.got_merge_temperature,
                select_top_k=config.got_select_top_k,
                max_iterations=config.got_iterations,
                roles=config.got_roles,
                use_random_roles=config.got_use_random_roles,
                use_select=config.got_use_select,
                merge_enabled=config.got_merge_enabled,
                score_enabled=config.got_score_enabled,
            )
        else:
            self.got_graph = None

        # Очистка кэша LLM (старые записи)
        self._cleanup_cache_if_needed()

        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "total_passages": 0,
            "processing_time": 0.0,
        }

    def _validate_config(self) -> None:
        if self.config.chunk_size <= self.config.overlap:
            raise ValueError("chunk_size должен быть больше overlap")

    def _cleanup_cache_if_needed(self) -> None:
        if self.config.auto_cleanup_cache:
            try:
                cutoff = datetime.now() - timedelta(hours=self.config.cache_cleanup_hours)
                conn = sqlite3.connect(self.config.cache_db)
                cur = conn.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff,))
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                logging.info(f"Очистка кэша LLM: удалено {deleted} записей.")
            except Exception as e:
                logging.warning(f"Не удалось очистить кэш LLM: {e}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self, 'store'):
            try:
                self.store.snapshot()
            except Exception as e:
                logging.error(f"Ошибка при создании снэпшота: {e}")
        logging.info("GraphRAG закрыт.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_text(self, text: str, source: str = "unknown") -> List[str]:
        start_time = time.time()
        chunks = self.text_splitter.split_text(text)
        logging.info(f"Текст разбит на {len(chunks)} чанков.")

        passage_ids = []
        total_entities = 0
        total_relations = 0

        self.store.enable_batch_cache()
        try:
            for idx, chunk in enumerate(chunks):
                logging.info(f"Обработка чанка {idx+1}/{len(chunks)}...")
                entities = self.entity_extractor.extract(chunk)
                relations = self.relation_extractor.extract(chunk, entities)

                passage_id = f"{source}_{idx}"
                try:
                    with self.store.transaction() as tx:
                        self.store.add_passage(chunk, passage_id, tx=tx)
                        for ent in entities:
                            self.store.add_entity(ent.text, ent.type, passage_id, tx=tx)
                        for rel in relations:
                            self.store.add_relation(rel.head, rel.relation, rel.tail, passage_id, tx=tx)
                except Exception as e:
                    logging.error(f"Ошибка сохранения чанка {idx}: {e}. Чанк полностью откачен.")
                    continue

                passage_ids.append(passage_id)
                total_entities += len(entities)
                total_relations += len(relations)
        finally:
            self.store.disable_batch_cache()

        elapsed = time.time() - start_time
        self.metrics["total_entities"] += total_entities
        self.metrics["total_relations"] += total_relations
        self.metrics["total_passages"] += len(passage_ids)
        self.metrics["processing_time"] += elapsed

        logging.info(f"Обработка завершена за {elapsed:.2f} сек. "
                     f"Сущностей: {total_entities}, отношений: {total_relations}.")
        return passage_ids

    def _get_context(self, query: str) -> str:
        query_embedding = self.embedding_model.embed_query(query)
        entities = self.store.vector_search(embedding=query_embedding, top_k=self.config.top_k)
        passages = self.store.vector_search_passages(embedding=query_embedding, top_k=self.config.top_k)

        context_parts = []
        for entity in entities:
            entity_name = entity.get("name")
            if not entity_name:
                continue
            relations = self.store.get_relations_with_context(entity_name)
            for rel in relations:
                snippet = rel.get('passage_text', '')
                context_parts.append(f"{rel['source']} {rel['relation']} {rel['target']} | фрагмент: {snippet}")

        for passage in passages:
            text = passage.get("text", "")
            if text:
                context_parts.append(f"Пассаж: {text}")

        if self.config.random_walk_depth > 0:
            for entity in entities:
                entity_name = entity.get("name")
                if not entity_name:
                    continue
                walk = self.store.random_walk(
                    start_entity=entity_name,
                    depth=self.config.random_walk_depth,
                    breadth=self.config.random_walk_breadth,
                )
                for step in walk:
                    context_parts.append(f"{step.get('head', '')} {step.get('relation', '')} {step.get('tail', '')}")

        if not context_parts:
            return "Нет релевантной информации."

        context = "\n".join(context_parts)
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length]
        return context

    def ask(self, question: str, use_got: bool = None) -> str:
        if use_got is None:
            use_got = self.config.got_enabled

        context = self._get_context(question)

        if not use_got or not self.got_graph:
            prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
            response = self.llm.invoke(prompt)
            answer = response.content
        else:
            answer = self.got_graph.run(question=question, context=context)

        self.history.add_entry(question, answer)
        return answer

    def resolve_conflicts(self, dry_run: bool = False) -> Dict[str, Any]:
        return self.conflict_resolver.detect_and_resolve(dry_run=dry_run)

    def get_history(self, last_n: Optional[int] = None) -> List:
        return self.history.get_history(last_n)

    def clear_history(self) -> None:
        self.history.clear()

    def clear(self) -> None:
        self.store.clear()
        self.clear_history()
        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "total_passages": 0,
            "processing_time": 0.0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics