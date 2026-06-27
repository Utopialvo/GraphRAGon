# src/graph_rag.py

"""
Главный модуль GraphRAG.
Объединяет извлечение знаний, построение графа, разрешение конфликтов,
векторный поиск (сущности + пассажи) и Graph of Thoughts для ответа на вопросы.
"""
import logging
import time
import atexit
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import yaml

from llm_client import LLMConfig, LLMClient
from embedding_client import EmbeddingClient
from llm_entity_extract import LLMEntityExtractor, Entity
from relation_extractor import RelationExtractor, Relation
from memgraph_store import MemgraphStore
from conflict_resolver import ConflictResolver
from history_manager import HistoryManager

from got import (
    GoTController, Prompter, Parser, OperationNode, GraphOfOperations,
    Generate, Score, Select, Refine, Merge,
    Thought, ROLES, get_random_role
)

_instances = []

def _atexit_shutdown():
    for instance in _instances:
        if not instance._closed:
            try:
                instance.store.snapshot()
                logging.info(f"Graceful shutdown для экземпляра {id(instance)}: снэпшот создан.")
            except Exception as e:
                logging.error(f"Ошибка graceful shutdown для экземпляра {id(instance)}: {e}")
atexit.register(_atexit_shutdown)


@dataclass
class GraphRAGConfig:
    """Конфигурация GraphRAG."""
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
    got_roles: List[str] = field(default_factory=lambda: [r["name"] for r in ROLES])
    got_iterations: int = 1
    got_select_top_k: int = 3
    got_use_select: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "GraphRAGConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class GraphRAG:
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self._validate_config()
        self._init_components()
        self.history = HistoryManager(max_size=config.history_max_size)
        self._cleanup_cache_if_needed()
        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "total_passages": 0,
            "processing_time": 0.0,
        }
        self._closed = False
        _instances.append(self)

        # Настройка GoT
        self.got_prompter = Prompter()
        self.got_parser = Parser()
        self.got_controller = None
        if config.got_enabled:
            self.got_controller = GoTController(
                self.llm_client,
                self.got_prompter,
                self.got_parser
            )

    def _validate_config(self) -> None:
        if self.config.chunk_size <= self.config.overlap:
            raise ValueError("chunk_size должен быть больше overlap")
        try:
            import requests
            resp = requests.get(f"{self.config.llm_base_url}/models", timeout=5)
            if resp.status_code != 200:
                logging.warning(f"LLM endpoint {self.config.llm_base_url} не отвечает")
        except Exception as e:
            logging.warning(f"Не удалось проверить LLM endpoint: {e}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self, 'store'):
            try:
                self.store.snapshot()
            except Exception as e:
                logging.error(f"Ошибка при создании снэпшота: {e}")
        if hasattr(self, 'embedding_client'):
            try:
                del self.embedding_client
            except Exception as e:
                logging.error(f"Ошибка при закрытии embedding_client: {e}")
        logging.info("GraphRAG закрыт.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_components(self) -> None:
        self.llm_config = LLMConfig(
            model_name=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
        )
        self.llm_client = LLMClient(self.llm_config, cache_db=self.config.cache_db)
        self.embedding_client = EmbeddingClient(
            model_name=self.config.embedding_model,
            cache_db=self.config.cache_db,
            batch_size=self.config.embedding_batch_size,
            precision=self.config.embedding_precision,
        )
        self.store = MemgraphStore(
            host=self.config.memgraph_host,
            port=self.config.memgraph_port,
            embedding_client=self.embedding_client,
        )
        self.entity_extractor = LLMEntityExtractor(self.llm_config)
        self.relation_extractor = RelationExtractor(self.llm_config)
        self.conflict_resolver = ConflictResolver(
            store=self.store,
            llm_config=self.llm_config,
            max_relations_per_pair=self.config.max_relations_per_pair,
        )

    def _cleanup_cache_if_needed(self) -> None:
        if self.config.auto_cleanup_cache:
            try:
                deleted = self.llm_client.cleanup_cache(max_age_hours=self.config.cache_cleanup_hours)
                logging.info(f"Очистка кэша: удалено {deleted} записей.")
            except Exception as e:
                logging.warning(f"Ошибка при очистке кэша: {e}")

    def _split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        max_iterations = text_len // max(1, self.config.chunk_size - self.config.overlap) + 2
        iterations = 0
        while start < text_len and iterations < max_iterations:
            iterations += 1
            end = min(start + self.config.chunk_size, text_len)
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            chunks.append(text[start:end])
            start = end - self.config.overlap
            if start < 0:
                start = 0
            if start >= text_len:
                break
        return chunks

    def process_text(self, text: str, source: str = "unknown") -> List[str]:
        start_time = time.time()
        chunks = self._split_text(text)
        logging.info(f"Текст разбит на {len(chunks)} чанков.")
    
        passage_ids = []
        total_entities = 0
        total_relations = 0
    
        self.store.enable_batch_cache()
        try:
            for idx, chunk in enumerate(chunks):
                logging.info(f"Обработка чанка {idx+1}/{len(chunks)}...")
                logging.debug("Извлечение сущностей...")
                entities = self.entity_extractor.extract(chunk)
                logging.debug(f"Найдено {len(entities)} сущностей")
                logging.debug("Извлечение отношений...")
                relations = self.relation_extractor.extract(chunk, entities)
                logging.debug(f"Найдено {len(relations)} отношений")
    
                passage_id = f"{source}_{idx}"
                try:
                    self.store.begin()
                    self.store.add_passage(chunk, passage_id)
                    for ent in entities:
                        self.store.add_entity(ent.text, ent.type, passage_id)
                    for rel in relations:
                        self.store.add_relation(rel.head, rel.relation, rel.tail, passage_id)
                    self.store.commit()
                except Exception as e:
                    try:
                        self.store.rollback()
                    except Exception:
                        pass
                    logging.error(f"Ошибка при сохранении чанка {idx}: {e}")
                    continue
    
                passage_ids.append(passage_id)
                total_entities += len(entities)
                total_relations += len(relations)
                del entities, relations
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
        """
        Гибридный семантический контекст:
          1. Векторный поиск по сущностям → извлечение отношений с текстом пассажа.
          2. Векторный поиск по пассажам → добавление фрагментов.
          3. Случайные блуждания при необходимости.
        """
        query_embedding = self.embedding_client.embed(query)

        # 1. Сущности
        entities = self.store.vector_search(embedding=query_embedding, top_k=self.config.top_k)

        # 2. Пассажи
        passages = self.store.vector_search_passages(embedding=query_embedding, top_k=self.config.top_k)

        context_parts = []

        # Отношения для найденных сущностей с фрагментами пассажей
        for entity in entities:
            entity_name = entity.get("name")
            if not entity_name:
                continue
            relations = self.store.get_relations_with_context(entity_name)
            for rel in relations:
                passage_snippet = rel.get('passage_text', '')[:200]
                context_parts.append(f"{rel['source']} {rel['relation']} {rel['target']} | фрагмент: {passage_snippet}")

        # Тексты пассажей
        for passage in passages:
            text = passage.get("text", "")
            if text:
                context_parts.append(f"Пассаж: {text[:500]}")

        # 3. Random walk
        if self.config.random_walk_depth > 0 and self.config.random_walk_breadth > 0:
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
                    if isinstance(step, dict):
                        context_parts.append(
                            f"{step.get('head', '')} {step.get('relation', '')} {step.get('tail', '')}"
                        )

        if not context_parts:
            return "Нет релевантной информации."

        context = "\n".join(context_parts)
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length]
        return context

    def ask(self, question: str, use_got: bool = None, **kwargs) -> str:
        if use_got is None:
            use_got = self.config.got_enabled

        context = self._get_context(question)

        if not use_got:
            prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
            answer = self.llm_client.chat(prompt)
            self.history.add_entry(question, answer)
            return answer

        # GoT
        gen_node = OperationNode(
            Generate(num_outputs=self.config.got_num_thoughts,
                     temperature=self.config.got_generation_temperature,
                     roles=self.config.got_roles,
                     use_random_roles=self.config.got_use_random_roles),
            dependencies=[],
            params={'context': context, 'question': question}
        )

        score_node = OperationNode(
            Score(),
            dependencies=[gen_node],
            params={'question': question}
        )
        
        nodes = [gen_node, score_node]
        current_dep = score_node

        if self.config.got_use_select:
            select_node = OperationNode(
                Select(top_k=self.config.got_select_top_k),
                dependencies=[current_dep],
                params={}
            )
            nodes.append(select_node)
            current_dep = select_node

        for _ in range(self.config.got_iterations):
            refine_node = OperationNode(
                Refine(temperature=self.config.got_refine_temperature),
                dependencies=[current_dep],
                params={'question': question}
            )
            nodes.append(refine_node)
            current_dep = refine_node

        if self.config.got_merge_enabled:
            merge_node = OperationNode(
                Merge(temperature=self.config.got_merge_temperature),
                dependencies=[current_dep],
                params={'question': question}
            )
            nodes.append(merge_node)
            current_dep = merge_node

        goo = GraphOfOperations(nodes)

        final_thoughts = self.got_controller.run_goo(goo)
        answer = final_thoughts[-1].content if final_thoughts else "Не удалось сгенерировать ответ."

        self.history.add_entry(question, answer)
        return answer

    def resolve_conflicts(self, dry_run: bool = False) -> Dict[str, Any]:
        return self.conflict_resolver.detect_and_resolve(dry_run=dry_run)

    def get_history(self, last_n: Optional[int] = None) -> List:
        return self.history.get_history(last_n)

    def clear_history(self) -> None:
        self.history.clear()

    def clear(self) -> None:
        """Очищает граф и историю."""
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