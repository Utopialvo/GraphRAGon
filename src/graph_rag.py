# src/graph_rag.py

"""
Модуль Graph RAG с поддержкой Graph of Thoughts (GoT).
Улучшен graceful shutdown: поддержка нескольких экземпляров, единая регистрация atexit.
Добавлена потокобезопасность при параллельном выполнении подвопросов.
"""

import logging
import time
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from utils import safe_parse_json


# ---------- глобальные настройки graceful shutdown ----------
_instances = []

def _atexit_shutdown():
    """Вызывается при завершении процесса, создаёт снепшоты для всех активных экземпляров."""
    for instance in _instances:
        if not instance._closed:
            try:
                instance.store.snapshot()
                logging.info(f"Graceful shutdown для экземпляра {id(instance)}: снепшот создан.")
            except Exception as e:
                logging.error(f"Ошибка graceful shutdown для экземпляра {id(instance)}: {e}")

atexit.register(_atexit_shutdown)


@dataclass
class GraphRAGConfig:
    """Конфигурация для GraphRAG."""
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
    got_parallel: bool = True
    got_max_workers: int = 4

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
        """Явное завершение работы: создаёт снепшот и удаляет экземпляр из списка активных."""
        if not self._closed:
            try:
                self.store.snapshot()
                logging.info("Снепшот создан при вызове close().")
            except Exception as e:
                logging.error(f"Ошибка при создании снепшота в close(): {e}")
            self._closed = True
            if self in _instances:
                _instances.remove(self)

    def _init_components(self) -> None:
        llm_config = LLMConfig(
            model_name=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
        )
        # основной LLM-клиент для однопоточных операций
        self.llm_client = LLMClient(llm_config, cache_db=self.config.cache_db)
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
            auto_create_entities=True,
            update_type_if_unknown=True,
        )
        self.entity_extractor = LLMEntityExtractor(llm_config)
        self.relation_extractor = RelationExtractor(llm_config)
        self.conflict_resolver = ConflictResolver(
            self.store,
            llm_config,
            max_relations_per_pair=self.config.max_relations_per_pair,
        )

    def _cleanup_cache_if_needed(self) -> None:
        if self.config.auto_cleanup_cache:
            deleted = self.llm_client.cleanup_cache(max_age_hours=self.config.cache_cleanup_hours)
            logging.info(f"Очистка кэша: удалено {deleted} записей")

    def process_text(self, text: str, chunk_size: Optional[int] = None,
                     overlap: Optional[int] = None, resolve_conflicts: bool = True) -> List[str]:
        start_time = time.time()
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.overlap

        logging.info("Начинаем обработку текста...")
        passage_ids = self.store.add_large_text(text, chunk_size, overlap)

        total_entities = 0
        total_relations = 0

        for pid in passage_ids:
            chunk = self.store.get_passage_text_by_id(pid)
            if not chunk:
                continue

            self.store.begin()
            try:
                entities = self.entity_extractor.extract(chunk)
                relations = self.relation_extractor.extract(chunk, entities)

                for entity in entities:
                    self.store.add_entity(entity.text, entity.type, pid)
                for rel in relations:
                    self.store.add_relation(rel.head, rel.relation, rel.tail, pid)

                self.store.commit()
                total_entities += len(entities)
                total_relations += len(relations)
                logging.info(f"Чанк {pid}: {len(entities)} сущностей, {len(relations)} отношений")
            except Exception as e:
                self.store.rollback()
                logging.error(f"Ошибка при обработке чанка {pid}: {e}")
                raise

        self.metrics["total_entities"] += total_entities
        self.metrics["total_relations"] += total_relations
        self.metrics["total_passages"] += len(passage_ids)

        if resolve_conflicts:
            logging.info("Разрешение конфликтов...")
            stats = self.conflict_resolver.detect_and_resolve(dry_run=False)
            logging.info(f"Конфликтов: {stats['conflicts_detected']}, разрешено: {stats['resolutions_applied']}")

        self.store.snapshot()
        elapsed = time.time() - start_time
        self.metrics["processing_time"] += elapsed
        logging.info(f"Обработка текста завершена за {elapsed:.2f} сек.")
        return passage_ids

    def _decompose_question(self, question: str) -> List[str]:
        prompt = f"""
Ты — система, помогающая отвечать на сложные вопросы. Разбей следующий вопрос на несколько более простых подвопросов, на которые можно ответить по отдельности, чтобы затем объединить ответы в полный ответ.
Верни ТОЛЬКО JSON-список строк с подвопросами.
Пример: ["Кто участвовал в действиях?","Для чего совершались действия?", "Какие действия выполнял субъект А?", "Какие действия выполняла субъект Б?", "В какое время совершались действия субъекта А по отношению к действиям субъекта Б"]
Вопрос: {question}
"""
        response = self.llm_client.chat(prompt, response_format={"type": "json_object"})
        try:
            data = safe_parse_json(response)
            if isinstance(data, list):
                return [str(q) for q in data]
            else:
                logging.warning("Некорректный ответ декомпозиции, возвращаем исходный вопрос")
                return [question]
        except Exception as e:
            logging.error(f"Ошибка декомпозиции: {e}")
            return [question]

    def _retrieve_context(self, query: str) -> str:
        hits = self.store.vector_search(query, top_k=self.config.top_k)
        context_parts = []

        for hit in hits:
            entity_name = hit['name']
            passages = self.store.get_passages_for_entity(entity_name, limit=2)
            context_parts.extend(passages)

            try:
                related = self.store.get_relations_with_context(entity_name)
                for rel in related[:self.config.random_walk_breadth]:
                    if rel['source'] == entity_name:
                        neighbor = rel['target']
                    else:
                        neighbor = rel['source']
                    neighbor_passages = self.store.get_passages_for_entity(neighbor, limit=1)
                    context_parts.extend(neighbor_passages)
            except Exception as e:
                logging.debug(f"Random walk для {entity_name} не удался: {e}")

        # Убираем дубликаты, сохраняя порядок
        unique_parts = list(dict.fromkeys(context_parts))
        return "\n".join(unique_parts)[:self.config.max_context_length]

    def _answer_subquestion(self, sub_q: str, llm_client: Optional[LLMClient] = None) -> str:
        """Отвечает на один подвопрос. Можно передать отдельный LLMClient для потоков."""
        context = self._retrieve_context(sub_q)
        client = llm_client if llm_client is not None else self.llm_client
        sub_prompt = f"""
На основе контекста ответь на подвопрос:
Подвопрос: {sub_q}
Контекст: {context if context else "Контекст отсутствует."}
Ответ (кратко):
"""
        try:
            return client.chat(sub_prompt)
        except Exception as e:
            logging.error(f"Ошибка при ответе на подвопрос '{sub_q}': {e}")
            return ""

    def _aggregate_answers(self, question: str, sub_answers: List[str], history_text: str) -> str:
        combined = "\n".join([f"- {ans}" for ans in sub_answers if ans])
        prompt = f"""
Ты — помощник, отвечающий на вопросы на русском языке на основе предоставленных ответов на подвопросы и истории диалога.

История диалога:
{history_text}

Ответы на подвопросы:
{combined}

Исходный вопрос пользователя: {question}

Сформулируй единый, связный и полный ответ на русском языке, используя информацию из ответов на подвопросы.
Ответ (кратко и по делу):
"""
        try:
            return self.llm_client.chat(prompt)
        except Exception as e:
            logging.error(f"Ошибка агрегации: {e}")
            return "Извините, не удалось сформулировать ответ."

    def ask(self, question: str, use_got: bool = True) -> str:
        start_time = time.time()
        history_text = self.history.get_history_as_text(last_n=self.config.history_max_size)

        if not use_got:
            context = self._retrieve_context(question)
            prompt = f"""
Ты — помощник, отвечающий на вопросы на русском языке на основе предоставленного контекста.
История диалога:
{history_text}

Контекст:
{context if context else "Контекст отсутствует."}

Вопрос: {question}
Ответ (кратко и по делу):
"""
            answer = self.llm_client.chat(prompt)
        else:
            sub_questions = self._decompose_question(question)
            logging.info(f"Декомпозиция: {sub_questions}")

            if self.config.got_parallel and len(sub_questions) > 1:
                sub_answers = [""] * len(sub_questions)
                # для каждого потока создаём свой LLMClient, чтобы избежать гонок в HTTP-клиенте
                llm_config = LLMConfig(
                    model_name=self.config.llm_model,
                    base_url=self.config.llm_base_url,
                    api_key=self.config.llm_api_key,
                    temperature=self.config.llm_temperature,
                )
                with ThreadPoolExecutor(max_workers=self.config.got_max_workers) as executor:
                    future_to_idx = {}
                    for idx, sub_q in enumerate(sub_questions):
                        # отдельный клиент на каждый поток
                        thread_client = LLMClient(llm_config, cache_db=self.config.cache_db)
                        future = executor.submit(self._answer_subquestion, sub_q, thread_client)
                        future_to_idx[future] = idx
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            sub_answers[idx] = future.result()
                        except Exception as e:
                            logging.error(f"Ошибка при обработке подвопроса {idx}: {e}")
                            sub_answers[idx] = ""
            else:
                sub_answers = [self._answer_subquestion(sub_q) for sub_q in sub_questions]

            answer = self._aggregate_answers(question, sub_answers, history_text)

        self.history.add_entry(question, answer)
        elapsed = time.time() - start_time
        logging.info(f"Ответ на вопрос занял {elapsed:.2f} сек.")
        return answer

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

    def clear(self) -> None:
        self.store.clear()
        self.history.clear()
        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "total_passages": 0,
            "processing_time": 0.0,
        }
        logging.info("Граф, история и метрики очищены.")

    def get_history(self) -> List[Dict[str, str]]:
        return [{"question": q, "answer": a} for q, a in self.history.get_history()]