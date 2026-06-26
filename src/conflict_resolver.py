# src/conflict_resolver.py

"""
Модуль, который разруливает конфликты между отношениями в графе знаний.
При слиянии (merge) в новое отношение попадают все ID пассажей из обоих исходных отношений.
Больше не пытаемся удалить уже удалённое — конфликты разрешаются пошагово внутри каждой пары сущностей.
"""

import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, ValidationError

from llm_client import LLMClient, LLMConfig
from memgraph_store import MemgraphStore
from utils import safe_parse_json


class ConflictResolution(BaseModel):
    conflict: bool
    resolution: Optional[str] = None   # keep_first, keep_second, merge
    new_relation: Optional[str] = None
    explanation: Optional[str] = None


class ConflictResolver:
    def __init__(
        self,
        store: MemgraphStore,
        llm_config: LLMConfig,
        max_retries: int = 2,
        max_relations_per_pair: int = 10,
    ):
        self.store = store
        self.llm_client = LLMClient(llm_config)
        self.max_retries = max_retries
        self.max_relations_per_pair = max_relations_per_pair
        self.conflict_prompt = """
Ты — система разрешения конфликтов в графе знаний.
Даны две тройки (субъект, отношение, объект), которые относятся к одним и тем же субъекту и объекту.
Также даны текстовые фрагменты (Passage), на основе которых были извлечены эти отношения.
Определи, являются ли эти отношения конфликтующими (противоречащими друг другу) или совместимыми.
Если они конфликтуют, предложи, какое отношение следует оставить (или как их объединить).
Верни ТОЛЬКО JSON с полями:
- "conflict": true/false
- "resolution": "keep_first", "keep_second", "merge" (если conflict == true)
- "new_relation": (если resolution == "merge") предложение нового отношения (строка)
- "explanation": краткое пояснение

Тройка 1: {triple1}
Текст основания 1: {passage1}
Тройка 2: {triple2}
Текст основания 2: {passage2}
Ответ:
"""

    def _get_resolution(self, triple1: str, passage1: str, triple2: str, passage2: str) -> Optional[ConflictResolution]:
        prompt = self.conflict_prompt.format(
            triple1=triple1,
            passage1=passage1,
            triple2=triple2,
            passage2=passage2,
        )
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat(prompt, response_format={"type": "json_object"})
                data = safe_parse_json(response)
                if data is None:
                    continue
                resolution = ConflictResolution(**data)
                if resolution.conflict and resolution.resolution not in ("keep_first", "keep_second", "merge"):
                    logging.warning(f"Некорректное разрешение: {resolution.resolution}, повторная попытка")
                    continue
                return resolution
            except (ValidationError, Exception) as e:
                logging.warning(f"Попытка {attempt+1}/{self.max_retries} не удалась: {e}")
                if attempt == self.max_retries - 1:
                    logging.error(
                        f"Не удалось получить валидное разрешение после {self.max_retries} попыток. "
                        f"Ответ: {response[:200] if 'response' in locals() else 'нет ответа'}"
                    )
                    return None
        return None

    def detect_and_resolve(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Проходим по всем парам сущностей, для каждой группы отношений вызываем пошаговое разрешение.
        """
        triples = self.store.get_all_relations_with_passage()
        groups = defaultdict(list)
        for t in triples:
            groups[(t['head'], t['tail'])].append(t)

        all_conflicts = []

        for (head, tail), rel_list in groups.items():
            if len(rel_list) < 2:
                continue

            if len(rel_list) > self.max_relations_per_pair:
                logging.warning(
                    f"Паре ({head}, {tail}) найдено {len(rel_list)} отношений, "
                    f"обрабатываем только первые {self.max_relations_per_pair}"
                )
                rel_list = rel_list[:self.max_relations_per_pair]

            # нормализуем based_on к списку
            normalized = []
            for r in rel_list:
                based_on = r['based_on'] if isinstance(r['based_on'], list) else [r['based_on']]
                normalized.append({
                    'relation': r['relation'],
                    'based_on': based_on
                })

            group_conflicts = self._resolve_group(head, tail, normalized, dry_run)
            all_conflicts.extend(group_conflicts)

        return {
            "conflicts_detected": len(all_conflicts),
            "conflicts": all_conflicts,
            "resolutions_applied": len(all_conflicts) if not dry_run else 0,
        }

    def _resolve_group(self, head: str, tail: str, relations: List[Dict[str, Any]], dry_run: bool) -> List[Dict[str, Any]]:
        """
        Итеративно разрешает конфликты внутри одной пары (head, tail).
        После каждого применённого действия обновляет локальный список отношений,
        чтобы не пытаться удалить уже удалённое.
        """
        resolved = []
        # рабочая копия списка отношений
        active = [{'relation': r['relation'], 'based_on': list(r['based_on'])} for r in relations]

        # если осталось 0 или 1 отношение, конфликтов быть не может
        while len(active) > 1:
            found_conflict = False
            n = len(active)
            for i in range(n):
                for j in range(i + 1, n):
                    r1 = active[i]
                    r2 = active[j]
                    if r1['relation'] == r2['relation']:
                        continue

                    # ищем текст первого пассажа для каждого отношения
                    p1_id = r1['based_on'][0] if r1['based_on'] else None
                    p2_id = r2['based_on'][0] if r2['based_on'] else None
                    passage1 = self.store.get_passage_text_by_id(p1_id) or "неизвестно"
                    passage2 = self.store.get_passage_text_by_id(p2_id) or "неизвестно"

                    triple1 = f"({head}, {r1['relation']}, {tail})"
                    triple2 = f"({head}, {r2['relation']}, {tail})"
                    resolution = self._get_resolution(triple1, passage1, triple2, passage2)

                    if not resolution or not resolution.conflict:
                        continue

                    conflict_entry = {
                        "head": head,
                        "tail": tail,
                        "relation1": r1['relation'],
                        "relation2": r2['relation'],
                        "based_on1": r1['based_on'],
                        "based_on2": r2['based_on'],
                        "resolution": resolution.resolution,
                        "new_relation": resolution.new_relation,
                        "explanation": resolution.explanation,
                    }
                    resolved.append(conflict_entry)

                    if not dry_run:
                        # Применяем изменения к графу и перестраиваем active
                        if resolution.resolution == "keep_first":
                            self.store.delete_relation(head, r2['relation'], tail)
                            # оставляем только r1, r2 выкидываем
                            active = [r for idx, r in enumerate(active) if idx != j]
                        elif resolution.resolution == "keep_second":
                            self.store.delete_relation(head, r1['relation'], tail)
                            active = [r for idx, r in enumerate(active) if idx != i]
                        elif resolution.resolution == "merge":
                            self.store.delete_relation(head, r1['relation'], tail)
                            self.store.delete_relation(head, r2['relation'], tail)
                            merged_rel = resolution.new_relation or "связан"
                            for bid in r1['based_on']:
                                self.store.add_relation(head, merged_rel, tail, bid)
                            for bid in r2['based_on']:
                                self.store.add_relation(head, merged_rel, tail, bid)
                            # новое объединённое отношение
                            new_r = {
                                'relation': merged_rel,
                                'based_on': r1['based_on'] + r2['based_on']
                            }
                            # убираем оба старых, добавляем новое
                            active = [r for idx, r in enumerate(active) if idx != i and idx != j]
                            active.append(new_r)
                        else:
                            logging.warning(f"Неизвестное разрешение: {resolution.resolution}")

                    # после любого изменения списка – начинаем цикл while заново
                    found_conflict = True
                    break
                if found_conflict:
                    break
            if not found_conflict:
                break   # больше конфликтов нет

        return resolved