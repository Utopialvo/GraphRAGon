# GraphRAGon/src/conflict_resolver.py
"""
Разрешение конфликтов между отношениями в графе знаний.
"""
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ConflictResolution(BaseModel):
    conflict: bool = Field(description="Есть ли конфликт")
    resolution: Optional[str] = Field(default=None, description="keep_first, keep_second или merge")
    new_relation: Optional[str] = Field(default=None, description="Новое отношение при merge")
    explanation: Optional[str] = Field(default=None, description="Пояснение")

class ConflictResolver:
    def __init__(
        self,
        store,
        llm: ChatOpenAI,
        max_retries: int = 2,
        max_relations_per_pair: int = 10,
    ):
        self.store = store
        self.llm = llm
        self.max_retries = max_retries
        self.max_relations_per_pair = max_relations_per_pair

        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """\
        <role>Разрешение конфликтов в графе знаний</role>
        <task>Даны две тройки (субъект, отношение, объект) для одной и той же пары сущностей,
        а также текстовые фрагменты, из которых они извлечены.
        Определи, противоречат ли эти отношения друг другу.
        Если конфликта нет – укажи "conflict": false.
        Если конфликт есть – выбери одно из решений:
        - "keep_first" – оставить первую тройку,
        - "keep_second" – оставить вторую тройку,
        - "merge" – объединить в новое отношение (поле "new_relation").
        Дай краткое пояснение в "explanation".</task>
        <rules>
        1. Если оба отношения могут быть истинны одновременно (например, "пошёл" и "ловил") – конфликта нет.
        2. Если одно исключает другое (противоположные действия, разные локации в одно время) – конфликт есть.
        3. При объединении выбери наиболее точное обобщение.
        </rules>
        <example>
        Тройка 1: (Иван, нарубил, дрова) | Пассаж 1: "Иван нарубил дрова"
        Тройка 2: (Иван, сломал, дрова) | Пассаж 2: "Иван сломал дрова"
        Ответ: {{"conflict": false}}
        </example>
        <example>
        Тройка 1: (Иван, пошёл в лес, утром) | Пассаж 1: "Иван пошёл в лес утром"
        Тройка 2: (Иван, остался дома, утром) | Пассаж 2: "Иван остался дома утром"
        Ответ: {{"conflict": true, "resolution": "keep_first", "new_relation": null, "explanation": "Второй пассаж противоречит первому, оставляем первый как более ранний."}}
        </example>"""),
            ("human", """\
        <case>
        <triple1>{triple1}</triple1>
        <passage1>{passage1}</passage1>
        <triple2>{triple2}</triple2>
        <passage2>{passage2}</passage2>
        </case>""")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(ConflictResolution)

    def _get_resolution(self, triple1: str, passage1: str, triple2: str, passage2: str) -> Optional[ConflictResolution]:
        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke({
                    "triple1": triple1,
                    "passage1": passage1,
                    "triple2": triple2,
                    "passage2": passage2,
                })
                if result.conflict and result.resolution not in ("keep_first", "keep_second", "merge"):
                    logging.warning(f"Некорректное разрешение: {result.resolution}")
                    continue
                return result
            except Exception as e:
                logging.warning(f"Попытка {attempt+1}/{self.max_retries} не удалась: {e}")
        return None

    def detect_and_resolve(self, dry_run: bool = False) -> Dict[str, Any]:
        triples = self.store.get_all_relations_with_passage()
        groups = defaultdict(list)
        for t in triples:
            groups[(t['head'], t['tail'])].append(t)

        all_conflicts = []
        for (head, tail), rel_list in groups.items():
            if len(rel_list) < 2:
                continue
            if len(rel_list) > self.max_relations_per_pair:
                logging.warning(f"Паре ({head}, {tail}) найдено {len(rel_list)} отношений, обрабатываем первые {self.max_relations_per_pair}")
                rel_list = rel_list[:self.max_relations_per_pair]

            normalized = []
            for r in rel_list:
                based_on = r['based_on'] if isinstance(r['based_on'], list) else [r['based_on']]
                normalized.append({'relation': r['relation'], 'based_on': based_on})

            group_conflicts = self._resolve_group(head, tail, normalized, dry_run)
            all_conflicts.extend(group_conflicts)

        return {
            "conflicts_detected": len(all_conflicts),
            "conflicts": all_conflicts,
            "resolutions_applied": len(all_conflicts) if not dry_run else 0,
        }

    def _resolve_group(self, head: str, tail: str, relations: List[Dict[str, Any]], dry_run: bool) -> List[Dict[str, Any]]:
        resolved = []
        active = [{'relation': r['relation'], 'based_on': list(r['based_on'])} for r in relations]
        max_loop = len(active) * 2  # защита от бесконечного цикла
        iterations = 0

        while len(active) > 1 and iterations < max_loop:
            iterations += 1
            found_conflict = False
            n = len(active)
            for i in range(n):
                for j in range(i + 1, n):
                    r1 = active[i]
                    r2 = active[j]
                    if r1['relation'] == r2['relation']:
                        continue

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
                        if resolution.resolution == "keep_first":
                            self.store.delete_relation(head, r2['relation'], tail)
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
                            new_r = {
                                'relation': merged_rel,
                                'based_on': r1['based_on'] + r2['based_on']
                            }
                            active = [r for idx, r in enumerate(active) if idx != i and idx != j]
                            active.append(new_r)
                        else:
                            logging.warning(f"Неизвестное разрешение: {resolution.resolution}")

                    found_conflict = True
                    break
                if found_conflict:
                    break
            if not found_conflict:
                break
        if iterations >= max_loop:
            logging.warning(f"Достигнут лимит итераций при разрешении конфликтов для пары ({head}, {tail})")
        return resolved