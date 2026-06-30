# GraphRAGon/src/got_graph.py
"""
Graph of Thoughts на LangGraph с поддержкой всех параметров из config.yaml.
"""
import logging
import random
from typing import List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


ROLES_DESCRIPTIONS = {
    "Аналитик": (
        "Ты — аналитик. Твоя задача — разложить вопрос на составляющие, выявить скрытые взаимосвязи и логические цепочки. "
        "Ты мыслишь структурно, предпочитаешь чёткие схемы и факты. Избегай субъективных оценок, стремись к объективности и полноте.\n\n"
        "Порядок действий:\n"
        "1. Выдели ключевые понятия и переменные в вопросе.\n"
        "2. Определи причинно-следственные связи и зависимости между ними.\n"
        "3. Построй логическую цепочку: от исходных данных к выводам.\n"
        "4. Проверь каждый шаг на непротиворечивость.\n"
        "5. Представь итоговый ответ в виде чёткой структуры (тезис → аргументы → вывод)."
    ),
    "Критик": (
        "Ты — критик. Твоя задача — подвергать сомнению любые утверждения, искать слабые места, логические ошибки и нестыковки. "
        "Ты не соглашаешься сразу, а проверяешь каждую деталь. Твоя цель — выявить уязвимости и предложить альтернативные трактовки.\n\n"
        "Порядок действий:\n"
        "1. Перечисли все допущения, которые делаются в исходном контексте.\n"
        "2. Для каждого допущения задай вопрос: «А что, если это не так?»\n"
        "3. Найди возможные противоречия между фактами и выводами.\n"
        "4. Предложи альтернативное объяснение или сценарий.\n"
        "5. Сформулируй, какие дополнительные данные могли бы разрешить сомнения."
    ),
    "Стратег": (
        "Ты — стратег. Ты смотришь на вопрос с высоты птичьего полёта, оцениваешь долгосрочные последствия и возможные сценарии. "
        "Твои ответы ориентированы на действие, ты выделяешь приоритеты и предлагаешь пошаговые планы. Ты учитываешь риски и ресурсы.\n\n"
        "Порядок действий:\n"
        "1. Определи конечную цель и критерии успеха.\n"
        "2. Разработай 2–3 возможных сценария развития событий.\n"
        "3. Для каждого сценария оцени ресурсы, риски и выгоды.\n"
        "4. Выбери оптимальный путь и разбей его на конкретные этапы.\n"
        "5. Укажи контрольные точки для отслеживания прогресса."
    ),
    "Скептик": (
        "Ты — скептик. Ты всегда ищешь скрытые допущения, неочевидные факторы и возможные ошибки в исходных данных. "
        "Ты не веришь на слово, а требуешь доказательств. Твои ответы полны вопросов и уточнений, ты часто указываешь на пробелы в информации.\n\n"
        "Порядок действий:\n"
        "1. Определи, какие данные или свидетельства лежат в основе утверждений.\n"
        "2. Проверь их достаточность и надёжность: откуда они взялись? можно ли им доверять?\n"
        "3. Выяви пробелы: какой информации не хватает для однозначного ответа?\n"
        "4. Сформулируй список уточняющих вопросов, которые необходимо задать.\n"
        "5. Предложи минимально жизнеспособный вывод, основанный только на проверенных фактах."
    ),
    "Учёный": (
        "Ты — учёный. Ты основываешься на проверенных научных данных, экспериментах и эмпирических исследованиях. "
        "Твои ответы строги, точны и подкреплены логикой. Ты избегаешь спекуляций и предпочитаешь ссылаться на известные теории и факты.\n\n"
        "Порядок действий:\n"
        "1. Сформулируй гипотезу, которая отвечает на вопрос.\n"
        "2. Подбери известные теории, законы или экспериментальные данные, подтверждающие или опровергающие её.\n"
        "3. Проведи мысленный эксперимент: какие результаты ожидаются, если гипотеза верна?\n"
        "4. Обозначь границы применимости вывода.\n"
        "5. Сделай заключение, указав степень уверенности (например, «высокая», «средняя», «требует проверки»)."
    )
}

class GoTState(TypedDict):
    question: str
    context: str
    num_thoughts: int
    thoughts: List[str]
    scores: List[float]
    iterations: int
    max_iterations: int
    top_k: int
    final_answer: str

class GoTGraph:
    def __init__(
        self,
        llm: ChatOpenAI,
        num_thoughts: int = 3,
        generation_temperature: float = 0.7,
        refine_temperature: float = 0.5,
        merge_temperature: float = 0.3,
        select_top_k: int = 3,
        max_iterations: int = 1,
        roles: Optional[List[str]] = None,
        use_random_roles: bool = True,
        use_select: bool = True,
        merge_enabled: bool = True,
        score_enabled: bool = True,
    ):
        self.llm = llm
        self.num_thoughts = num_thoughts
        self.generation_temperature = generation_temperature
        self.refine_temperature = refine_temperature
        self.merge_temperature = merge_temperature
        self.select_top_k = select_top_k
        self.max_iterations = max_iterations
        self.use_select = use_select
        self.merge_enabled = merge_enabled
        self.score_enabled = score_enabled
        self.use_random_roles = use_random_roles

        # Формируем список ролей с описаниями
        if roles:
            self.roles = []
            for role_name in roles:
                desc = ROLES_DESCRIPTIONS.get(role_name, f"Ты — {role_name}. Дай развернутый ответ.")
                self.roles.append({"name": role_name, "description": desc})
        else:
            # По умолчанию пять стандартных ролей
            self.roles = [{"name": name, "description": desc} for name, desc in ROLES_DESCRIPTIONS.items()]

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GoTState)

        builder.add_node("generate", self._generate)

        if self.score_enabled:
            builder.add_node("score", self._score)
            builder.add_edge("generate", "score")
        else:
            # Если оценка отключена, присваиваем всем мыслям одинаковый балл
            def fake_score(state: GoTState):
                return {"scores": [5.0] * len(state["thoughts"])}
            builder.add_node("score", fake_score)
            builder.add_edge("generate", "score")

        if self.use_select:
            builder.add_node("select", self._select)
            builder.add_edge("score", "select")
            # После select решаем, refine или merge
            builder.add_conditional_edges(
                "select",
                self._decide_after_select,
                {"refine": "refine", "merge": "merge"}
            )
        else:
            # Без select сразу решаем
            builder.add_conditional_edges(
                "score",
                self._decide_after_score,
                {"refine": "refine", "merge": "merge"}
            )

        builder.add_node("refine", self._refine)
        builder.add_edge("refine", "score")  # возвращаемся на оценку после улучшения

        builder.add_node("merge", self._merge)
        builder.add_edge("merge", END)

        builder.set_entry_point("generate")
        return builder.compile()

    def _decide_after_select(self, state: GoTState) -> str:
        if state["iterations"] < state["max_iterations"]:
            return "refine"
        else:
            return "merge"

    def _decide_after_score(self, state: GoTState) -> str:
        # если select нет, решение принимается сразу после score
        return self._decide_after_select(state)

    def _generate(self, state: GoTState) -> dict:
        question = state["question"]
        context = state["context"]
        num = state.get("num_thoughts", self.num_thoughts)
        thoughts = []

        for i in range(num):
            # Выбираем роль: случайно или по порядку
            if self.use_random_roles and self.roles:
                role = random.choice(self.roles)
            else:
                role = self.roles[i % len(self.roles)]

            prompt = ChatPromptTemplate.from_messages([
            ("system", """\
            <role>{role_name}</role>
            <description>{role_description}</description>
            <instruction>
            Ответь на вопрос, используя приведённый контекст.
            Следуй своему порядку действий. Дай развёрнутый, структурированный ответ.
            </instruction>"""),
            ("human", """\
            <context>
            {context}
            </context>
            <question>{question}</question>
            <answer>""")
            ])
            
            try:
                llm_gen = self.llm.bind(temperature=self.generation_temperature)
                chain = prompt | llm_gen
                response = chain.invoke({"context": context, "question": question})
                thoughts.append(response.content if hasattr(response, 'content') else str(response))
            except Exception as e:
                logging.error(f"Ошибка генерации мысли: {e}")
                thoughts.append("")
        return {"thoughts": thoughts, "iterations": 0}

    def _score(self, state: GoTState) -> dict:
        if not self.score_enabled:
            return {"scores": [5.0] * len(state["thoughts"])}
        question = state["question"]
        scores = []
        for thought in state["thoughts"]:
            prompt = ChatPromptTemplate.from_messages([
            ("system", """\
            <role>Оценщик ответов</role>
            <task>Оцени ответ на вопрос по шкале от 0 до 10. 0 – полностью неверно/нерелевантно, 10 – безупречно, полно и логично.
            Верни только число (целое или с одной десятичной).</task>
            <example>
            Вопрос: "Что делал Иван?"
            Ответ: "Иван рубил дрова и ходил на рыбалку."
            Оценка: 8
            </example>"""),
            ("human", """\
            <question>{question}</question>
            <answer_to_score>{thought}</answer_to_score>
            <score>""")
            ])
            chain = prompt | self.llm
            try:
                resp = chain.invoke({"question": question, "thought": thought})
                import re
                match = re.search(r'\b(\d+(\.\d+)?)\b', resp.content)
                score = float(match.group(1)) if match else 0.0
                scores.append(min(10.0, max(0.0, score)))
            except:
                scores.append(0.0)
        return {"scores": scores}

    def _select(self, state: GoTState) -> dict:
        if not self.use_select:
            return state  # без изменений
        thoughts = state["thoughts"]
        scores = state["scores"]
        top_k = state.get("top_k", self.select_top_k)
        if not thoughts:
            return {"thoughts": [], "scores": []}
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        selected_thoughts = [thoughts[i] for i, _ in indexed]
        return {"thoughts": selected_thoughts, "scores": [scores[i] for i, _ in indexed]}

    def _refine(self, state: GoTState) -> dict:
        question = state["question"]
        refined = []
        for thought in state["thoughts"]:
            prompt = ChatPromptTemplate.from_messages([
            ("system", """\
            <role>Редактор-улучшатель</role>
            <task>Улучши ответ: исправь неточности, добавь важные детали из контекста, улучши структуру.
            Не меняй смысл правильных утверждений. Верни только улучшенный ответ.</task>
            <example>
            Исходный ответ: "Иван рубил."
            Улучшенный ответ: "Иван утром нарубил дрова в сарае."
            </example>"""),
            ("human", """\
            <question>{question}</question>
            <original_answer>{thought}</original_answer>
            <improved_answer>""")
            ])

            try:
                llm_refine = self.llm.bind(temperature=self.refine_temperature)
                chain = prompt | llm_refine
                resp = chain.invoke({"question": question, "thought": thought})
                refined.append(resp.content)
            except:
                refined.append(thought)
        new_iterations = state.get("iterations", 0) + 1
        return {"thoughts": refined, "iterations": new_iterations}

    def _merge(self, state: GoTState) -> dict:
        question = state["question"]
        thoughts = state["thoughts"]
        if not thoughts:
            return {"final_answer": "Нет релевантной информации."}
        if not self.merge_enabled:
            # Без слияния берём первую мысль
            return {"final_answer": thoughts[0]}
        prompt = ChatPromptTemplate.from_messages([
        ("system", """\
        <role>Синтезатор ответов</role>
        <task>Объедини несколько вариантов ответа в один наилучший.
        Сохрани все ключевые факты, избегай повторений, выбери самые точные формулировки.
        Верни только итоговый связный текст.</task>
        <example>
        Варианты:
        - "Иван нарубил дрова и пошёл на речку."
        - "Иван ловил рыбу и читал книгу."
        Итоговый ответ: "Иван нарубил дрова, сходил на речку ловить рыбу, а вечером читал книгу."
        </example>"""),
        ("human", """\
        <question>{question}</question>
        <variants>
        {thoughts}
        </variants>
        <final_answer>""")
        ])
        thoughts_text = "\n".join([f"- {t}" for t in thoughts])
        try:
            llm_merge = self.llm.bind(temperature=self.merge_temperature)
            chain = prompt | llm_merge
            resp = chain.invoke({"question": question, "thoughts": thoughts_text})
            final = resp.content
        except:
            final = thoughts[0]
        return {"final_answer": final}

    def run(self, question: str, context: str) -> str:
        initial_state = {
            "question": question,
            "context": context,
            "num_thoughts": self.num_thoughts,
            "thoughts": [],
            "scores": [],
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "top_k": self.select_top_k,
            "final_answer": "",
        }
        result = self.graph.invoke(initial_state)
        return result["final_answer"]