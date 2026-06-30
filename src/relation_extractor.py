# GraphRAGon/src/relation_extractor.py
"""
Извлечение отношений между сущностями с помощью LangChain и структурированного вывода.
"""
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class Relation(BaseModel):
    head: str = Field(description="Головная сущность")
    relation: str = Field(description="Тип отношения (глагол или фраза)")
    tail: str = Field(description="Хвостовая сущность")

class RelationsList(BaseModel):
    relations: List[Relation] = Field(default_factory=list)

class RelationExtractor:
    def __init__(self, llm: ChatOpenAI, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """\
        <role>Извлечение отношений</role>
        <task>Найди отношения между перечисленными сущностями, опираясь на текст.  
        Отношение – краткий глагол или фраза (инфинитив или прошедшее время).  
        Верни только JSON-список объектов с ключами "head", "relation", "tail".  
        Если отношений нет – верни пустой список.</task>
        <example>
        Сущности: Иван, дрова, сарай, утро, нарубил, печь, Варвара, растопила
        Текст: "Утром Иван нарубил дрова в сарае. Затем Варвара растопила печь."
        Ответ:
        [
          {{"head": "Иван", "relation": "нарубил", "tail": "дрова"}},
          {{"head": "Иван", "relation": "находился в", "tail": "сарай"}},
          {{"head": "Варвара", "relation": "растопила", "tail": "печь"}}
        ]
        </example>
        <example>
        Сущности: рыба, Иван, речка, ловить
        Текст: "Иван пошёл на речку и стал ловить рыбу."
        Ответ:
        [
          {{"head": "Иван", "relation": "пошёл на", "tail": "речка"}},
          {{"head": "Иван", "relation": "ловил", "tail": "рыбу"}}
        ]
        </example>"""),
            ("human", "<entities>{entities}</entities>\n<text>{text}</text>")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(RelationsList)

    def extract(self, text: str, entities: List) -> List[Relation]:
        entity_texts = [e.text if hasattr(e, 'text') else e.get('text', '') for e in entities]
        entities_str = ", ".join(entity_texts)
        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke({"entities": entities_str, "text": text})
                return result.relations
            except Exception as e:
                logging.warning(f"Попытка {attempt+1}/{self.max_retries+1} не удалась: {e}")
        logging.error("Не удалось извлечь отношения после всех попыток.")
        return []