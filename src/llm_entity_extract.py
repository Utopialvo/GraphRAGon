# GraphRAGon/src/llm_entity_extract.py
"""
Извлечение именованных сущностей с помощью LangChain и структурированного вывода.
"""
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class Entity(BaseModel):
    text: str = Field(description="Текст сущности")
    type: str = Field(description="Тип сущности (PERSON, LOCATION, OBJECT, ACTION, TIME и т.д.)")

class EntitiesList(BaseModel):
    entities: List[Entity] = Field(default_factory=list)

class LLMEntityExtractor:
    def __init__(self, llm: ChatOpenAI, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """\
        <role>Извлечение именованных сущностей</role>
        <task>Найди в тексте все именованные сущности: люди, места, предметы, действия, временные отрезки.  
        Верни только JSON-список объектов с ключами "text" и "type".  
        Если сущностей нет – верни пустой список.</task>
        <types>PERSON, LOCATION, OBJECT, ACTION, TIME</types>
        <example>
        Текст: "Утром Иван нарубил дрова в сарае."
        Ответ:
        [
          {{"text": "утром", "type": "TIME"}},
          {{"text": "Иван", "type": "PERSON"}},
          {{"text": "нарубил", "type": "ACTION"}},
          {{"text": "дрова", "type": "OBJECT"}},
          {{"text": "сарай", "type": "LOCATION"}}
        ]
        </example>
        <example>
        Текст: "Дождь шёл весь день."
        Ответ:
        [
          {{"text": "дождь", "type": "OBJECT"}},
          {{"text": "шёл", "type": "ACTION"}},
          {{"text": "день", "type": "TIME"}}
        ]
        </example>"""),
            ("human", "<text>{text}</text>")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(EntitiesList)

    def extract(self, text: str) -> List[Entity]:
        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke({"text": text})
                return result.entities
            except Exception as e:
                logging.warning(f"Попытка {attempt+1}/{self.max_retries+1} не удалась: {e}")
        logging.error("Не удалось извлечь сущности после всех попыток.")
        return []