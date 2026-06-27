# src/got/operations.py

"""
Операции для Graph of Thoughts: генерация, оценка, улучшение, слияние.
"""
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .thought import Thought
from .prompter import Prompter
from .parser import Parser
from .roles import ROLES, get_role_description, get_random_role


class Operation(ABC):
    @abstractmethod
    def execute(
        self,
        inputs: List[Thought],
        prompter: Prompter,
        parser: Parser,
        llm_client,
        **kwargs
    ) -> List[Thought]:
        pass


class Generate(Operation):
    """
    Генерирует несколько мыслей-ответов, каждой назначается своя роль.
    Можно запретить повторение ролей в одном поколении.
    """
    def __init__(
        self,
        num_outputs: int = 3,
        temperature: float = 0.7,
        roles: List[str] = None,
        use_random_roles: bool = True,
        allow_duplicate_roles: bool = False,
    ):
        self.num_outputs = num_outputs
        self.temperature = temperature
        self.roles = roles or [role["name"] for role in ROLES]
        self.use_random_roles = use_random_roles
        self.allow_duplicate_roles = allow_duplicate_roles

    def execute(
        self,
        inputs: List[Thought],
        prompter: Prompter,
        parser: Parser,
        llm_client,
        **kwargs
    ) -> List[Thought]:
        context = kwargs.get('context', '')
        question = kwargs.get('question', '')

        if not inputs and (context or question):
            base_prompt = prompter.format_generate(context, question)
        else:
            combined = "\n".join([t.content for t in inputs])
            base_prompt = prompter.format_generate(combined, question)

        # Подбираем роли
        selected_roles = []
        available_roles = self.roles.copy()
        for _ in range(self.num_outputs):
            if self.use_random_roles:
                if not self.allow_duplicate_roles and available_roles:
                    role_name = random.choice(available_roles)
                    available_roles.remove(role_name)
                else:
                    role_name = random.choice(self.roles)
                selected_roles.append({"name": role_name})
            else:
                # Используем последовательно из списка
                idx = len(selected_roles) % len(self.roles)
                selected_roles.append({"name": self.roles[idx]})

        outputs = []
        for i, role in enumerate(selected_roles):
            role_name = role["name"]
            role_desc = get_role_description(role_name)
            prompt = prompter.format_generate_with_role(
                base_prompt, question, role_name, role_desc
            )
            if i > 0:
                prompt += "\n\nПостарайся дать ответ, отличающийся от предыдущих вариантов."
            response = llm_client.chat(prompt, temperature=self.temperature)
            thought = Thought(
                content=response,
                parent_ids=[t.id for t in inputs],
                metadata={"role": role_name}
            )
            outputs.append(thought)

        return outputs


class Score(Operation):
    """Оценивает каждую мысль по шкале от 0 до 10."""
    def execute(
        self,
        inputs: List[Thought],
        prompter: Prompter,
        parser: Parser,
        llm_client,
        **kwargs
    ) -> List[Thought]:
        question = kwargs.get('question', '')
        for thought in inputs:
            prompt = prompter.format_score(thought.content, question)
            response = llm_client.chat(prompt)
            thought.score = parser.parse_score(response)
        return inputs


class Refine(Operation):
    """Улучшает существующую мысль."""
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def execute(
        self,
        inputs: List[Thought],
        prompter: Prompter,
        parser: Parser,
        llm_client,
        **kwargs
    ) -> List[Thought]:
        question = kwargs.get('question', '')
        refined = []
        for thought in inputs:
            prompt = prompter.format_refine(thought.content, question)
            response = llm_client.chat(prompt, temperature=self.temperature)
            new_thought = Thought(
                content=response,
                parent_ids=[thought.id],
                metadata={"refined_from": thought.id}
            )
            refined.append(new_thought)
        return refined


class Merge(Operation):
    """Сливает несколько мыслей в одну итоговую."""
    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature

    def execute(
        self,
        inputs: List[Thought],
        prompter: Prompter,
        parser: Parser,
        llm_client,
        **kwargs
    ) -> List[Thought]:
        if not inputs:
            return []
        question = kwargs.get('question', '')
        thoughts_text = [t.content for t in inputs]
        prompt = prompter.format_merge(thoughts_text, question)
        response = llm_client.chat(prompt, temperature=self.temperature)
        merged = Thought(
            content=response,
            parent_ids=[t.id for t in inputs],
            metadata={"merged_from": [t.id for t in inputs]}
        )
        return [merged]


class Select(Operation):
    """Выбирает top_k мыслей по наибольшей оценке."""
    def __init__(self, top_k: int = 2, score_key: str = 'score'):
        self.top_k = top_k
        self.score_key = score_key

    def execute(self, inputs: List[Thought], prompter: Prompter, parser: Parser, llm_client, **kwargs) -> List[Thought]:
        if not inputs:
            return []
        sorted_inputs = sorted(inputs, key=lambda t: getattr(t, self.score_key, 0.0), reverse=True)
        return sorted_inputs[:self.top_k]