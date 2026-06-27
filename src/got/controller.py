# src/got/controller.py

"""
Контроллер выполнения операций над графом мыслей.
"""
import logging
from typing import List, Dict, Any, Optional
from .graph_state import GraphReasoningState
from .operations import Operation
from .prompter import Prompter
from .parser import Parser
from .thought import Thought


class GoTController:
    def __init__(self, llm_client, prompter: Prompter = None, parser: Parser = None):
        self.llm_client = llm_client
        self.prompter = prompter or Prompter()
        self.parser = parser or Parser()
        self.state = GraphReasoningState()

    def execute_operation(self, operation: Operation, inputs: List[str] = None, **kwargs) -> List[str]:
        if inputs is None:
            current_thoughts = self.state.get_current_thoughts()
        else:
            current_thoughts = self.state.get_thoughts(inputs)

        outputs = operation.execute(
            current_thoughts,
            self.prompter,
            self.parser,
            self.llm_client,
            **kwargs
        )
        for thought in outputs:
            self.state.add_thought(thought)
        self.state.set_current([t.id for t in outputs])
        return [t.id for t in outputs]

    def run_pipeline(self, operations: List[Operation], initial_input: str = None, **kwargs) -> str:
        if initial_input:
            init_thought = Thought(content=initial_input)
            self.state.add_thought(init_thought)
            self.state.set_current([init_thought.id])

        for op in operations:
            logging.info(f"Выполнение операции: {op.__class__.__name__}")
            output_ids = self.execute_operation(op, **kwargs)
            logging.info(f"Операция сгенерировала {len(output_ids)} мыслей.")

        current = self.state.get_current_thoughts()
        if current:
            return current[-1].content
        return ""