# src/got/controller.py

"""
Контроллер выполнения графа операций (Graph of Operations).
"""
import logging
from typing import List
from .thought import Thought
from .goo import GraphOfOperations

class GoTController:
    def __init__(self, llm_client, prompter=None, parser=None):
        self.llm_client = llm_client
        self.prompter = prompter
        self.parser = parser

    def run_goo(self, goo: GraphOfOperations) -> List[Thought]:
        """
        Выполняет граф операций (GoO) в топологическом порядке.
        Возвращает список мыслей из последнего узла графа.
        """
        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            order.append(node)

        for node in goo.nodes:
            dfs(node)

        node_outputs = {}
        for node in order:
            inputs = []
            for dep in node.dependencies:
                inputs.extend(node_outputs.get(dep, []))
            outputs = node.operation.execute(
                inputs,
                self.prompter,
                self.parser,
                self.llm_client,
                **node.params
            )
            node_outputs[node] = outputs

        if goo.nodes:
            return node_outputs.get(goo.nodes[-1], [])
        return []