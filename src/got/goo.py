# src/got/goo.py

from typing import List, Dict, Any, Optional
from .operations import Operation

class OperationNode:
    """Узел графа операций. Содержит операцию, зависимости и параметры."""
    def __init__(self, operation: Operation, dependencies: List['OperationNode'] = None, params: Dict[str, Any] = None):
        self.operation = operation
        self.dependencies = dependencies or []
        self.params = params or {}

class GraphOfOperations:
    """Граф операций (GoO) — последовательность узлов с зависимостями."""
    def __init__(self, nodes: List[OperationNode] = None):
        self.nodes = nodes or []