# src/got/__init__.py

from .thought import Thought
from .operations import Operation, Generate, Score, Refine, Merge, Select
from .controller import GoTController
from .prompter import Prompter
from .parser import Parser
from .roles import ROLES, get_role_description, get_random_role
from .goo import OperationNode, GraphOfOperations