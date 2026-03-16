"""
AgentKube-Mini: A tiny agent orchestration engine.

Implements a task DAG, dependency-aware parallel scheduler, and event system
for multi-agent pipelines in ~400 lines of Python with zero dependencies.
"""

from agentkube_mini.agent import Agent
from agentkube_mini.compat import auto_agent
from agentkube_mini.events import Event, EventBus
from agentkube_mini.runtime import Runtime
from agentkube_mini.scheduler import RunResult, Scheduler
from agentkube_mini.task_graph import TaskGraph

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "TaskGraph",
    "Scheduler",
    "RunResult",
    "Runtime",
    "EventBus",
    "Event",
    "auto_agent",
]
