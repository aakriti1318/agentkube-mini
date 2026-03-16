from __future__ import annotations

from typing import Any

from agentkube_mini.events import EventBus
from agentkube_mini.scheduler import RunResult, Scheduler
from agentkube_mini.task_graph import TaskGraph


class Runtime:
    def __init__(self, graph: TaskGraph, event_bus: EventBus | None = None, max_workers: int = 4) -> None:
        self.scheduler = Scheduler(graph=graph, event_bus=event_bus, max_workers=max_workers)

    def run(self, initial_input: Any = None) -> RunResult:
        return self.scheduler.run(initial_input=initial_input)
