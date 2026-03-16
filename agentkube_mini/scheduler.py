from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from typing import Any

from agentkube_mini.events import EventBus
from agentkube_mini.task_graph import TaskGraph


@dataclass
class RunResult:
    outputs: dict[str, Any]
    memory: dict[str, Any]     # shared state across agents (blueprint §12)
    events: list[Any]


class Scheduler:
    def __init__(self, graph: TaskGraph, event_bus: EventBus | None = None, max_workers: int = 4) -> None:
        self.graph = graph
        self.event_bus = event_bus or EventBus()
        self.max_workers = max_workers

    def run(self, initial_input: Any = None) -> RunResult:
        self.graph.validate()
        outputs: dict[str, Any] = {}
        memory: dict[str, Any] = {}          # shared memory (blueprint §12)
        state_lock = Lock()

        pending = set(self.graph.nodes)
        running: dict[Future[Any], str] = {}

        def deps_ready(task: str) -> bool:
            return all(dep in outputs for dep in self.graph.edges[task])

        def execute(task_name: str) -> Any:
            dep_outputs = {dep: outputs[dep] for dep in self.graph.edges[task_name]}
            self.event_bus.emit("task_started", task_name, {"depends_on": list(dep_outputs)})
            agent = self.graph.nodes[task_name]
            # Agent.run(input, dependency_outputs) — simple signature
            result = agent.run(initial_input, dep_outputs)
            self.event_bus.emit("task_completed", task_name, result)
            with state_lock:
                memory[task_name] = result    # shared state (blueprint §12)
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while pending or running:
                with state_lock:
                    ready = [task for task in pending if deps_ready(task)]
                    for task in ready:
                        future = pool.submit(execute, task)
                        running[future] = task
                        pending.remove(task)

                if not running:
                    unresolved = ", ".join(sorted(pending))
                    raise RuntimeError(f"No runnable tasks. Cycle likely present: {unresolved}")

                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task_name = running.pop(future)
                    try:
                        output = future.result()
                    except Exception as exc:  # noqa: BLE001
                        self.event_bus.emit("task_failed", task_name, str(exc))
                        raise RuntimeError(f"Task failed: {task_name}") from exc
                    with state_lock:
                        outputs[task_name] = output

        return RunResult(outputs=outputs, memory=memory, events=self.event_bus.history)
