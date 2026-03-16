from __future__ import annotations

from dataclasses import dataclass, field

from agentkube_mini.agent import Agent


@dataclass
class TaskGraph:
    nodes: dict[str, Agent] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)

    def add(
        self,
        agent: Agent,
        depends: list[str] | None = None,
        depends_on: list[str] | None = None,
    ) -> None:
        deps = depends or depends_on or []
        if agent.name in self.nodes:
            raise ValueError(f"Task already exists: {agent.name}")
        self.nodes[agent.name] = agent
        self.edges[agent.name] = deps

    def validate(self) -> None:
        for task, deps in self.edges.items():
            for dep in deps:
                if dep not in self.nodes:
                    raise ValueError(f"{task} depends on unknown task: {dep}")

    def as_edge_list(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for task, deps in self.edges.items():
            for dep in deps:
                pairs.append((dep, task))
        return pairs

    def visualize(self) -> str:
        """Simple text visualization (blueprint §9)."""
        lines: list[str] = []
        for dep, task in self.as_edge_list():
            lines.append(f"{dep} -> {task}")
        return "\n".join(lines) if lines else "(no edges)"

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for node in self.nodes:
            if not self.edges[node]:
                lines.append(f"    {node}")
        for dep, task in self.as_edge_list():
            lines.append(f"    {dep} --> {task}")
        return "\n".join(lines)
