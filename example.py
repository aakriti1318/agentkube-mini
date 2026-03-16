from __future__ import annotations

import json

from agentkube_mini import Agent, Runtime, TaskGraph


def build_graph() -> TaskGraph:
    """Blueprint §8 & §16 — simple string-in / string-out agents."""
    graph = TaskGraph()

    research = Agent("research", lambda topic: f"data about {topic}")
    analysis = Agent("analysis", lambda topic, deps: f"analysis of {deps['research']}")
    writer   = Agent("writer",   lambda topic, deps: f"article based on {deps['analysis']}")
    critic   = Agent("critic",   lambda topic, deps: f"score=9 for {deps['writer']}")

    graph.add(research)
    graph.add(analysis, depends=["research"])
    graph.add(writer,   depends=["analysis"])
    graph.add(critic,   depends=["writer"])

    return graph


def main() -> None:
    graph = build_graph()
    runtime = Runtime(graph, max_workers=4)
    result = runtime.run(initial_input="AI agents")

    # Blueprint §9 — simple text visualization
    print("=== Graph ===")
    print(graph.visualize())

    # Blueprint §9 — mermaid visualization
    print("\n=== DAG (Mermaid) ===")
    print(graph.to_mermaid())

    # Outputs
    print("\n=== Outputs ===")
    for name, output in result.outputs.items():
        print(f"  {name} → {output}")

    # Shared memory (blueprint §12)
    print("\n=== Shared Memory ===")
    for name, value in result.memory.items():
        print(f"  memory[\"{name}\"] = {value}")

    # Events (blueprint §11)
    print("\n=== Events ===")
    for event in result.events:
        print(f"  {event.type:14} | {event.task}")


if __name__ == "__main__":
    main()
