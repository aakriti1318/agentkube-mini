from agentkube_mini import Runtime
from example import build_graph


def main() -> None:
    graph = build_graph()
    result = Runtime(graph).run("AI agents")

    # All four agents ran
    assert set(result.outputs.keys()) == {"research", "analysis", "writer", "critic"}

    # Outputs are simple strings (blueprint §8)
    assert "data about AI agents" in result.outputs["research"]
    assert "analysis of" in result.outputs["analysis"]
    assert "article based on" in result.outputs["writer"]
    assert "score=9" in result.outputs["critic"]

    # Shared memory populated (blueprint §12)
    assert set(result.memory.keys()) == {"research", "analysis", "writer", "critic"}

    # Events emitted (blueprint §11)
    event_types = {event.type for event in result.events}
    assert "task_started" in event_types
    assert "task_completed" in event_types

    print("smoke_test: ok")


if __name__ == "__main__":
    main()
