# AgentKube-Mini

<p align="center">
  <img src="https://raw.githubusercontent.com/aakriti1318/agentkube-mini/main/agentkube-mini-banner.png" alt="AgentKube-Mini"/>
</p>

A tiny agent orchestration engine. Implements a task DAG, dependency-aware parallel scheduler, and event system for multi-agent pipelines — all in about 400 lines of Python with zero dependencies. The idea is to show how agent orchestration *actually works* under the hood. 

### What is AgentKube-Mini?

**One-Line Truth:** AgentKube-Mini is not the most powerful — but it's one of the simplest and cleanest for task orchestration. It's for people who want to *see and understand* how multi-agent systems work, not hide complexity behind abstractions.

**What it does:**
- **Task DAG**: Define agents and their dependencies. The scheduler figures out parallelism.
- **Parallel execution**: Independent tasks run concurrently. Dependent tasks wait intelligently.
- **Event observability**: Every step emits events (start, complete, fail) for logging and debugging.
- **Shared memory**: All agents write their outputs to a dict that downstream agents can access.

**What it's NOT:**
- Not a framework for tool-calling loops (use LangGraph, LangChain for that).
- Not for human-in-the-loop interrupts or state persistence (LangGraph does this better).
- Not for distributed execution or fault tolerance (add that yourself or use LangGraph's checkpointer).

**When to use AgentKube-Mini:**
- You have specialized agent services (music catalog lookup, invoice retrieval, etc.) and need to **orchestrate them** in a DAG.
- You want to **see the code** — all 400 lines — and understand exactly what's happening.
- You're building a **hybrid system**: wrap your LangGraph sub-agents and let AgentKube-Mini orchestrate the top-level flow (verify → load_memory → route_to_specialist → save_preferences).
- You need **zero dependencies** and want something educational that actually ships.

**When to use LangGraph instead:**
- You need agentic loops (LLM → tools → observe → repeat).
- You need human-in-the-loop with interrupts/resume.
- You need to save/restore state across sessions.
- You're fine with more complexity for more power.

**The hybrid approach (recommended for real systems):**
```
Your LangGraph Agents (do the hard parts: tool-calling, reasoning, memory)
                ↓
        AgentKube-Mini DAG (orchestrate the pipeline)
                ↓
        Event log + shared memory (observe what happened)
```

See `hybrid_orchestration.py` for a full worked example of this pattern.

### Installation

```bash
pip install agentkube-mini
```

### Quick start

### Example usage

Define agents as simple functions, wire them into a DAG, and run:

```python
from agentkube_mini import Agent, TaskGraph, Runtime

# define agents — each is just a name + function
research = Agent("research", lambda topic: f"data about {topic}")
analysis = Agent("analysis", lambda topic, deps: f"analysis of {deps['research']}")
writer   = Agent("writer",   lambda topic, deps: f"article based on {deps['analysis']}")
critic   = Agent("critic",   lambda topic, deps: f"score=9 for {deps['writer']}")

# wire the DAG
graph = TaskGraph()
graph.add(research)
graph.add(analysis, depends=["research"])
graph.add(writer,   depends=["analysis"])
graph.add(critic,   depends=["writer"])

# run it
result = Runtime(graph).run("AI agents")
print(result.outputs)
```

Output:

```
research → data about AI agents
analysis → analysis of data about AI agents
writer   → article based on analysis of data about AI agents
critic   → score=9 for article based on analysis of data about AI agents
```

The scheduler automatically figures out which agents can run in parallel (independent nodes run concurrently via `ThreadPoolExecutor`) and which must wait for dependencies. Events are emitted at each step (`task_started`, `task_completed`, `task_failed`) so you get observability for free.

### Visualization

The task graph can print itself as text or as a Mermaid diagram:

```python
print(graph.visualize())
# research -> analysis
# analysis -> writer
# writer -> critic

print(graph.to_mermaid())
# graph TD
#     research
#     research --> analysis
#     analysis --> writer
#     writer --> critic
```

### Events and shared memory

Every run produces an event log and a shared memory dict that all agents write into:

```python
from agentkube_mini import Runtime, EventBus

bus = EventBus()
bus.subscribe("task_failed", lambda e: print(f"ALERT: {e.task} failed"))

result = Runtime(graph, event_bus=bus).run("AI agents")

# event log
for ev in result.events:
    print(ev.type, ev.task)

# shared memory — every agent's output is accessible
result.memory["research"]  # → "data about AI agents"
```

### Using with existing code

You don't have to rewrite your services. Wrap them with `auto_agent`, which auto-detects the function signature:

```python
from agentkube_mini import auto_agent

# your existing function, unchanged
def my_research_service(topic: str) -> dict:
    return {"topic": topic, "facts": ["f1", "f2"]}

research = auto_agent("research", my_research_service)
```

See `integration_example.py` for a full working example with legacy service classes.

### How it works

The core abstraction is: **agents are nodes, dependencies are edges, the scheduler walks the DAG**. That's the whole idea. If you understand this, you understand the center of every real multi-agent runtime — the rest (distributed workers, retries, message queues, state stores) is extensions on top.

```
Agent graph  →  Scheduler  →  Parallel execution + Events
```

### File structure

```
agent.py               — the Agent dataclass (name + callable)
task_graph.py          — DAG: add nodes, validate, visualize
scheduler.py           — dependency-aware parallel scheduler
runtime.py             — thin wrapper around scheduler
events.py              — event bus with subscribe/emit/history
example.py             — the demo shown above
compat.py              — auto_agent adapter for existing code
smoke_test.py          — tiny correctness test
integration_example.py — legacy service adapter demo
```

### Running tests

```bash
python3 smoke_test.py
```

### License

MIT


