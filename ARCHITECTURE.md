# AgentKube-Mini: Multi-Agent System Comparison & Integration Guide

This directory contains two complementary approaches to building multi-agent systems:

1. **`multi_agent.ipynb`** — Advanced LangGraph approach with full complexity
2. **`notebook_adapter.py`** — Comprehensive comparison guide + hybrid approach
3. **`hybrid_orchestration.py`** — Working example of Agentkube-Mini + LangGraph integration

## Quick Start

### View the Comparison
```bash
python3 notebook_adapter.py
```

### Run the Hybrid Example
```bash
python3 hybrid_orchestration.py
```

This demonstrates:
- Customer verification
- Long-term memory (preferences)
- Routing to specialists
- Preference saving

---

## Architecture Overview

### The LangGraph Notebook (`multi_agent.ipynb`)

**Best for:** Complex LLM reasoning, tool-calling loops, human interrupts

```
┌─────────────────────────────────────────┐
│   Customer Support Multi-Agent System   │
│              (LangGraph)                │
├─────────────────────────────────────────┤
│ verify_info                             │
│  ↓                                      │
│ load_memory                             │
│  ↓                                      │
│ supervisor (routes to specialists)     │
│  ├─→ music_subagent (ReAct loop)        │
│  └─→ invoice_subagent (ReAct loop)      │
│  ↓                                      │
│ create_memory                           │
│  ↓                                      │
│ Return response                         │
└─────────────────────────────────────────┘
```

**Key Features:**
- Thread-aware state management
- Native human-in-the-loop interrupts
- LangSmith integration for evals
- Tool-calling loops (ReAct) inside sub-agents
- Sophisticated supervisor pattern

**Complexity:** 300+ lines (before evaluations)

---

### The Hybrid Approach (`hybrid_orchestration.py`)

**Best for:** Simple top-level orchestration + complex sub-agents

```
┌──────────────────────────────────────────────┐
│    AGENTKUBE-MINI ORCHESTRATION LAYER       │
│           (Simple, Deterministic DAG)        │
├──────────────────────────────────────────────┤
│ Agent: verify           Agent: load_memory   │
│     ↓                           ↓             │
│     └─────────────┬─────────────┘             │
│                   ↓                           │
│     Agent: route (invokes LangGraph)         │
│                   ↓                           │
│     Agent: save                              │
│                   ↓                           │
│           Return response                    │
└──────────────────────────────────────────────┘
        │                          │
        ├──→ music_subagent ←──────┤
        │    (LangGraph ReAct)     │
        │                          │
        └──→ invoice_subagent ←────┘
             (LangGraph ReAct)
```

**Key Features:**
- Agentkube-Mini handles orchestration (50-100 lines)
- LangGraph sub-agents handle specialized reasoning
- Automatic parallelization at orchestration level
- Clear separation of concerns
- Easy to test and debug

**Complexity:** ~150 lines total (orchestration + simulation)

---

## Comparison Table

| Feature | LangGraph | Agentkube-Mini | Hybrid |
|---------|-----------|----------------|--------|
| **Orchestration complexity** | High (300+) | Low (50) | Low (50+LangGraph) |
| **Parallelization** | Sequential | Parallel | Parallel |
| **Tool-calling loops** | ✓ Built-in | ✗ Manual | ✓ In LangGraph |
| **Human interrupts** | ✓ Native | ✗ Manual | ◐ Simplified |
| **Testing** | Hard (stateful) | Easy (pure) | Easy + Powerful |
| **Debuggability** | Hard (nested) | Easy (linear) | Easy (linear) |
| **Learning curve** | Steep | Gentle | Moderate |
| **Production ready** | ✓ | ✓ | ✓ (recommended) |

---

## When to Use Each Approach

### ✓ Use LangGraph Notebook When:
- Building a single sophisticated agent
- Complex reasoning and tool-calling loops needed
- Want native human-in-the-loop interrupts
- Need detailed LangSmith tracing
- Each conversation is truly independent

### ✓ Use Agentkube-Mini Only When:
- Entire system is deterministic orchestration (no loops)
- Multiple independent tasks in parallel
- No LLM reasoning needed (or keep it hidden)
- Prioritize simplicity and performance
- Clear DAG structure with no cycles

### ✓ Use Hybrid (RECOMMENDED) When:
- Clear top-level orchestration flow
- Some steps require LLM reasoning (keep in LangGraph)
- Want simplicity at top + power inside specialists
- Example: verify → load_prefs → [route to expert] → save
- Most production systems fit here

---

## Code Examples

### Agentkube-Mini: Pure Orchestration
```python
from agentkube_mini import Agent, TaskGraph, Runtime

def step1(data):
    # Simple logic
    return {"result": process(data)}

def step2(data):
    # Depends on step1
    return {"final": combine(data["result"])}

graph = TaskGraph()
graph.add(Agent("step1", step1))
graph.add(Agent("step2", step2), depends_on=["step1"])

result = Runtime(graph).run(initial_input)
print(result.memory)  # All intermediate results
```

### LangGraph: Complex Reasoning
```python
from langgraph.prebuilt import create_react_agent

music_agent = create_react_agent(
    llm,
    tools=[get_albums, get_tracks, get_songs_by_genre],
    name="music_agent",
    prompt=music_prompt
)

# LLM automatically decides which tools to call
# Loops until it has enough info to answer
result = music_agent.invoke(
    {"messages": [HumanMessage(content="What albums by U2?")]},
    config={"configurable": {"thread_id": thread_id}}
)
```

### Hybrid: Best of Both
```python
def route_to_specialist(input_data):
    question = input_data["question"]
    
    if "music" in question.lower():
        # Invoke LangGraph sub-agent
        result = music_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
    elif "invoice" in question.lower():
        # Invoke different LangGraph sub-agent
        result = invoice_agent.invoke(...)
    
    return {"response": result["messages"][-1].content}

# Wrap in Agentkube-Mini
graph.add(Agent("route", route_to_specialist), depends_on=["verify", "load_memory"])
result = Runtime(graph).run(initial_input)
```

---

## File Guide

### `notebook_adapter.py`
Comprehensive comparison document showing:
1. **Comparison Table** — Feature-by-feature breakdown
2. **LangGraph Flow** — Detailed walkthrough with code + diagrams
3. **Agentkube-Mini Flow** — Detailed walkthrough with code + diagrams
4. **Hybrid Approach** — Architecture combining both
5. **Quick Reference** — Patterns and decision tree

Run it to understand the tradeoffs:
```bash
python3 notebook_adapter.py | less
```

### `hybrid_orchestration.py`
Working example implementing:
- Customer verification with ID extraction
- Long-term memory storage
- Routing to specialists
- Preference saving
- Human-in-the-loop verification loop

Run it to see hybrid in action:
```bash
python3 hybrid_orchestration.py
```

### `multi_agent.ipynb`
Full LangGraph implementation with:
- 2 ReAct sub-agents (music + invoice)
- Supervisor pattern
- Human interrupts for verification
- Long-term memory with UserProfile
- Swarm agent variant
- Comprehensive evaluations (LangSmith)

Use Jupyter to explore interactively.

---

## Migration Path: From Notebook to Production

```
Step 1: Start with Notebook (LangGraph)
        ↓ Validate ideas, test LLM behavior

Step 2: Identify Top-Level Flow
        ↓ What's the step sequence? (verify → route → save)

Step 3: Extract Step Functions
        ↓ Create Python functions for each step

Step 4: Build Agentkube-Mini DAG
        ↓ Define TaskGraph with dependencies

Step 5: Test & Deploy
        ↓ Easier to debug and parallelize
```

---

## Performance Characteristics

### Orchestration Level (Agentkube-Mini)
- **Latency:** ~100-500ms (pure Python, no LLM)
- **Parallelization:** ThreadPoolExecutor (CPU-bound work)
- **Throughput:** Limited by slowest task

### Sub-Agent Level (LangGraph)
- **Latency:** 1-30s per agent (depends on LLM calls + tool execution)
- **Tool Calls:** Dynamic (LLM decides sequence)
- **Parallelization:** Sequential within agent (ReAct loop)

### Hybrid System
- **Total Latency:** Orchestration + max(sub-agent latencies)
- **Throughput:** Independent orchestration + sub-agents
- **Bottleneck:** LLM latency (tool calling) in sub-agents

---

## Extending the System

### Add a New Specialist Agent
```python
# Create LangGraph sub-agent
complaint_agent = create_react_agent(
    llm,
    tools=[submit_ticket, track_complaint],
    name="complaint_agent"
)

# Update routing in Agentkube-Mini
def route_to_specialist(input_data):
    if "complaint" in input_data["question"].lower():
        return call_langraph_agent(complaint_agent, input_data)
    # ... existing routes ...
```

### Add Memory Persistence
```python
# Replace MemoryStore with database
class DatabaseStore:
    def get(self, key, subkey):
        return db.query(...).first()
    
    def put(self, key, subkey, value):
        db.save(key, subkey, value)
```

### Add Observability
```python
# Subscribe to Agentkube-Mini events
event_bus.subscribe("task_started", lambda e: log(f"Task: {e.name}"))
event_bus.subscribe("task_completed", lambda e: log(f"Result: {e.data}"))

# Invoke with event bus
runtime = Runtime(graph, event_bus=event_bus)
```

---

## Recommended Architecture for Production

```
┌─────────────────────────────────────────────────────┐
│         REST API / Chat Interface                   │
├─────────────────────────────────────────────────────┤
│     Agentkube-Mini Orchestration Layer              │
│  (verify → load_prefs → route → save) [50-100 LOC] │
├─────────────────────────────────────────────────────┤
│        LangGraph Sub-Agent Layer                    │
│  • Music Specialist (ReAct, 200-300 LOC)            │
│  • Invoice Specialist (ReAct, 200-300 LOC)          │
│  • Complaint Specialist (ReAct, 200-300 LOC)        │
├─────────────────────────────────────────────────────┤
│         Shared Services                             │
│  • Database (Chinook, customers, preferences)       │
│  • Cache (Redis for preferences)                    │
│  • LLM (OpenAI, Llama, etc.)                        │
│  • LangSmith (tracing, evals)                       │
└─────────────────────────────────────────────────────┘
```

**Benefits:**
- Simple top-level orchestration (easy to debug)
- Powerful specialists (sophisticated reasoning)
- Clear separation of concerns
- Easy to test each layer independently
- Easy to add new specialists
- Scalable (parallelize orchestration + sub-agents)

---

## Resources

- **Agentkube-Mini Docs:** See `README.md` in repo root
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangChain Academy:** https://academy.langchain.com/courses/intro-to-langgraph
- **Multi-Agent Swarms:** https://www.youtube.com/watch?v=JeyDrn1dSUQ

---

## Questions?

- **How do I parallelize LangGraph?** → Use Agentkube-Mini orchestration layer
- **How do I simplify my LangGraph?** → Extract pure functions, wrap in Agentkube-Mini
- **Can I use this in production?** → Yes, but migrate database from MemoryStore
- **What's the learning curve?** → Gentle for Agentkube, steep for LangGraph, moderate for hybrid
- **Which should I choose?** → Start with hybrid; scale based on needs
