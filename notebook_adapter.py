"""
COMPARISON: LangGraph Multi-Agent vs. Agentkube-Mini Orchestration

This file provides a detailed side-by-side comparison of:
1. The LangGraph approach (from multi_agent.ipynb)
2. The Agentkube-Mini hybrid approach (practical implementation)

KEY TAKEAWAYS:
- LangGraph: Excels at complex reasoning, tool-calling loops, human-in-the-loop interrupts
- Agentkube-Mini: Excels at deterministic DAG orchestration, simplicity, observability
- Hybrid: Combine them - use Agentkube-Mini at the top level, LangGraph for specialized agents
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json


# ============================================================================
# ARCHITECTURE COMPARISON TABLE
# ============================================================================

COMPARISON_TABLE = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LANGGRAPH vs AGENTKUBE-MINI COMPARISON                     ║
╠════════════════════════════════╦════════════════════════════════╦══════════════╣
║ FEATURE                        ║ LANGGRAPH                      ║ AGENTKUBE    ║
╠════════════════════════════════╬════════════════════════════════╬══════════════╣
║ Agent Definition               ║ LLM-based with tool binding    ║ Simple func  ║
║ State Management               ║ Thread-aware checkpoints       ║ Dict-based   ║
║ Graph Type                     ║ Cyclic (ReAct loops)           ║ Acyclic DAG  ║
║ Human Interruption             ║ Native interrupt/resume        ║ Manual loop  ║
║ Tool Calling                   ║ Built-in (bind_tools)          ║ N/A          ║
║ Memory (Long-term)             ║ InMemoryStore + namespace      ║ Dict + save  ║
║ Supervisor Pattern             ║ create_supervisor()            ║ Manual route ║
║ Parallelization                ║ Sequential within thread       ║ ThreadPool   ║
║ Observability                  ║ LangSmith integration          ║ EventBus     ║
║ Learning Curve                 ║ Steep (many abstractions)      ║ Gentle       ║
║ Code Complexity (basic flow)   ║ 300+ lines                     ║ 50-100 lines ║
╚════════════════════════════════╩════════════════════════════════╩══════════════╝
"""


# ============================================================================
# PART 1: LANGGRAPH APPROACH (from multi_agent.ipynb)
# ============================================================================

LANGGRAPH_FLOW = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LANGGRAPH FLOW (Notebook)                          │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Define State (TypedDict)
────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps

→ State flows through ALL nodes, updated after each step
→ Managed by LangGraph's `add_messages` reducer


STEP 2: Define Tools (LangChain @tool decorator)
────────────────────────────────────────────────────────────────────────────
@tool
def get_albums_by_artist(artist: str):
    '''Get albums by an artist.'''
    return db.run(f"SELECT ... FROM Album WHERE Artist = {artist}")

@tool
def get_invoices_by_customer(customer_id: str):
    '''Get invoices for a customer.'''
    return db.run(f"SELECT * FROM Invoice WHERE CustomerId = {customer_id}")

→ Tools are bound to LLM: llm_with_tools = llm.bind_tools([get_albums, ...])
→ LLM decides when/how to call them (part of ReAct loop)


STEP 3: Define Nodes (Functions + LLM)
────────────────────────────────────────────────────────────────────────────
# Node 1: LLM Reasoning
def music_assistant(state: State, config: RunnableConfig):
    response = llm_with_music_tools.invoke(
        [SystemMessage(prompt)] + state["messages"]
    )
    return {"messages": [response]}

# Node 2: Tool Execution
music_tool_node = ToolNode(music_tools)

# Node 3: Customer Verification
def verify_info(state: State, config: RunnableConfig):
    if not state.get("customer_id"):
        # Parse input for customer ID
        parsed = structured_llm.invoke([...])
        if found_id:
            return {"customer_id": found_id}
        else:
            return {"messages": [prompt_for_id_response]}

→ Each node receives FULL state, returns PARTIAL updates
→ LangGraph merges updates before passing to next node


STEP 4: Define Edges (Conditional Routing)
────────────────────────────────────────────────────────────────────────────
def should_continue(state: State, config: RunnableConfig):
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return "end"  # LLM produced final answer
    else:
        return "continue"  # Call tools

graph.add_conditional_edges(
    "music_assistant",
    should_continue,
    {
        "continue": "music_tool_node",
        "end": END,
    }
)

→ Conditional edges enable dynamic routing (branching based on state)
→ Cycles allowed (e.g., tool → reasoning → tool loop)


STEP 5: Add Human-in-the-Loop (Interrupts)
────────────────────────────────────────────────────────────────────────────
def human_input(state: State, config: RunnableConfig):
    user_input = interrupt("Please provide input.")
    return {"messages": [user_input]}

# Graph pauses at human_input node
# User/caller can inspect state and resume with new data
# Usage:
result = graph.invoke(initial_input, config)  # Pauses at interrupt
result = graph.invoke(Command(resume=new_input), config)  # Resumes

→ Native interrupt/resume mechanism (no manual loop needed)
→ Thread ID ensures state is preserved across interrupts


STEP 6: Add Long-Term Memory
────────────────────────────────────────────────────────────────────────────
def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    namespace = ("memory_profile", state["customer_id"])
    existing = store.get(namespace, "user_memory")
    formatted = format_user_memory(existing.value)
    return {"loaded_memory": formatted}

def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    updated = llm.with_structured_output(UserProfile).invoke([...])
    store.put(namespace, "user_memory", {"memory": updated})

→ InMemoryStore for persistence between conversations
→ Structured output (Pydantic) for type safety
→ Automatic state saving/loading per customer


STEP 7: Compile & Run
────────────────────────────────────────────────────────────────────────────
graph = StateGraph(State)
# ... add nodes and edges ...
compiled = graph.compile(
    checkpointer=MemorySaver(),  # Short-term (thread) memory
    store=InMemoryStore()         # Long-term (customer) memory
)

result = compiled.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"configurable": {"thread_id": uuid.uuid4()}}
)

→ Checkpointer manages thread-level state persistence
→ Store manages cross-session customer data
→ Thread ID isolates conversations


FLOW DIAGRAM:
┌──────────────┐
│   START      │
└───────┬──────┘
        │
        ▼
┌──────────────────────┐
│  verify_info         │◄──────────────┐
│  (extract customer)  │               │
└─────┬────────────────┘               │
      │ (if verified)                  │
      │                                │
      ▼                           ┌────────────┐
┌─────────────────┐              │ human_input│
│  load_memory    │              │ (interrupt)│
│  (user prefs)   │              └────┬───────┘
└────────┬────────┘                   │
         │                            │
         ▼                            │
┌──────────────────────────────┐      │
│     supervisor               │      │
│  (routes to sub-agents)      │      │
└────┬─────────────────────────┘      │
     │                                │
     ├─────────────────────┬──────────┘
     │                     │
     ▼                     ▼
┌──────────────┐    ┌─────────────────┐
│ music_agent  │    │ invoice_agent   │
│  (ReAct loop)│    │  (ReAct loop)   │
└─────┬────────┘    └────┬────────────┘
      │                  │
      └──────┬───────────┘
             ▼
┌──────────────────────┐
│  create_memory       │
│  (save preferences)  │
└─────────┬────────────┘
          │
          ▼
      ┌───────┐
      │ END   │
      └───────┘

ADVANTAGES:
✓ Powerful tool-calling loops (ReAct)
✓ Native human interruption (pause/resume)
✓ Type-safe state management
✓ Thread-aware multi-conversation isolation
✓ LangSmith integration for evals
✓ Supervisor pattern (create_supervisor)
✓ Fine-grained control over routing

DISADVANTAGES:
✗ High complexity (many abstractions)
✗ Steep learning curve
✗ Harder to debug (nested graph structures)
✗ Overkill for simple deterministic flows
✗ Must understand: edges, reducers, managed state, interrupts
✗ Slower for independent parallel tasks (sequential by nature)
"""


# ============================================================================
# PART 2: AGENTKUBE-MINI APPROACH (Practical)
# ============================================================================

AGENTKUBE_FLOW = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENTKUBE-MINI FLOW (Hybrid)                         │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Define Agents as Simple Functions
────────────────────────────────────────────────────────────────────────────
def verify_customer_id(input_data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Extract and validate customer ID from input.\"\"\"
    question = input_data.get("question", "")
    customer_id = extract_customer_id(question)
    return {
        "customer_id": customer_id,
        "question": question,
        "verified": bool(customer_id),
    }

def load_user_memory(input_data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Load user preferences from long-term memory.\"\"\"
    customer_id = input_data.get("customer_id")
    preferences = ""
    if customer_id:
        namespace = ("memory_profile", customer_id)
        stored = memory_store.get(namespace, "user_memory")
        preferences = format_preferences(stored)
    return {"preferences": preferences}

def route_to_specialist(input_data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Route query to music or invoice specialist.\"\"\"
    question = input_data.get("question", "")
    # Could invoke LangGraph sub-agents here
    if "music" in question.lower():
        response = music_subagent.invoke(...)  # Call LangGraph agent
    elif "invoice" in question.lower():
        response = invoice_subagent.invoke(...)
    return {"response": response}

def save_updated_preferences(input_data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Analyze conversation and save new preferences.\"\"\"
    customer_id = input_data.get("customer_id")
    question = input_data.get("question", "")
    response = input_data.get("response", "")
    
    new_prefs = extract_music_preferences(question, response)
    if new_prefs and customer_id:
        namespace = ("memory_profile", customer_id)
        memory_store.put(namespace, "user_memory", {"music_preferences": new_prefs})
        return {"saved": True, "new_preferences": new_prefs}
    return {"saved": False}

→ Each agent is a pure function: input → compute → output
→ No LLM reasoning inside agents (keep that in LangGraph sub-agents)
→ Agents are easily testable and deterministic


STEP 2: Build DAG (TaskGraph)
────────────────────────────────────────────────────────────────────────────
graph = TaskGraph()

# Add agents with explicit dependencies
graph.add(Agent("verify", verify_customer_id))
graph.add(Agent("load_memory", load_user_memory), depends_on=["verify"])
graph.add(Agent("route", route_to_specialist), depends_on=["verify", "load_memory"])
graph.add(Agent("save", save_updated_preferences), depends_on=["verify", "route"])

# Validate the DAG
graph.validate()

→ Dependencies are explicit (no implicit state flow)
→ Acyclic (no loops - keep loops inside LangGraph)
→ All independent tasks run in parallel


STEP 3: Run with Runtime
────────────────────────────────────────────────────────────────────────────
runtime = Runtime(graph)
result = runtime.run({"question": "My ID is 123. What albums by U2?"})

# Access outputs
print(result.outputs)    # Final results from each agent
print(result.memory)     # Shared state across all agents
print(result.events)     # Task start/complete events

→ Single call, automatic parallelization
→ All intermediate results stored in result.memory
→ Events provide observability


STEP 4: Handle Human-in-the-Loop Manually
────────────────────────────────────────────────────────────────────────────
def run_conversation(initial_question: str) -> str:
    attempt = 0
    question = initial_question
    
    while attempt < 3:
        # Run orchestration graph
        graph = build_orchestration_graph()
        runtime = Runtime(graph)
        result = runtime.run({"question": question})
        
        # Check if verification succeeded
        verify_result = result.memory.get("verify", {})
        if verify_result.get("verified"):
            return result.memory.get("route", {}).get("response")
        
        # Verification failed, ask user and retry
        print("Please provide your Customer ID, email, or phone number.")
        user_input = input("You: ").strip()
        question = f"{user_input}. {initial_question}"
        attempt += 1
    
    return "Maximum attempts exceeded."

→ Manual loop (no native interrupt mechanism)
→ Each attempt creates a fresh graph and runs it
→ Simpler but less elegant than LangGraph interrupts


STEP 5: Long-Term Memory (Simple Dict + Save)
────────────────────────────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self):
        self.data = {}
    
    def get(self, key: tuple, subkey: str) -> Optional[Dict]:
        if key in self.data and subkey in self.data[key]:
            return self.data[key][subkey]
        return None
    
    def put(self, key: tuple, subkey: str, value: Dict) -> None:
        if key not in self.data:
            self.data[key] = {}
        self.data[key][subkey] = value

# Usage in save_updated_preferences:
namespace = ("memory_profile", customer_id)
memory_store.put(namespace, "user_memory", {"music_preferences": new_prefs})

→ Simple dict-based store (replace with Redis/DB for production)
→ No automatic threading (manage it yourself in outer loop)
→ Manual namespace management (but explicit)


STEP 6: Observability via EventBus
────────────────────────────────────────────────────────────────────────────
from agentkube_mini import EventBus

event_bus = EventBus()

# Subscribe to events
event_bus.subscribe("task_started", lambda e: print(f"Task started: {e.task_name}"))
event_bus.subscribe("task_completed", lambda e: print(f"Task done: {e.data}"))

# Runtime automatically emits events
runtime = Runtime(graph, event_bus=event_bus)
result = runtime.run({"question": "..."})

→ EventBus provides lightweight observability
→ No LangSmith integration (use custom logging if needed)


FLOW DIAGRAM:
┌──────────────┐
│   START      │
└───────┬──────┘
        │
        ▼
  ┌────────────────┐
  │ run_conversion │  ◄─────────────────┐
  │  (manual loop) │                    │
  └────┬───────────┘                    │
       │                                │
       ▼                                │
  ┌────────────────────────────┐        │
  │ BUILD ORCHESTRATION GRAPH   │        │
  │  (fresh TaskGraph)          │        │
  └────────┬───────────────────┘        │
           │                            │
           ▼                            │
  ┌─────────────────┐                   │
  │  Agent: verify  │                   │
  │  (extract ID)   │                   │
  └────────┬────────┘                   │
           │                            │
        ┌──┴──┐                         │
        │     │                         │
    YES │     │ NO                      │
        │     └──────────────────────────┘
        │
        ▼
  ┌────────────────┐
  │ Agent: load_   │
  │ memory (prefs) │
  └────────┬───────┘
           │
           ▼
  ┌────────────────────────┐
  │ Agent: route_to_       │
  │ specialist             │
  │ (invoke LangGraph sub-├─→ music_subagent (LangGraph ReAct)
  │ agents or simulate)    │
  │                        ├─→ invoice_subagent (LangGraph ReAct)
  └────────┬───────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ Agent: save_updated_    │
  │ preferences             │
  └────────┬────────────────┘
           │
           ▼
      ┌─────────────┐
      │ RETURN      │
      │ response    │
      └─────────────┘

ADVANTAGES:
✓ Simple, easy to understand
✓ Pure functions (easy to test)
✓ Explicit dependencies (clear data flow)
✓ Automatic parallelization
✓ Shallow learning curve
✓ Ideal for deterministic orchestration
✓ Can wrap LangGraph sub-agents

DISADVANTAGES:
✗ No native tool-calling (keep in LangGraph)
✗ Manual human-in-the-loop (must write own loop)
✗ No cyclic graphs (no ReAct loops here)
✗ No thread-aware interrupts (manual state mgmt)
✗ No LangSmith integration (custom logging)
✗ Less sophisticated supervisor (manual routing logic)
"""


# ============================================================================
# PART 3: HYBRID APPROACH (RECOMMENDED)
# ============================================================================

HYBRID_APPROACH = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID APPROACH (BEST OF BOTH)                      │
└─────────────────────────────────────────────────────────────────────────────┘

ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────────┐
│                     AGENTKUBE-MINI ORCHESTRATION LAYER                      │
│                         (Simple, Deterministic DAG)                         │
│                                                                             │
│  ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐              │
│  │ verify   │───▶│ load    │───▶│ route   │───▶│ save     │              │
│  │ customer │    │ memory  │    │ query   │    │ prefs    │              │
│  └──────────┘    └─────────┘    └────┬────┘    └──────────┘              │
│                                       │                                    │
│                           ┌───────────┴───────────┐                        │
│                           │                       │                        │
│                 ┌─────────▼─────────┐   ┌────────▼─────────┐             │
│                 │ Calls LangGraph   │   │ Calls LangGraph  │             │
│                 │ Music Sub-Agent   │   │ Invoice Sub-Agent│             │
│                 └───────────────────┘   └──────────────────┘             │
│                                                                             │
│  BENEFITS:                                                                  │
│  • Orchestration is simple (50-100 lines)                                  │
│  • Each "route" call can invoke sophisticated LangGraph agents             │
│  • Parallel execution at orchestration level                               │
│  • Clear separation of concerns                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (invoke)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH SUB-AGENTS (LLM-Based)                       │
│                    (Complex Reasoning, Tool-Calling Loops)                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  MUSIC SUB-AGENT (create_react_agent)                              │  │
│  │                                                                      │  │
│  │  State: {messages, customer_id, loaded_memory}                     │  │
│  │  Tools: get_albums, get_tracks, get_songs_by_genre, check_songs   │  │
│  │  Flow: LLM → (decision) → Tool → LLM → ... → Final Answer         │  │
│  │                                                                      │  │
│  │  Returns: Final response to customer about music catalog           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  INVOICE SUB-AGENT (create_react_agent)                            │  │
│  │                                                                      │  │
│  │  State: {messages, customer_id, loaded_memory}                     │  │
│  │  Tools: get_invoices, get_by_price, get_employee_info             │  │
│  │  Flow: LLM → (decision) → Tool → LLM → ... → Final Answer         │  │
│  │                                                                      │  │
│  │  Returns: Final response to customer about invoices                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  BENEFITS:                                                                  │
│  • Each sub-agent uses full ReAct loop (tool-calling, reasoning)           │
│  • LLM can dynamically decide tool sequences                               │
│  • Type-safe state management per agent                                    │
│  • Can use LangSmith for detailed eval/tracing                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


WHEN TO USE EACH:

┌────────────────────┬──────────────────────────────────────────────────────┐
│ Use LANGGRAPH if:  │ • Queries require complex reasoning & tool-calling  │
│                    │ • You need human interruption/resume in middle      │
│                    │ • You want LangSmith integration                    │
│                    │ • You handle cyclic/ReAct patterns                  │
│                    │ • Each conversation is independent                 │
│                    │ • You need sophisticated supervisor pattern        │
│                    │ • You're building a single-agent OR multi-agent    │
│                    │   where routing/orchestration is simple           │
└────────────────────┴──────────────────────────────────────────────────────┘

┌────────────────────┬──────────────────────────────────────────────────────┐
│ Use AGENTKUBE-MINI │ • Orchestration is deterministic (no loops)         │
│ if:                │ • You have multiple independent tasks               │
│                    │ • Tasks are mostly IO or simple logic               │
│                    │ • You want parallelization                          │
│                    │ • You prefer simplicity & testability               │
│                    │ • You don't need thread-aware interrupts            │
│                    │ • Top-level coordination is straightforward         │
└────────────────────┴──────────────────────────────────────────────────────┘

┌────────────────────┬──────────────────────────────────────────────────────┐
│ Use HYBRID:        │ • Your system has a clear top-level flow (DAG)      │
│                    │ • But some steps require LLM reasoning/tools        │
│                    │ • You want simplicity at the top, power inside      │
│                    │ • E.g.: verify → load_prefs → route to expert      │
│                    │   where each expert is a LangGraph agent           │
│                    │                                                     │
│                    │ EXAMPLE: Customer Support Multi-Agent System        │
│                    │ • Orchestration: Agentkube-Mini (50 lines)          │
│                    │ • Specialists: LangGraph agents (300+ lines each)   │
│                    │ • Result: Simple top-level, powerful sub-agents    │
└────────────────────┴──────────────────────────────────────────────────────┘


CODE EXAMPLE: Hybrid Invocation

# ─────── Agentkube-Mini Orchestration Layer ─────────
def route_to_specialist(input_data: Dict[str, Any]) -> Dict[str, Any]:
    question = input_data.get("question", "")
    customer_id = input_data.get("customer_id")
    
    # Prepare input for LangGraph sub-agents
    config = {"configurable": {"thread_id": uuid.uuid4(), "user_id": customer_id}}
    
    if "music" in question.lower():
        # Invoke LangGraph music sub-agent
        result = music_catalog_subagent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        response = result["messages"][-1].content
    elif "invoice" in question.lower():
        # Invoke LangGraph invoice sub-agent
        result = invoice_information_subagent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        response = result["messages"][-1].content
    else:
        response = "I'm not sure how to help with that."
    
    return {"response": response}

# Then wrap in Agentkube-Mini agent
graph.add(Agent("route", route_to_specialist), depends_on=["verify", "load_memory"])

→ The "route" agent is thin wrapper (10 lines)
→ It delegates to LangGraph sub-agents for real work
→ Agentkube-Mini handles top-level orchestration


METRICS COMPARISON:

Feature                     │ LangGraph Only  │ Agentkube Only  │ Hybrid
────────────────────────────┼─────────────────┼─────────────────┼──────────
Orchestration complexity    │ High (300+)     │ Low (50)        │ Low (50+300)
Parallelization             │ Sequential      │ Parallel        │ Parallel
Tool-calling loops          │ ✓ Built-in      │ ✗ Manual        │ ✓ In LangGraph
Human interrupts            │ ✓ Native        │ ✗ Manual        │ ◐ Simplified
Long-term memory            │ ✓ InMemoryStore │ ◐ Dict+save     │ ◐ Dict+save
Thread safety               │ ✓ Native        │ ◐ Manual config │ ◐ Manual config
Testing                     │ Hard (stateful) │ Easy (pure fn)  │ Easy + Powerful
Debuggability               │ Hard (nested)   │ Easy (linear)   │ Easy (linear)
Learning curve              │ Steep           │ Gentle          │ Moderate
Production ready            │ ✓               │ ✓               │ ✓ (recommended)
"""


# ============================================================================
# PART 4: QUICK REFERENCE
# ============================================================================

QUICK_REFERENCE = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUICK REFERENCE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

WHEN NOTEBOOK (LangGraph) IS THE RIGHT CHOICE:
  • You're building a single sophisticated agent
  • Your agent needs tool-calling loops (ReAct)
  • You want human-in-the-loop with native interrupts
  • Each task is LLM-driven (not deterministic)
  • You need detailed tracing (LangSmith)
  
WHEN HYBRID IS THE RIGHT CHOICE (MOST PRODUCTION SYSTEMS):
  • You have a clear orchestration flow at top level
  • But some steps require LLM reasoning
  • You want simplicity + power
  • You want parallelization + sophistication
  • You want testable + performant
  
WHEN AGENTKUBE-MINI ONLY IS THE RIGHT CHOICE:
  • Your entire system is deterministic DAG orchestration
  • No LLM reasoning needed (or keep it hidden inside functions)
  • You need fast parallelization
  • You prioritize simplicity
  • You're not doing tool-calling loops


MIGRATION PATH (from Notebook to Production):

Step 1: Start with Notebook (LangGraph)
   └─ Validate ideas, test LLM behavior, iterate

Step 2: Identify Top-Level Flow
   └─ What's the sequence of steps? (verify → route → save)

Step 3: Extract Step Functions
   └─ Create Python functions for each step
   └─ Some call LangGraph agents, some are pure logic

Step 4: Build Agentkube-Mini DAG
   └─ Define TaskGraph with dependencies
   └─ Replace StateGraph with simple Dict passing

Step 5: Test & Deploy
   └─ Easier to debug and parallelize
   └─ Still has power of LangGraph inside specialized agents


CODE PATTERNS:

Pattern 1: Pure Agentkube-Mini (Simple Orchestration)
────────────────────────────────────────────────────
agent1 = Agent("step1", step1_fn)
agent2 = Agent("step2", step2_fn)
graph.add(agent1)
graph.add(agent2, depends_on=["step1"])
result = Runtime(graph).run(input_data)

Pattern 2: LangGraph Sub-Agents (Complex Reasoning)
────────────────────────────────────────────────────
music_agent = create_react_agent(llm, music_tools, ...)
invoice_agent = create_react_agent(llm, invoice_tools, ...)
result = music_agent.invoke({"messages": [HumanMessage(...)]}, config)

Pattern 3: Hybrid (Recommended)
────────────────────────────────
def orchestrate(input_data):
    # Step 1: Simple logic
    customer_id = extract_id(input_data)
    # Step 2: Load preferences
    prefs = memory_store.get(customer_id)
    # Step 3: Invoke LangGraph specialist
    specialist = music_agent if is_music_query else invoice_agent
    result = specialist.invoke({"messages": [...]}, config)
    # Step 4: Save preferences
    save_prefs(customer_id, extracted_prefs)
    return result

# Wrap in Agentkube-Mini
graph.add(Agent("orchestrate", orchestrate))
Runtime(graph).run(initial_input)
"""


# ============================================================================
# PRINT EVERYTHING
# ============================================================================

def main():
    print(COMPARISON_TABLE)
    print("\n" * 2)
    print(LANGGRAPH_FLOW)
    print("\n" * 2)
    print(AGENTKUBE_FLOW)
    print("\n" * 2)
    print(HYBRID_APPROACH)
    print("\n" * 2)
    print(QUICK_REFERENCE)


if __name__ == "__main__":
    main()
