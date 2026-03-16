from __future__ import annotations

import inspect
from typing import Any, Callable

from agentkube_mini.agent import Agent


def auto_agent(name: str, fn: Callable[..., Any]) -> Agent:
    """Auto-wrap any callable into an Agent.

    Supports:
      fn()                          → Agent ignores input and deps
      fn(input)                     → Agent receives initial_input
      fn(input, dependency_outputs) → Agent receives both
      fn(dependency_outputs)        → when param is named deps/dependencies
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters)

    if len(params) == 0:
        return Agent(name=name, fn=lambda: fn())

    if len(params) == 1:
        if params[0].lower() in {"deps", "dependencies", "dependency_outputs"}:
            return Agent(name=name, fn=lambda _input, dep_out: fn(dep_out))
        return Agent(name=name, fn=fn)

    if len(params) == 2:
        return Agent(name=name, fn=fn)

    raise ValueError(
        f"Unsupported callable signature for '{name}'. Expected 0-2 params, got {len(params)}"
    )
