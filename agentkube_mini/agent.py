from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Agent:
    """Minimal agent: a named worker with a callable.

    Supports three calling conventions:
      fn(input)                        # simple — like the blueprint
      fn(input, dependency_outputs)    # when agent needs upstream results
      fn()                             # no-arg thunk
    """

    name: str
    fn: Callable[..., Any]

    def run(self, input: Any, dependency_outputs: dict[str, Any] | None = None) -> Any:
        sig = inspect.signature(self.fn)
        n_params = len(sig.parameters)
        if n_params == 0:
            return self.fn()
        if n_params == 1:
            return self.fn(input)
        return self.fn(input, dependency_outputs or {})
