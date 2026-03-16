from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict


EventHandler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    type: str
    task: str
    payload: Any
    timestamp: str


class EventBus:
    def __init__(self) -> None:
        self._handlers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self.history: list[Event] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, task: str, payload: Any = None) -> Event:
        event = Event(
            type=event_type,
            task=task,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.history.append(event)
        for handler in self._handlers.get(event_type, []):
            handler(event)
        return event
