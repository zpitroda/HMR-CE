"""
Tier 0: Active Working Context Buffer
Maintains the last K uncompressed turns in working memory as a ring buffer.
Evicted turns persist safely in Tier 1.
"""

from collections import deque
from threading import Lock
from typing import List, Optional
from hmr_ce.schemas import Tier1Record


class Tier0Buffer:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._buffer: deque[Tier1Record] = deque(maxlen=max_turns)
        self._lock = Lock()

    def append(self, turn: Tier1Record) -> Optional[Tier1Record]:
        """
        Append a turn to the active working context.
        Returns the evicted turn if capacity was exceeded, else None.
        """
        with self._lock:
            evicted = None
            if len(self._buffer) >= self.max_turns:
                evicted = self._buffer[0]
            self._buffer.append(turn)
            return evicted

    def get_turns(self) -> List[Tier1Record]:
        """Return a copy of the ordered turns in the buffer."""
        with self._lock:
            return list(self._buffer)

    def get_context_text(self) -> str:
        """Concatenate buffer turns as conversational context for perplexity conditioning."""
        with self._lock:
            lines = []
            for t in self._buffer:
                lines.append(f"{t.speaker_role.value}: {t.raw_text}")
            return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Format active buffer for injection into LLM generation prompt."""
        with self._lock:
            if not self._buffer:
                return ""
            formatted = ["=== ACTIVE WORKING CONTEXT (Tier 0) ==="]
            for t in self._buffer:
                formatted.append(f"[{t.speaker_role.value}] (Turn {t.turn_id}): {t.raw_text}")
            return "\n".join(formatted)

    def clear(self) -> None:
        """Reset the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
