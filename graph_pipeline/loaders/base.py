from abc import ABC, abstractmethod
from typing import Iterator


class DataLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[dict]:
        """Load records from path. Always returns a flat list of dicts."""

    @abstractmethod
    def stream(self, path: str) -> Iterator[dict]:
        """Yield records one at a time without loading the full file."""

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Return True if this loader handles the given file."""
