from typing import Iterator

from graph_pipeline.loaders.base import DataLoader
from graph_pipeline.loaders.jsonl_loader import JsonlLoader
from graph_pipeline.loaders.json_loader import JsonLoader
from graph_pipeline.loaders.csv_loader import CsvLoader
from graph_pipeline.loaders.sql_loader import SqlLoader

_LOADERS: list[DataLoader] = [
    JsonlLoader(),
    JsonLoader(),
    CsvLoader(),
    SqlLoader(),
]


def load(path: str) -> list[dict]:
    """Detect format and load records. Raises ValueError if no loader matches."""
    for loader in _LOADERS:
        if loader.can_handle(path):
            return loader.load(path)
    raise ValueError(f"No loader found for: {path}")


def stream(path: str) -> Iterator[dict]:
    """Detect format and stream records one at a time. Raises ValueError if no loader matches."""
    for loader in _LOADERS:
        if loader.can_handle(path):
            yield from loader.stream(path)
            return
    raise ValueError(f"No loader found for: {path}")
