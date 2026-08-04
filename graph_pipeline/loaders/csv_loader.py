import csv
from typing import Iterator

from graph_pipeline.loaders.base import DataLoader


def _coerce(value: str):
    """Convert string to int, float, None, or leave as str."""
    if value == "":
        return None
    try:
        as_int = int(value)
        return as_int
    except ValueError:
        pass
    try:
        as_float = float(value)
        return as_float
    except ValueError:
        pass
    return value


class CsvLoader(DataLoader):
    def can_handle(self, path: str) -> bool:
        return path.endswith(".csv") or path.endswith(".tsv")

    def load(self, path: str) -> list[dict]:
        delimiter = "\t" if path.endswith(".tsv") else ","
        records = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append({k: _coerce(v) for k, v in row.items()})
        return records

    def stream(self, path: str) -> Iterator[dict]:
        delimiter = "\t" if path.endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                yield {k: _coerce(v) for k, v in row.items()}
