import json
from typing import Iterator

from graph_pipeline.loaders.base import DataLoader


class JsonlLoader(DataLoader):
    def can_handle(self, path: str) -> bool:
        return path.endswith(".jsonl") or path.endswith(".ndjson")

    def load(self, path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("kind") == "export-dump-header":
                    continue
                records.append(record)
        return records

    def stream(self, path: str) -> Iterator[dict]:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("kind") == "export-dump-header":
                    continue
                yield record
