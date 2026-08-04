import json
from typing import Iterator

from graph_pipeline.loaders.base import DataLoader


class JsonLoader(DataLoader):
    def can_handle(self, path: str) -> bool:
        return path.endswith(".json")

    def load(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Root object with exactly one array value → use that array
            array_values = [(k, v) for k, v in data.items() if isinstance(v, list)]
            if len(array_values) == 1:
                return array_values[0][1]

            # Nested object → flatten top-level keys into records, preserving structure
            records = []
            for key, value in data.items():
                if isinstance(value, dict):
                    record = {"_key": key, **value}
                else:
                    record = {"_key": key, "_value": value}
                records.append(record)
            return records

        return []

    def stream(self, path: str) -> Iterator[dict]:
        # JSON has no streaming format; load fully then yield.
        yield from self.load(path)
