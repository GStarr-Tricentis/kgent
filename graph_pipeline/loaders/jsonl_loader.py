import json

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
