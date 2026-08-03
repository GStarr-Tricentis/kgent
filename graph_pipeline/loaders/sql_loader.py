import sqlite3
from typing import Iterator

from graph_pipeline.loaders.base import DataLoader

_CHUNK_SIZE = 1000


class SqlLoader(DataLoader):
    def can_handle(self, path: str) -> bool:
        return path.endswith(".sqlite") or path.endswith(".db")

    def load(self, path: str) -> list[dict]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            records = []
            for table in tables:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                for row in rows:
                    record = dict(row)
                    record["_table"] = table
                    records.append(record)
            return records
        finally:
            conn.close()

    def stream(self, path: str) -> Iterator[dict]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"SELECT * FROM {table}")  # noqa: S608
                while True:
                    rows = cursor.fetchmany(_CHUNK_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        yield {**dict(row), "_table": table}
        finally:
            conn.close()
