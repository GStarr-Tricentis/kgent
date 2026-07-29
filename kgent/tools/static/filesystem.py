from __future__ import annotations

import os
from pathlib import Path

from kgent.agent.types import RegisteredTool, ToolSource

MAX_OUTPUT = 8192


def _read_file(args: dict) -> str:
    try:
        p = Path(args["path"]).resolve()
        text = p.read_text(errors="replace")
        if len(text) > MAX_OUTPUT:
            text = text[:MAX_OUTPUT] + "\n[TRUNCATED — file exceeds 8192 chars]"
        return text
    except Exception as exc:
        return f"ERROR: {exc}"


def _write_file(args: dict) -> str:
    try:
        p = Path(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return f"Wrote {len(args['content'])} bytes to {p}"
    except Exception as exc:
        return f"ERROR: {exc}"


def _list_dir(args: dict) -> str:
    try:
        p = Path(args.get("path", ".")).resolve()
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        if not entries:
            return "(empty directory)"
        lines = []
        for e in entries:
            kind = "file" if e.is_file() else "dir "
            size = e.stat().st_size if e.is_file() else ""
            size_str = f"  {size} bytes" if size != "" else ""
            lines.append(f"{kind}  {e.name}{size_str}")
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"


READ_FILE_TOOL = RegisteredTool(
    name="read_file",
    description="Read the text contents of a file. Returns the content as a string.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file."},
        },
        "required": ["path"],
    },
    callable=_read_file,
    source=ToolSource.STATIC,
)

WRITE_FILE_TOOL = RegisteredTool(
    name="write_file",
    description="Write text content to a file, creating parent directories as needed.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    },
    callable=_write_file,
    source=ToolSource.STATIC,
)

LIST_DIR_TOOL = RegisteredTool(
    name="list_dir",
    description="List the contents of a directory with file names, sizes, and types.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list. Defaults to '.'."},
        },
        "required": [],
    },
    callable=_list_dir,
    source=ToolSource.STATIC,
)
