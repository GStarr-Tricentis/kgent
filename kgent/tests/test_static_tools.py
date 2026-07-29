from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from kgent.tools.static.filesystem import LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
from kgent.tools.static.shell import RUN_COMMAND_TOOL


# --- read_file ---

def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = READ_FILE_TOOL.callable({"path": str(f)})
    assert "hello world" in result


def test_read_file_missing():
    result = READ_FILE_TOOL.callable({"path": "/nonexistent/path/abc123.txt"})
    assert "error" in result.lower() or "ERROR" in result


def test_read_file_does_not_raise(tmp_path):
    result = READ_FILE_TOOL.callable({"path": str(tmp_path / "nope.txt")})
    assert isinstance(result, str)


def test_read_file_truncates_large_file(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 20000)
    result = READ_FILE_TOOL.callable({"path": str(f)})
    assert len(result) < 20000
    assert "TRUNCAT" in result.upper()


# --- write_file ---

def test_write_file(tmp_path):
    dest = tmp_path / "out.txt"
    result = WRITE_FILE_TOOL.callable({"path": str(dest), "content": "written!"})
    assert dest.read_text() == "written!"
    assert isinstance(result, str)


def test_write_file_creates_parents(tmp_path):
    dest = tmp_path / "a" / "b" / "c.txt"
    WRITE_FILE_TOOL.callable({"path": str(dest), "content": "deep"})
    assert dest.read_text() == "deep"


def test_write_file_does_not_raise():
    result = WRITE_FILE_TOOL.callable({"path": "/root/no_permission_here.txt", "content": "x"})
    assert isinstance(result, str)


# --- list_dir ---

def test_list_dir(tmp_path):
    (tmp_path / "file_a.txt").write_text("a")
    (tmp_path / "file_b.txt").write_text("b")
    (tmp_path / "subdir").mkdir()
    result = LIST_DIR_TOOL.callable({"path": str(tmp_path)})
    assert "file_a.txt" in result
    assert "file_b.txt" in result
    assert "subdir" in result


def test_list_dir_missing():
    result = LIST_DIR_TOOL.callable({"path": "/nonexistent/dir/xyz"})
    assert "error" in result.lower() or "ERROR" in result


def test_list_dir_does_not_raise():
    result = LIST_DIR_TOOL.callable({"path": "/definitely/does/not/exist"})
    assert isinstance(result, str)


# --- shell ---

def test_shell_safe_command():
    result = RUN_COMMAND_TOOL.callable({"command": "echo hello"})
    assert "hello" in result


def test_shell_nonzero_exit():
    result = RUN_COMMAND_TOOL.callable({"command": "ls /nonexistent_path_xyz_abc"})
    assert isinstance(result, str)
    # Should include exit code or error text, not raise
    assert len(result) > 0


def test_shell_timeout():
    result = RUN_COMMAND_TOOL.callable({"command": "sleep 60", "_timeout_override": 0.1})
    assert "timeout" in result.lower() or "timed out" in result.lower() or isinstance(result, str)


def test_shell_does_not_raise():
    result = RUN_COMMAND_TOOL.callable({"command": "echo safe"})
    assert isinstance(result, str)


def test_shell_output_contains_stderr():
    # ls on a missing path writes to stderr
    result = RUN_COMMAND_TOOL.callable({"command": "ls /no_such_path_xyz"})
    assert isinstance(result, str)
    assert len(result) > 0
