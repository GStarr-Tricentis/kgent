from __future__ import annotations

import subprocess
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "agent_poc" / "config" / "config.yaml"


def get_ollama_models() -> list[str]:
    """Return model names from `ollama list`. Falls back to the config default."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])
        if models:
            return models
    except Exception:
        pass

    from agent_poc.config.loader import load_config
    return [load_config(CONFIG_PATH).model.model_name]
