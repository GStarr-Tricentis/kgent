from __future__ import annotations

from agent_poc.agent.types import ModelBackend
from agent_poc.config.loader import AgentPocConfig
from agent_poc.models.openai_compatible import OpenAICompatibleBackend


def make_backend(
    config: AgentPocConfig,
    provider: str | None = None,
    model_override: str | None = None,
) -> ModelBackend:
    provider = provider or config.model.provider
    if model_override:
        config.model.model_name = model_override
    if provider == "local":
        return OpenAICompatibleBackend(config.model)
    if provider == "tricentis":
        from agent_poc.models.tricentis_backend import TricentisBackend
        deployment = model_override or config.tricentis.deployment
        if not deployment:
            raise ValueError(
                "Tricentis provider requires a deployment name. "
                "Set tricentis.deployment in config.yaml or pass --model."
            )
        return TricentisBackend(deployment=deployment, temperature=config.model.temperature)
    raise ValueError(f"Unknown provider: {provider!r}. Valid options: local, tricentis")
