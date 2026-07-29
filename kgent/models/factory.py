from __future__ import annotations

from kgent.agent.types import ModelBackend
from kgent.config.loader import AgentPocConfig
from kgent.models.openai_compatible import OpenAICompatibleBackend


async def make_backend(
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
        from kgent.models.tricentis_backend import TricentisBackend
        deployment = model_override or config.tricentis.deployment
        if not deployment:
            raise ValueError(
                "Tricentis provider requires a deployment name. "
                "Set tricentis.deployment in config.yaml or pass --model."
            )
        return await TricentisBackend.create(deployment=deployment, temperature=config.model.temperature)
    if provider == "bedrock":
        from kgent.models.bedrock_backend import BedrockBackend
        model_id = model_override or config.bedrock.model_id
        if not model_id:
            raise ValueError(
                "Bedrock provider requires a model_id. "
                "Pass --model or set bedrock.model_id in config.yaml."
            )
        region = config.bedrock.region
        if region.startswith("${"):
            region = "us-east-1"
        return BedrockBackend(
            model_id=model_id,
            region=region,
            temperature=config.model.temperature,
        )
    raise ValueError(f"Unknown provider: {provider!r}. Valid options: local, tricentis, bedrock")
