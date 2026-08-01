"""OpenAI LLM client wrapper for grounded answer generation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the OpenAI generation client."""

    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 1200


class LLMClient:
    """Async OpenAI client with bounded output for rate-limit efficiency."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200")),
        )
        api_key = os.getenv(self.config.api_key_env)
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    def is_ready(self) -> bool:
        """Check if the required API key is present."""

        return self.client is not None

    async def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate a grounded answer using the OpenAI Responses API."""

        if self.client is None:
            logger.warning("OpenAI API key is missing; skipping generation")
            return None

        try:
            response = await self.client.responses.create(
                model=self.config.model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=self.config.max_output_tokens,
            )
            answer = response.output_text.strip()
            return answer or None
        except Exception as exc:
            logger.warning("OpenAI generation failed for model %s: %s", self.config.model, exc)
            return None
