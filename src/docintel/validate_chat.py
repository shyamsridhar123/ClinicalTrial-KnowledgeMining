from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence

import httpx
from httpx import Timeout
from openai import (
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

_LOGGER = logging.getLogger("docintel.validate_chat")
_DEFAULT_PROMPT = "Granite Docling health check: reply with a short acknowledgement."


@dataclass(slots=True)
class ValidationConfig:
    base_url: str
    api_key: str
    model: str
    prompt: str
    timeout: float
    probe_models: bool
    max_tokens: Optional[int]
    temperature: float


async def _probe_models_endpoint(config: ValidationConfig) -> bool:
    models_url = config.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    _LOGGER.info("Probing MAX models endpoint | url=%s", models_url)
    start = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=config.timeout, headers=headers) as http_client:
            response = await http_client.get(models_url)
    except Exception:  # pragma: no cover - unexpected networking errors
        _LOGGER.exception("Models endpoint probe failed")
        return False

    duration = perf_counter() - start
    _LOGGER.info(
        "Models endpoint response | status=%s | latency_s=%.2f | body_preview=%s",
        response.status_code,
        duration,
        response.text[:200].replace("\n", " "),
    )
    if response.status_code >= 400:
        _LOGGER.error("Models endpoint returned error status %s", response.status_code)
        return False
    return True


async def run_chat_validation(config: ValidationConfig) -> int:
    _LOGGER.info(
        "Starting MAX chat validation | base_url=%s | model=%s | timeout_s=%.1f | probe_models=%s",
        config.base_url,
        config.model,
        config.timeout,
        config.probe_models,
    )

    timeout = Timeout(
        config.timeout,
        connect=min(10.0, config.timeout),
        read=config.timeout,
        write=config.timeout,
    )

    if config.probe_models and not await _probe_models_endpoint(config):
        _LOGGER.error("Aborting chat validation because models probe failed")
        return 1

    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=timeout,
    )

    messages = [
        {
            "role": "system",
            "content": "You are Granite Docling responding to a health-check.",
        },
        {
            "role": "user",
            "content": config.prompt,
        },
    ]

    chat_kwargs = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        chat_kwargs["max_tokens"] = config.max_tokens

    start = perf_counter()
    try:
        completion = await client.chat.completions.create(**chat_kwargs)
    except (APITimeoutError, RateLimitError, InternalServerError, APIStatusError) as exc:
        duration = perf_counter() - start
        _LOGGER.error(
            "Chat completion failed | duration_s=%.2f | error_type=%s | status=%s | request_id=%s",
            duration,
            exc.__class__.__name__,
            getattr(exc, "status_code", "n/a"),
            getattr(exc, "request_id", "n/a"),
        )
        await client.close()
        return 1
    except Exception as exc:  # pragma: no cover - unexpected OpenAI client errors
        duration = perf_counter() - start
        _LOGGER.exception("Chat completion failed unexpectedly | duration_s=%.2f", duration)
        await client.close()
        return 1

    duration = perf_counter() - start
    await client.close()

    usage = getattr(completion, "usage", None)
    choice = completion.choices[0].message if completion.choices else None
    content = ""
    if choice is not None:
        raw_content = getattr(choice, "content", "")
        if isinstance(raw_content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            content = str(raw_content)

    _LOGGER.info(
        "Chat completion succeeded | latency_s=%.2f | completion_id=%s | prompt_tokens=%s | completion_tokens=%s",
        duration,
        getattr(completion, "id", "n/a"),
        getattr(usage, "prompt_tokens", "n/a"),
        getattr(usage, "completion_tokens", "n/a"),
    )
    _LOGGER.info("Model response preview: %s", content[:200].replace("\n", " "))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that the Modular MAX chat completions endpoint responds to requests.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DOCINTEL_DOCLING_MAX_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL for the MAX server (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DOCINTEL_DOCLING_API_KEY", "EMPTY"),
        help="API key forwarded to the MAX server (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DOCINTEL_DOCLING_MODEL_NAME", "ibm-granite/granite-docling-258M"),
        help="Model identifier exposed by the MAX server (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Prompt sent to the model (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("DOCINTEL_MAX_CHAT_TIMEOUT_SECONDS", "30")),
        help="Total timeout (seconds) applied to the chat completion request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens limit for the chat completion response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the health-check request.",
    )
    parser.add_argument(
        "--no-model-probe",
        action="store_true",
        help="Skip the preliminary GET /models request before issuing the chat completion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging for additional diagnostics.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = ValidationConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        timeout=float(args.timeout),
        probe_models=not args.no_model_probe,
        max_tokens=args.max_tokens,
        temperature=float(args.temperature),
    )

    try:
        return asyncio.run(run_chat_validation(config))
    except KeyboardInterrupt:
        _LOGGER.warning("Validation interrupted by user")
        return 130


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
