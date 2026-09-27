"""Vision calls through Chat Completions.

The provider, model ids, base URL, image detail, and timeout come from
config.yaml. This module does not substitute a different model when a call fails.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx
from openai import OpenAI
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, ValidationError

register_heif_opener()

from vision.config import Settings, require_api_key
from vision.schemas import IndexResponse

T = TypeVar("T", bound=BaseModel)
_cache_lock = threading.Lock()

logger = logging.getLogger("vision.client")

def index_prompt(max_objects: int, query: str | None = None) -> str:
    """Short scene list. A cap keeps the later question small enough to answer quickly.

    A query asks for objects that could be what the user is looking for.
    Omitting it keeps the full-scene prompt used by offline indexing.
    """
    question = " ".join(query.split()) if isinstance(query, str) and query.strip() else None
    if question:
        shown = question.replace('"', "'")
        listing = (
            f'The user is looking for: "{shown}". '
            "List visible objects that could be what they asked for. "
            "Include the same kind of object when color, brand, or other details are uncertain. "
            "A short label is fine. Do not leave an object out only because you cannot confirm every word. "
            f"At most {max_objects}."
        )
    else:
        listing = (
            f"List the distinct objects a person might ask you to find. At most {max_objects}."
        )
    return f"""You are the scene-understanding module for a tabletop laser turret.
{listing}
Skip walls, the floor, the ceiling, the bare table, shadows, and small background clutter.

For each object provide:
- label: a short noun phrase, such as "blue water bottle"
- description: color, brand, any readable text, and one neighboring object. One sentence.
- box: a tight bounding box as [x1, y1, x2, y2], normalized from 0 to 1000, origin at the top-left
- count: how many of that same object sit inside the box
- drug_name: the medication name if the object is a drug package, otherwise null
- expiry_text: any visible expiry date text, otherwise null
- location: where it sits relative to one nearby object, said so a person can find it. A short
  phrase that starts with a preposition, such as "on top of the blue tape roll" or "next to the
  white energy drink can". Use "on top of" only when it rests on the other object. null if
  nothing is nearby.

Only include objects you can actually see. Do not invent hidden items.
"""


INDEX_PROMPT = index_prompt(12)

_KEY_PATTERN = re.compile(r"(?:xai-|sk-)[A-Za-z0-9_\-]{8,}")
_AUTH_PATTERN = re.compile(r"(?i)authorization:\s*bearer\s+\S+")
# OpenAI image input: PNG, JPEG, WEBP, and non-animated GIF.
# https://platform.openai.com/docs/guides/images-vision
# HEIC is accepted on disk and converted to JPEG before it reaches this map.
_ALLOWED_SUFFIXES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_ALLOWED_MIMES = set(_ALLOWED_SUFFIXES.values())
_RETRIABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class GrokCallError(RuntimeError):
    """The vision call failed, or the response did not match the schema."""


def redact(text: str) -> str:
    """Remove API-key-like strings and Authorization headers before logging."""
    redacted = _KEY_PATTERN.sub("[REDACTED]", text)
    return _AUTH_PATTERN.sub("Authorization: Bearer [REDACTED]", redacted)


def response_cache_key(
    *, image_bytes: bytes, prompt: str, model: str, image_detail: str
) -> str:
    """SHA-256 of the image, prompt, model, and image detail.

    Image detail is included so a config change does not reuse boxes from a
    different detail level. The plan's key is image bytes + prompt + model.
    """
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(image_detail.encode("utf-8"))
    return digest.hexdigest()


def describe_image(
    image_path: Path,
    settings: Settings,
    *,
    client_factory: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    use_cache: bool = True,
) -> tuple[IndexResponse, str]:
    """List objects in one image. Returns the parsed response and cache status."""
    path = Path(image_path)
    image_bytes, mime = _load_image_bytes(path, settings)
    image_bytes, mime = _shrink_for_api(image_bytes, mime, settings.api_max_edge)
    return parse_image_bytes(
        image_bytes,
        mime,
        settings,
        prompt=index_prompt(settings.max_objects_per_image),
        response_model=IndexResponse,
        model=settings.fast_model,
        client_factory=client_factory,
        sleeper=sleeper,
        use_cache=use_cache,
    )


def parse_image_bytes(
    image_bytes: bytes,
    mime: str,
    settings: Settings,
    *,
    prompt: str,
    response_model: type[T],
    model: str,
    client_factory: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    use_cache: bool = True,
) -> tuple[T, str]:
    """Structured vision call. Cache key is image bytes + prompt + model + detail."""
    _check_encoded_image(image_bytes, mime, settings)
    key = response_cache_key(
        image_bytes=image_bytes,
        prompt=prompt,
        model=model,
        image_detail=settings.image_detail,
    )
    messages = _messages(prompt, image_bytes, mime, settings.image_detail)
    return _run_completion(
        settings,
        model=model,
        messages=messages,
        response_model=response_model,
        cache_key=key,
        client_factory=client_factory,
        sleeper=sleeper,
        use_cache=use_cache,
    )


def parse_text(
    prompt: str,
    settings: Settings,
    *,
    response_model: type[T],
    model: str,
    client_factory: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    use_cache: bool = True,
) -> tuple[T, str]:
    """Structured text call. No image part and no reasoning-effort field."""
    key = response_cache_key(
        image_bytes=b"",
        prompt=prompt,
        model=model,
        image_detail="text",
    )
    messages = [{"role": "user", "content": prompt}]
    return _run_completion(
        settings,
        model=model,
        messages=messages,
        response_model=response_model,
        cache_key=key,
        client_factory=client_factory,
        sleeper=sleeper,
        use_cache=use_cache,
    )


def _run_completion(
    settings: Settings,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    cache_key: str,
    client_factory: Callable[..., Any] | None,
    sleeper: Callable[[float], None],
    use_cache: bool,
) -> tuple[T, str]:
    cache_file = settings.cache_path / f"{cache_key}.json"
    if use_cache:
        cached = _read_cache(cache_file, model, response_model)
        if cached is not None:
            logger.info(
                "vision response model=%s latency_ms=0 cache=hit validation=passed",
                model,
            )
            return cached, "hit"

    api_key = require_api_key(settings)
    factory = client_factory or _default_client
    client = factory(
        api_key=api_key,
        base_url=settings.base_url,
        timeout=httpx.Timeout(settings.request_timeout_s),
        max_retries=0,
    )

    attempts = settings.validation_retries + 1
    last_error = "response did not match the schema"
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            completion = _call_with_transport_retries(
                client, model, messages, response_model, settings, sleeper
            )
            parsed = _parse_completion(completion, response_model)
        except GrokCallError:
            raise
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = redact(str(exc))
            latency_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "vision response model=%s latency_ms=%.0f cache=miss validation=failed attempt=%s error=%s",
                model,
                latency_ms,
                attempt,
                last_error,
            )
            if attempt == attempts:
                break
            continue
        else:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "vision response model=%s latency_ms=%.0f cache=miss validation=passed",
                model,
                latency_ms,
            )
            if use_cache:
                _write_cache(cache_file, model, parsed)
            return parsed, "miss"

    raise GrokCallError(
        "The model returned a response that failed schema validation "
        f"{attempts} time(s) for model {model!r}. The image was skipped. "
        f"Last error: {last_error}"
    )


def _check_encoded_image(image_bytes: bytes, mime: str, settings: Settings) -> None:
    if mime not in _ALLOWED_MIMES:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise GrokCallError(
            f"Unsupported image type {mime!r}. Supported types are {allowed}: {settings.docs_urls[-1]}"
        )
    _reject_animated_gif(image_bytes, mime, settings.docs_urls[-1])
    if not image_bytes:
        raise GrokCallError("Image is empty")
    if len(image_bytes) > settings.max_image_bytes:
        raise GrokCallError(
            f"Image is {len(image_bytes)} bytes, above the configured limit of "
            f"{settings.max_image_bytes} bytes ({settings.docs_urls[-1]})."
        )


def _reject_animated_gif(image_bytes: bytes, mime: str, docs_url: str) -> None:
    if mime != "image/gif":
        return
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            animated = bool(getattr(image, "is_animated", False))
    except Exception:
        return
    if animated:
        raise GrokCallError(
            "Animated GIF is not supported. The image docs allow a non-animated GIF only: "
            f"{docs_url}"
        )


def _default_client(**kwargs: Any) -> OpenAI:
    return OpenAI(**kwargs)


def _load_image_bytes(path: Path, settings: Settings) -> tuple[bytes, str]:
    if not path.is_file():
        raise GrokCallError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".heic":
        # Convert before the size check. describe_image then downscales, and the
        # bytes that reach the API are JPEG. HEIC itself is never an image part.
        return _heic_file_as_jpeg(path), "image/jpeg"
    mime = _ALLOWED_SUFFIXES.get(suffix)
    if mime is None:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise GrokCallError(
            f"Unsupported image type {suffix!r}. Supported types are {allowed}: {settings.docs_urls[-1]}"
        )
    image_bytes = path.read_bytes()
    if not image_bytes:
        raise GrokCallError(f"Image file is empty: {path}")
    if len(image_bytes) > settings.max_image_bytes:
        raise GrokCallError(
            f"Image is {len(image_bytes)} bytes, above the configured limit of "
            f"{settings.max_image_bytes} bytes ({settings.docs_urls[-1]})."
        )
    image_bytes = _upright_image_bytes(image_bytes, mime)
    if len(image_bytes) > settings.max_image_bytes:
        raise GrokCallError(
            f"Image is {len(image_bytes)} bytes, above the configured limit of "
            f"{settings.max_image_bytes} bytes ({settings.docs_urls[-1]})."
        )
    return image_bytes, mime


def _heic_file_as_jpeg(path: Path) -> bytes:
    """Decode a HEIC still and return upright JPEG bytes.

    The vision API does not accept HEIC. EXIF orientation is baked in here so
    the JPEG matches the RGB path used when a folder is indexed.
    """
    try:
        with Image.open(path) as image:
            image.load()
            turned = ImageOps.exif_transpose(image)
            if turned is None:
                turned = image
            buffer = io.BytesIO()
            turned.convert("RGB").save(buffer, format="JPEG", quality=95)
            encoded = buffer.getvalue()
    except Exception as exc:
        raise GrokCallError(f"Could not convert HEIC to JPEG: {path}: {exc}") from exc
    if not encoded:
        raise GrokCallError(f"Image file is empty: {path}")
    return encoded


def _upright_image_bytes(image_bytes: bytes, mime: str) -> bytes:
    """Bake a sideways EXIF orientation into the pixels.

    Phone JPEGs are often stored rotated, with the orientation only in metadata.
    Indexing already applies that tag. A one-photo describe call has to do the
    same, or the boxes come back in a different frame than the file on disk.
    Bytes that are not a real image are returned unchanged.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            orientation = image.getexif().get(274)
            if not orientation or int(orientation) == 1:
                return image_bytes
            upright = ImageOps.exif_transpose(image)
            if upright is None:
                return image_bytes
            buffer = io.BytesIO()
            if mime == "image/jpeg":
                upright.convert("RGB").save(buffer, format="JPEG", quality=95)
            elif mime == "image/png":
                upright.save(buffer, format="PNG")
            elif mime == "image/webp":
                upright.save(buffer, format="WEBP")
            elif mime == "image/gif":
                upright.save(buffer, format="GIF")
            else:
                return image_bytes
            encoded = buffer.getvalue()
    except Exception:
        return image_bytes
    return encoded or image_bytes


def _shrink_for_api(image_bytes: bytes, mime: str, max_edge: int) -> tuple[bytes, str]:
    """Downscale a large photo before upload. Smaller files are returned unchanged."""
    if max_edge <= 0:
        return image_bytes, mime
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if max(image.size) <= max_edge:
                return image_bytes, mime
            turned = ImageOps.exif_transpose(image)
            if turned is None:
                turned = image
            scale = max_edge / float(max(turned.size))
            turned = turned.resize(
                (
                    max(1, int(round(turned.size[0] * scale))),
                    max(1, int(round(turned.size[1] * scale))),
                ),
                Image.Resampling.BILINEAR,
            )
            buffer = io.BytesIO()
            turned.convert("RGB").save(buffer, format="JPEG", quality=80)
    except Exception:
        return image_bytes, mime
    encoded = buffer.getvalue()
    return (encoded or image_bytes), "image/jpeg"


def _messages(prompt: str, image_bytes: bytes, mime: str, detail: str) -> list[dict[str, Any]]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                        "detail": detail,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _call_with_transport_retries(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    settings: Settings,
    sleeper: Callable[[float], None],
) -> Any:
    attempts = settings.max_transport_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            # Chat Completions structured output. The current OpenAI SDK example is
            # client.chat.completions.parse(..., response_format=<pydantic model>).
            # https://developers.openai.com/api/docs/guides/structured-outputs
            # reasoning.effort is a Responses API field and is not sent here.
            return client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
            )
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in {400}:
                raise GrokCallError(_docs_error(settings, model, status, exc)) from exc
            if status in {401, 403}:
                raise GrokCallError(
                    f"The API rejected the key (HTTP {status}). "
                    f"Check {settings.api_key_env} in .env. If the key leaked, revoke it at "
                    f"{settings.key_help_url} and create a new one. "
                    f"Details: {redact(str(exc))}"
                ) from exc
            retriable = status in _RETRIABLE_STATUS or type(exc).__name__ in {
                "APIConnectionError",
                "APITimeoutError",
                "RateLimitError",
            }
            if not retriable or attempt == attempts:
                raise GrokCallError(_docs_error(settings, model, status, exc)) from exc
            delay = settings.retry_base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "vision transport retry model=%s attempt=%s delay_s=%s error=%s",
                model,
                attempt,
                delay,
                redact(f"{type(exc).__name__}: {exc}"),
            )
            sleeper(delay)
    raise GrokCallError(_docs_error(settings, model, None, RuntimeError("retries exhausted")))


def _docs_error(settings: Settings, model: str, status: int | None, exc: BaseException) -> str:
    status_text = f"HTTP {status}" if status is not None else type(exc).__name__
    links = " ".join(settings.docs_urls)
    return (
        f"API call failed ({status_text}) for model {model!r}. "
        "Check the current vision model names and Chat Completions parameters in the docs "
        f"instead of changing the model in code: {links}. "
        f"Details: {redact(str(exc))}"
    )


def _parse_completion(completion: Any, response_model: type[T]) -> T:
    choices = getattr(completion, "choices", None)
    if not choices:
        raise ValueError("response had no choices")
    message = choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"model refusal: {refusal}")
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, response_model):
        return parsed
    if parsed is not None:
        return response_model.model_validate(parsed)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("response content was empty")
    return response_model.model_validate_json(content)


def _read_cache(path: Path, model: str, response_model: type[T]) -> T | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != model:
            logger.warning("cache entry model mismatch path=%s; ignoring", path.name)
            return None
        return response_model.model_validate(payload["response"])
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
        logger.warning(
            "cache entry failed validation path=%s error=%s; ignoring",
            path.name,
            redact(str(exc)),
        )
        return None


def _write_cache(path: Path, model: str, response: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "response": response.model_dump()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2)
    with _cache_lock:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
