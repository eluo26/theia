"""Load config.yaml and .env. The API key stays out of the settings object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_API_KEY = "your-key-here"
ProviderName = Literal["openai", "xai"]


class ConfigError(RuntimeError):
    """config.yaml is missing or does not match the settings schema."""


class MissingAPIKeyError(ConfigError):
    """The active provider's API key is unset or still the example placeholder."""


class Settings(BaseModel):
    """Values from config.yaml. Model ids are never defaulted in code."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName

    openai_base_url: str = Field(min_length=1)
    openai_api_key_env: str = Field(min_length=1)
    openai_fast_model: str = Field(min_length=1)
    openai_reasoning_model: str = Field(min_length=1)

    grok_fast_model: str = Field(min_length=1)
    grok_reasoning_model: str = Field(min_length=1)
    xai_base_url: str = Field(min_length=1)
    xai_api_key_env: str = Field(min_length=1)

    image_detail: Literal["auto", "low", "high", "original"]
    request_timeout_s: float = Field(gt=0)
    max_image_bytes: int = Field(gt=0)

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    hfov_deg: float = Field(gt=0, lt=180)
    vfov_deg: float = Field(gt=0, lt=180)
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist_coeffs: list[float] | None = None

    tile_grid: tuple[int, int]
    tile_overlap: float = Field(ge=0, lt=1)
    max_concurrency: int = Field(ge=1)
    refine_iou: float = Field(ge=0, le=1)
    label_sim: float = Field(ge=0, le=100)
    merge_deg: float = Field(gt=0)
    video_sample_every_s: float = Field(gt=0)
    blur_threshold: float = Field(ge=0)
    crop_pad: float = Field(ge=0)
    clinic_mode: bool

    # fast uses the provider's fast model for questions. reasoning uses the
    # flagship id. Both ids still come from config.yaml.
    query_model: Literal["fast", "reasoning"] = "fast"
    api_max_edge: int = Field(default=768, gt=0)
    max_objects_per_image: int = Field(default=12, ge=1)
    save_debug: bool = False
    verify_match: bool = False

    detector: Literal["owlv2", "grounding_dino"]
    owlv2_model: str = Field(min_length=1)
    grounding_dino_model: str = Field(min_length=1)

    cache_dir: str = Field(min_length=1)
    out_dir: str = Field(min_length=1)

    max_transport_retries: int = Field(ge=0)
    retry_base_delay_s: float = Field(gt=0)
    validation_retries: int = Field(ge=0)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    @property
    def cache_path(self) -> Path:
        return self.resolve_path(self.cache_dir)

    @property
    def out_path(self) -> Path:
        return self.resolve_path(self.out_dir)

    @property
    def fast_model(self) -> str:
        if self.provider == "openai":
            return self.openai_fast_model
        return self.grok_fast_model

    @property
    def reasoning_model(self) -> str:
        if self.provider == "openai":
            return self.openai_reasoning_model
        return self.grok_reasoning_model

    @property
    def active_query_model(self) -> str:
        if self.query_model == "fast":
            return self.fast_model
        return self.reasoning_model

    @property
    def base_url(self) -> str:
        if self.provider == "openai":
            return self.openai_base_url
        return self.xai_base_url

    @property
    def api_key_env(self) -> str:
        if self.provider == "openai":
            return self.openai_api_key_env
        return self.xai_api_key_env

    @property
    def key_help_url(self) -> str:
        if self.provider == "openai":
            return "https://platform.openai.com/api-keys"
        return "https://console.x.ai"

    @property
    def docs_urls(self) -> tuple[str, ...]:
        if self.provider == "openai":
            return (
                "https://developers.openai.com/api/docs/models",
                "https://developers.openai.com/api/docs/guides/structured-outputs",
                "https://platform.openai.com/docs/guides/images-vision",
            )
        return (
            "https://docs.x.ai/developers/models",
            "https://docs.x.ai/developers/model-capabilities/text/structured-outputs",
            "https://docs.x.ai/developers/model-capabilities/legacy/chat-completions",
        )


def load_settings(config_path: Path | None = None, *, load_env: bool = True) -> Settings:
    """Load settings from config.yaml. A .env file is read when it exists."""
    if load_env:
        env_path = REPO_ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)

    path = config_path if config_path is not None else REPO_ROOT / "config.yaml"
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a YAML mapping of settings.")
    raw = _apply_env_overrides(raw)

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file {path}:\n{exc}") from exc


def _apply_env_overrides(raw: dict) -> dict:
    """Honor VISION_MODEL from .env without changing provider selection in code."""
    model = os.environ.get("VISION_MODEL", "").strip()
    if not model:
        return raw
    provider = raw.get("provider")
    if provider == "xai":
        raw["grok_fast_model"] = model
        raw["grok_reasoning_model"] = model
    elif provider == "openai":
        raw["openai_fast_model"] = model
        raw["openai_reasoning_model"] = model
    return raw


def api_key_is_present(settings: Settings | None = None) -> bool:
    """True when the active provider's key is set and is not the example placeholder."""
    if settings is None:
        settings = load_settings(load_env=False)
    key = os.environ.get(settings.api_key_env, "").strip()
    return bool(key) and key != PLACEHOLDER_API_KEY


def require_api_key(settings: Settings | None = None) -> str:
    """Return the active provider's key. Never log the return value."""
    if settings is None:
        settings = load_settings(load_env=False)
    env_name = settings.api_key_env
    key = os.environ.get(env_name, "").strip()
    if not key or key == PLACEHOLDER_API_KEY:
        raise MissingAPIKeyError(
            f"{env_name} is not set. Add it to .env in the project root."
        )
    return key
