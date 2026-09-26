"""Pydantic models for Grok responses and the teammate-facing contract.

The indexing schema is what milestone 1 asks Grok to return.
QueryResult is the JSON contract from the plan; locate() fills it in a later milestone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _check_norm_box(value: list[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not all(0 <= number <= 1000 for number in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be between 0 and 1000")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return [x1, y1, x2, y2]


class IndexedObject(BaseModel):
    """One physical object found by Grok in a single image or tile."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, description="Short noun phrase for the object")
    description: str = Field(
        description="Color, brand, visible text, and neighboring objects"
    )
    box: list[float] = Field(
        description="Tight box [x1, y1, x2, y2] normalized 0-1000, origin at the top-left",
        min_length=4,
        max_length=4,
    )
    count: int = Field(ge=1, description="How many of this object are inside the box")
    drug_name: str | None = Field(
        description="Medication name when the object is a drug package, otherwise null"
    )
    expiry_text: str | None = Field(
        description="Visible expiry text, otherwise null"
    )

    @field_validator("box")
    @classmethod
    def box_in_range(cls, value: list[float]) -> list[float]:
        return _check_norm_box(value)


class IndexResponse(BaseModel):
    """Structured output for one Grok vision call."""

    model_config = ConfigDict(extra="forbid")

    objects: list[IndexedObject] = Field(
        description="Every distinct physical object visible in the image"
    )


class Frame(BaseModel):
    """One angle-tagged capture. The image array is attached by ingest, not stored in JSON."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    pan_deg: float
    tilt_deg: float
    timestamp: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Detection(BaseModel):
    """One object observation in full-frame pixels, before catalog dedupe."""

    model_config = ConfigDict(extra="forbid")

    label: str
    description: str
    bbox_px: list[int] = Field(min_length=4, max_length=4)
    grok_bbox_px: list[int] | None = None
    count: int = Field(ge=1)
    drug_name: str | None = None
    expiry_text: str | None = None
    confidence: float = Field(ge=0, le=1)
    box_source: Literal["grok", "detector"]
    frame_file: str
    pan_deg: float
    tilt_deg: float
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    timestamp: str | None = None


class CatalogObject(BaseModel):
    """A deduped object the query stage can point at."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    label: str
    description: str
    count: int = Field(ge=1)
    drug_name: str | None = None
    expiry_text: str | None = None
    azimuth_deg: float
    elevation_deg: float
    confidence: float = Field(ge=0, le=1)
    frame_file: str
    bbox_px: list[int] = Field(min_length=4, max_length=4)
    grok_bbox_px: list[int] | None = None
    box_source: Literal["grok", "detector"] = "grok"
    last_seen: str | None = None
    pan_deg: float
    tilt_deg: float
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)


class FrameMemory(BaseModel):
    """Detections remembered for one photo so an unchanged file is not uploaded again."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    fingerprint: str
    detections: list[Detection]


class Catalog(BaseModel):
    """Written to data/out/catalog.json. frames is the per-photo object memory."""

    model_config = ConfigDict(extra="forbid")

    objects: list[CatalogObject]
    created_at: str
    scan_dir: str | None = None
    fingerprint: str | None = None
    frames: list[FrameMemory] = Field(default_factory=list)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None


class InventoryMatch(BaseModel):
    """Clinic-closet match against data/inventory.json. Absent outside clinic mode."""

    model_config = ConfigDict(extra="forbid")

    name: str
    count: int | None = None
    expiry: str | None = None
    resource_links: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=100)


class ObjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int | None = None
    drug_name: str | None = None
    expiry_text: str | None = None
    last_seen: str | None = None
    inventory: InventoryMatch | None = None


class QueryDecision(BaseModel):
    """Structured answer from the reasoning model. Geometry is filled in afterwards."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["found", "ambiguous", "not_found"]
    object_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str
    candidates: list[Candidate] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    """Crop check: is this the object, and where is the tight box on the crop."""

    model_config = ConfigDict(extra="forbid")

    is_match: bool
    box: list[float] | None = None
    reason: str

    @field_validator("box")
    @classmethod
    def box_in_range(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        return _check_norm_box(value)


class QueryResult(BaseModel):
    """JSON contract consumed by the turret teammate. range_m stays null."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["found", "ambiguous", "not_found"]
    object_id: str | None = None
    label: str | None = None
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    range_m: None = None
    confidence: float = Field(ge=0, le=1)
    frame_file: str | None = None
    bbox_px: list[int] | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    metadata: ObjectMetadata = Field(default_factory=ObjectMetadata)
    reason: str


def norm_box_to_pixels(
    box: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Map a 0-1000 box onto pixel coordinates, clamped to the image."""
    x1, y1, x2, y2 = box
    px1 = _clamp(round(x1 / 1000 * width), 0, width - 1)
    py1 = _clamp(round(y1 / 1000 * height), 0, height - 1)
    px2 = _clamp(round(x2 / 1000 * width), 0, width - 1)
    py2 = _clamp(round(y2 / 1000 * height), 0, height - 1)
    if px2 <= px1:
        px2 = min(width - 1, px1 + 1)
    if py2 <= py1:
        py2 = min(height - 1, py1 + 1)
    return px1, py1, px2, py2


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
