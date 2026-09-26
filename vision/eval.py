"""Score locate() against a scene's ground_truth.json.

Each row is {query, expected_label, az, el}. The report is the correct-object
rate and the mean angular error in degrees over the correct rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vision.config import REPO_ROOT, Settings, load_settings
from vision.geometry import angular_distance_deg
from vision.index import catalog_for_query
from vision.ingest import IngestError
from vision.matching import token_ratio
from vision.query import locate
from vision.schemas import QueryResult


class EvalError(ValueError):
    """The scene has no usable ground truth."""


class GroundTruthItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    expected_label: str
    az: float
    el: float


class EvalRow(BaseModel):
    query: str
    expected_label: str
    expected_az: float
    expected_el: float
    status: str
    label: str | None = None
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    correct: bool
    angular_error_deg: float | None = None


class EvalReport(BaseModel):
    scene: str
    queries: int
    correct: int
    correct_object_rate: float
    mean_angular_error_deg: float | None
    rows: list[EvalRow]


def resolve_scene(scene: str) -> Path:
    path = Path(scene)
    if path.is_dir():
        return path
    candidate = REPO_ROOT / "data" / "test_scenes" / scene
    if candidate.is_dir():
        return candidate
    raise IngestError(
        f"Scene not found: {scene}. Pass a directory or a name under data/test_scenes/."
    )


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    if not path.is_file():
        raise EvalError(f"Missing ground truth: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("queries", raw.get("items"))
    if not isinstance(raw, list):
        raise EvalError(f"{path} must be a list of {{query, expected_label, az, el}}.")
    return [GroundTruthItem.model_validate(item) for item in raw]


def evaluate_scene(
    scene: str | Path,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    use_cache: bool = True,
) -> EvalReport:
    settings = settings if settings is not None else load_settings()
    scene_dir = resolve_scene(str(scene)) if not isinstance(scene, Path) else (
        scene if scene.is_dir() else resolve_scene(str(scene))
    )
    items = load_ground_truth(scene_dir / "ground_truth.json")
    if not items:
        raise EvalError(
            f"{scene_dir / 'ground_truth.json'} has no queries. "
            "describe, index, and ask do not use this file. "
            "After a real scan, add objects you measured as "
            '{query, expected_label, az, el}.'
        )
    catalog = catalog_for_query(
        scene_dir,
        settings=settings,
        client_factory=client_factory,
        detector=detector,
        use_cache=use_cache,
    )
    results = [
        locate(
            item.query,
            catalog,
            settings=settings,
            client_factory=client_factory,
            detector=detector,
            use_cache=use_cache,
        )
        for item in items
    ]
    return score_scene(scene_dir.name, items, results, settings.label_sim)


def score_scene(
    scene: str,
    items: list[GroundTruthItem],
    results: list[QueryResult],
    label_sim: float,
) -> EvalReport:
    rows = [
        score_row(item, result, label_sim)
        for item, result in zip(items, results)
    ]
    correct = sum(1 for row in rows if row.correct)
    errors = [row.angular_error_deg for row in rows if row.correct and row.angular_error_deg is not None]
    mean = sum(errors) / len(errors) if errors else None
    queries = len(rows)
    rate = correct / queries if queries else 0.0
    return EvalReport(
        scene=scene,
        queries=queries,
        correct=correct,
        correct_object_rate=rate,
        mean_angular_error_deg=mean,
        rows=rows,
    )


def score_row(item: GroundTruthItem, result: QueryResult, label_sim: float) -> EvalRow:
    correct = result.status == "found" and result.label is not None and labels_match(
        result.label, item.expected_label, label_sim
    )
    error = None
    if correct and result.azimuth_deg is not None and result.elevation_deg is not None:
        error = angular_distance_deg(result.azimuth_deg, result.elevation_deg, item.az, item.el)
    return EvalRow(
        query=item.query,
        expected_label=item.expected_label,
        expected_az=item.az,
        expected_el=item.el,
        status=result.status,
        label=result.label,
        azimuth_deg=result.azimuth_deg,
        elevation_deg=result.elevation_deg,
        correct=correct,
        angular_error_deg=error,
    )


def labels_match(found: str, expected: str, label_sim: float) -> bool:
    if found.casefold() == expected.casefold():
        return True
    if expected.casefold() in found.casefold() or found.casefold() in expected.casefold():
        return True
    return token_ratio(found, expected) >= label_sim
