"""One function the UI and hardware teammates can call. No web framework.

The API key stays in this process. answer() uses build_catalog / locate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from vision.config import Settings, load_settings
from vision.index import catalog_for_query
from vision.query import locate, public_query_dict
from vision.schemas import CatalogObject


def answer(
    scan_dir: str | Path,
    query: str,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    sweep: tuple[float, float, float] | None = None,
    angles_csv: str | Path | None = None,
    use_cache: bool = True,
    save_debug: bool | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Index a scan if needed, answer the question, and say whether to fire.

    fire_laser is true only when the query status is "found".
    aim is {azimuth_deg, elevation_deg} when firing, and null otherwise.
    """
    settings = settings if settings is not None else load_settings()
    catalog = catalog_for_query(
        scan_dir,
        settings=settings,
        client_factory=client_factory,
        detector=detector,
        sweep=sweep,
        angles_csv=angles_csv,
        use_cache=use_cache,
        save_debug=save_debug,
        on_progress=on_progress,
        query=query,
    )
    if on_progress is not None:
        on_progress({"phase": "matching", "done": 0, "total": 1})
    result = locate(
        query,
        catalog,
        settings=settings,
        client_factory=client_factory,
        detector=detector,
        use_cache=use_cache,
    )
    if on_progress is not None:
        on_progress({"phase": "matching", "done": 1, "total": 1})
    fire_laser = result.status == "found"
    aim = None
    if fire_laser:
        aim = {
            "azimuth_deg": result.azimuth_deg,
            "elevation_deg": result.elevation_deg,
        }
    return {
        "fire_laser": fire_laser,
        "aim": aim,
        "result": public_query_dict(result),
        "items": [public_item_dict(obj) for obj in catalog.objects],
    }


def public_item_dict(obj: CatalogObject) -> dict:
    """One catalog object for the UI. id matches result.object_id."""
    payload = obj.model_dump()
    payload["id"] = obj.object_id
    return payload
