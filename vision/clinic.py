"""Clinic-closet helpers. Fuzzy-match a package name against data/inventory.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vision.config import REPO_ROOT
from vision.matching import token_ratio
from vision.schemas import InventoryMatch

logger = logging.getLogger("vision.clinic")

DEFAULT_INVENTORY = REPO_ROOT / "data" / "inventory.json"


def match_inventory(
    drug_name: str | None,
    *,
    threshold: float,
    inventory_path: Path | None = None,
) -> InventoryMatch | None:
    if not drug_name:
        return None
    path = inventory_path if inventory_path is not None else DEFAULT_INVENTORY
    items = _load_items(path)
    best: InventoryMatch | None = None
    for item in items:
        name = str(item.get("drug_name") or item.get("name") or "").strip()
        if not name:
            continue
        score = token_ratio(drug_name, name)
        if score < threshold:
            continue
        if best is not None and score <= best.score:
            continue
        links = item.get("resource_links") or []
        best = InventoryMatch(
            name=name,
            count=item.get("count"),
            expiry=item.get("expiry"),
            resource_links=[str(link) for link in links],
            score=score,
        )
    return best


def _load_items(path: Path) -> list[dict]:
    if not path.is_file():
        logger.warning("clinic inventory not found at %s", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("clinic inventory %s is not valid JSON: %s", path, exc)
        return []
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload
    if not isinstance(items, list):
        logger.warning("clinic inventory %s has no items list", path)
        return []
    return [item for item in items if isinstance(item, dict)]
