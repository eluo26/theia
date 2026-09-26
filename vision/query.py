"""Answer a question about a catalog.

Public entry point: locate(query, catalog) -> QueryResult.
range_m is always null.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from vision import detector as detector_lib
from vision import grok_client
from vision.clinic import match_inventory
from vision.config import REPO_ROOT, Settings, load_settings, require_api_key
from vision.detector import tighter_agreeing_box
from vision.geometry import box_center_angles
from vision.grok_client import GrokCallError, redact
from vision.matching import match_text, same_label_family, side_view_core, token_ratio
from vision.preprocess import encode_png
from vision.references import reference_note
from vision.schemas import (
    Candidate,
    Catalog,
    CatalogObject,
    ObjectMetadata,
    QueryDecision,
    QueryResult,
    VerifyResponse,
    norm_box_to_pixels,
)

logger = logging.getLogger("vision.query")

# Spatial questions name a relation, not one catalog label. Examples from the prompt.
_RELATIONAL_PHRASES = (
    "next to",
    "left of",
    "right of",
    "beside",
    "in front of",
    "behind the",
    "near the",
    "the thing",
)

QUERY_PROMPT = """You are the query module for a tabletop laser turret.
The user is looking for an object. The catalog lists what the camera saw.
Pick the catalog entry that answers the question.

Handle all of these:
- direct names, such as "blue water bottle" or "Jardiance"
- relational questions, such as "the thing left of the lamp" or "next to the mug", using the descriptions
- category questions, such as "something to drink" or "the diabetes medication", using labels, descriptions, and drug names

status:
- found: one catalog object answers the question
- ambiguous: more than one catalog object could answer it; put the best ones in candidates
- not_found: nothing in the catalog answers it

object_id must be an id from the catalog, or null when nothing matches.
candidates: up to 3 catalog objects, best first. Each needs object_id, label, confidence, and reason.
confidence: from 0 to 1.
reason: one short sentence.

Catalog:
{catalog}

Question: {query}
"""

VERIFY_PROMPT = """Is this the {label}?
If it is, set is_match true and return a tight box around that object only.
The box is [x1, y1, x2, y2], normalized from 0 to 1000, origin at the top-left.
If this image is not that object, set is_match false and box to null.
reason: one short sentence.
"""


def locate(
    query: str,
    catalog: Catalog,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    use_cache: bool = True,
) -> QueryResult:
    """Ask which catalog object the question refers to, then tighten its box."""
    settings = settings if settings is not None else load_settings()
    require_api_key(settings)
    if not catalog.objects:
        return _empty("not_found", 0.0, "The catalog has no objects.")

    direct = _direct_catalog_decision(query, catalog, settings)
    if direct is not None:
        decision = direct
        candidates = _known_candidates(decision.candidates, catalog)[:3]
        check_crop = False
        logger.info("catalog label match; skipping the text model")
    else:
        decision, _cache = grok_client.parse_text(
            QUERY_PROMPT.format(catalog=catalog_text(catalog), query=query),
            settings,
            response_model=QueryDecision,
            model=settings.active_query_model,
            client_factory=client_factory,
            use_cache=use_cache,
        )
        candidates = _known_candidates(decision.candidates, catalog)[:3]
        decision = _collapse_same_kind(decision, candidates, catalog, settings)
        check_crop = settings.verify_match
    chosen = _resolve(decision, catalog)
    if decision.status == "not_found" or chosen is None:
        return QueryResult(
            status="not_found" if chosen is None and decision.status == "found" else decision.status,
            object_id=None if chosen is None else chosen.object_id,
            label=None,
            azimuth_deg=None,
            elevation_deg=None,
            range_m=None,
            confidence=decision.confidence,
            frame_file=None,
            bbox_px=None,
            candidates=candidates,
            metadata=ObjectMetadata(),
            reason=_missing_reason(decision, chosen),
        )

    bbox = list(chosen.bbox_px)
    azimuth = chosen.azimuth_deg
    elevation = chosen.elevation_deg
    confidence = decision.confidence
    reason = decision.reason
    box_source = chosen.box_source
    refined = None
    if check_crop:
        chosen_detector = detector_lib.resolve_detector(settings, detector)
        try:
            refined = _verify_crop(
                chosen,
                catalog,
                settings,
                client_factory=client_factory,
                detector=chosen_detector,
                use_cache=use_cache,
            )
        except GrokCallError as exc:
            logger.warning("crop check failed for %s: %s", chosen.object_id, redact(str(exc)))
            refined = None
    if refined is not None:
        bbox, azimuth, elevation, box_source, verify_reason, matched = refined
        if verify_reason:
            reason = f"{reason} {verify_reason}".strip()
        if not matched:
            confidence = min(confidence, confidence * 0.5)

    metadata = ObjectMetadata(
        count=chosen.count,
        drug_name=chosen.drug_name,
        expiry_text=chosen.expiry_text,
        last_seen=chosen.last_seen,
    )
    if settings.clinic_mode:
        metadata.inventory = match_inventory(chosen.drug_name, threshold=settings.label_sim)
        note = reference_note(REPO_ROOT / "data" / "references")
        if note and note not in reason:
            reason = f"{reason} {note}".strip()

    return QueryResult(
        status=decision.status if decision.status != "found" or chosen is not None else "not_found",
        object_id=chosen.object_id,
        label=chosen.label,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        range_m=None,
        confidence=confidence,
        frame_file=chosen.frame_file,
        bbox_px=bbox,
        candidates=candidates,
        metadata=metadata,
        reason=reason,
    )


def catalog_text(catalog: Catalog) -> str:
    """The only catalog fields the reasoning model is allowed to see."""
    blocks = []
    for obj in catalog.objects:
        drug = obj.drug_name if obj.drug_name else "null"
        blocks.append(
            "\n".join(
                [
                    f"id: {obj.object_id}",
                    f"label: {obj.label}",
                    f"description: {obj.description}",
                    f"count: {obj.count}",
                    f"drug_name: {drug}",
                ]
            )
        )
    return "\n\n".join(blocks)


def public_query_dict(result: QueryResult) -> dict:
    """JSON for the turret teammate. range_m stays null. inventory is omitted when unset."""
    payload = result.model_dump()
    if payload["metadata"].get("inventory") is None:
        payload["metadata"].pop("inventory", None)
    return payload


def _verify_crop(obj, catalog, settings, *, client_factory, detector, use_cache):
    path = _frame_path(catalog, obj)
    if path is None or not path.is_file():
        logger.warning("frame %s is missing; keeping the catalog box", obj.frame_file)
        return None
    from vision.ingest import load_rgb

    image = load_rgb(path)
    crop, left, top = _crop(image, obj.bbox_px, settings.crop_pad)
    parsed, _cache = grok_client.parse_image_bytes(
        encode_png(crop),
        "image/png",
        settings,
        prompt=VERIFY_PROMPT.format(label=obj.label),
        response_model=VerifyResponse,
        model=settings.fast_model,
        client_factory=client_factory,
        use_cache=use_cache,
    )
    crop_h, crop_w = crop.shape[:2]
    frame_h, frame_w = image.shape[:2]
    grok_full = None
    if parsed.is_match and parsed.box is not None:
        local = norm_box_to_pixels(parsed.box, crop_w, crop_h)
        grok_full = _shift_crop(local, left, top, frame_w, frame_h)
    detector_full = None
    hits = detector.detect(crop, obj.label)
    if hits and grok_full is not None:
        local_grok = norm_box_to_pixels(parsed.box, crop_w, crop_h)
        best = None
        best_iou = -1.0
        from vision.detector import box_iou

        for hit in hits:
            overlap = box_iou(local_grok, hit.bbox_px)
            if overlap > best_iou:
                best = hit
                best_iou = overlap
        if best is not None and best_iou >= settings.refine_iou:
            detector_full = _shift_crop(best.bbox_px, left, top, frame_w, frame_h)
    chosen, source = tighter_agreeing_box(grok_full, detector_full, settings.refine_iou)
    if chosen is None:
        return (
            list(obj.bbox_px),
            obj.azimuth_deg,
            obj.elevation_deg,
            obj.box_source,
            parsed.reason,
            parsed.is_match,
        )
    azimuth, elevation = box_center_angles(
        chosen, frame_w, frame_h, obj.pan_deg, obj.tilt_deg, settings
    )
    return chosen, azimuth, elevation, source, parsed.reason, parsed.is_match


def _crop(image, bbox, pad: float):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    left = max(0, math.floor(x1 - pad * box_w))
    top = max(0, math.floor(y1 - pad * box_h))
    right = min(width, math.ceil(x2 + pad * box_w))
    bottom = min(height, math.ceil(y2 + pad * box_h))
    if right <= left or bottom <= top:
        raise GrokCallError("The crop around the catalog box is empty.")
    return image[top:bottom, left:right].copy(), left, top


def _shift_crop(box, left, top, width, height) -> list[int]:
    from vision.preprocess import shift_box

    return shift_box(box, left, top, width, height)


def _frame_path(catalog: Catalog, obj: CatalogObject) -> Path | None:
    raw = Path(obj.frame_file)
    if raw.is_file():
        return raw
    if catalog.scan_dir:
        candidate = Path(catalog.scan_dir) / obj.frame_file
        if candidate.is_file():
            return candidate
    return raw if raw.is_absolute() else None


def catalog_object_map(catalog: Catalog) -> dict[str, CatalogObject]:
    return {obj.object_id: obj for obj in catalog.objects}


def _collapse_same_kind(
    decision: QueryDecision,
    candidates: list,
    catalog: Catalog,
    settings: Settings,
) -> QueryDecision:
    """Aim at the best view when one kind of object was recognized more than once.

    Two photos of a laptop used to stay ambiguous, so the laser would not point.
    Same labels (the configured rapidfuzz threshold) collapse to the
    highest-confidence candidate. A left view and a right view of one object
    count as the same label. Different objects stay ambiguous.
    """
    if decision.status != "ambiguous" or not _same_kind(candidates, catalog, settings.label_sim):
        return decision
    by_id = catalog_object_map(catalog)
    best = max(candidates, key=lambda item: (item.confidence, by_id[item.object_id].confidence))
    note = "Same kind of object in more than one photo; aiming at the highest-confidence view."
    reason = decision.reason if note in decision.reason else f"{decision.reason} {note}".strip()
    logger.info("collapsed %s same-kind candidates to %s", len(candidates), best.object_id)
    return decision.model_copy(
        update={
            "status": "found",
            "object_id": best.object_id,
            "confidence": best.confidence,
            "reason": reason,
        }
    )


def _same_kind(candidates, catalog: Catalog, label_sim: float) -> bool:
    if len(candidates) < 2:
        return False
    by_id = catalog_object_map(catalog)
    if len({item.object_id for item in candidates}) == 1:
        return True
    labels = [by_id[item.object_id].label for item in candidates]
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if not same_label_family(left, right, label_sim):
                return False
    return True


def _is_relational(query: str) -> bool:
    text = match_text(query).casefold()
    return any(phrase in text for phrase in _RELATIONAL_PHRASES)


def _name_score(query: str, name: str) -> float:
    return token_ratio(match_text(query), match_text(name))


def _object_names(obj: CatalogObject) -> list[str]:
    names = [obj.label]
    if obj.drug_name:
        names.append(obj.drug_name)
    core = side_view_core(obj.label)
    if core.casefold() != obj.label.casefold():
        names.append(core)
    return names


def _object_score(query: str, obj: CatalogObject) -> float:
    return max(_name_score(query, name) for name in _object_names(obj))


def _objects_same_family(objects: list[CatalogObject], label_sim: float) -> bool:
    labels = [obj.label for obj in objects]
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if not same_label_family(left, right, label_sim):
                return False
    return True


def _direct_catalog_decision(
    query: str,
    catalog: Catalog,
    settings: Settings,
) -> QueryDecision | None:
    """Answer from catalog labels when the question names them.

    Relational questions and category questions that do not literally match a
    label or drug name return None so the text model still runs.
    """
    if _is_relational(query):
        return None
    hits = [obj for obj in catalog.objects if _object_score(query, obj) >= settings.label_sim]
    if not hits:
        return None
    ranked = sorted(hits, key=lambda obj: obj.confidence, reverse=True)
    candidates = [
        Candidate(
            object_id=obj.object_id,
            label=obj.label,
            confidence=obj.confidence,
            reason="The question names this catalog label.",
        )
        for obj in ranked[:3]
    ]
    if not _objects_same_family(hits, settings.label_sim):
        return QueryDecision(
            status="ambiguous",
            object_id=None,
            confidence=ranked[0].confidence,
            reason="More than one different catalog object matches the question.",
            candidates=candidates,
        )
    best = ranked[0]
    reason = "Matched the catalog label."
    if len(hits) > 1:
        reason = (
            f"{reason} Same kind of object in more than one photo; "
            "aiming at the highest-confidence view."
        )
    return QueryDecision(
        status="found",
        object_id=best.object_id,
        confidence=best.confidence,
        reason=reason,
        candidates=candidates,
    )


def _resolve(decision: QueryDecision, catalog: Catalog) -> CatalogObject | None:
    by_id = catalog_object_map(catalog)
    if decision.object_id and decision.object_id in by_id:
        return by_id[decision.object_id]
    if decision.status == "ambiguous":
        for candidate in decision.candidates:
            if candidate.object_id in by_id:
                return by_id[candidate.object_id]
    return None


def _known_candidates(candidates, catalog: Catalog):
    by_id = catalog_object_map(catalog)
    kept = []
    for candidate in candidates:
        if candidate.object_id not in by_id:
            continue
        if not candidate.label:
            candidate = candidate.model_copy(update={"label": by_id[candidate.object_id].label})
        kept.append(candidate)
    return kept


def _missing_reason(decision: QueryDecision, chosen) -> str:
    if chosen is None and decision.object_id:
        return f"{decision.reason} The id {decision.object_id} is not in the catalog."
    return decision.reason


def _empty(status: str, confidence: float, reason: str) -> QueryResult:
    return QueryResult(
        status=status,  # type: ignore[arg-type]
        object_id=None,
        label=None,
        azimuth_deg=None,
        elevation_deg=None,
        range_m=None,
        confidence=confidence,
        frame_file=None,
        bbox_px=None,
        candidates=[],
        metadata=ObjectMetadata(),
        reason=reason,
    )
