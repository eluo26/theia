"""Build a catalog from a scan directory.

Public entry point: build_catalog(scan_dir) -> Catalog.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from vision import detector as detector_lib
from vision import grok_client
from vision.config import Settings, load_settings
from vision.detector import GROK_ONLY_CONFIDENCE, refine_box
from vision.geometry import angular_distance_deg, box_center_angles
from vision.grok_client import GrokCallError, redact
from vision.ingest import (
    IngestError,
    LoadedFrame,
    _IMAGE_SUFFIXES,
    load_scan,
    load_still_frames,
    no_images_message,
    still_frame_meta,
)
from vision.matching import token_ratio
from vision.preprocess import Tile, encode_jpeg, make_tiles, prepare_frame, shift_box
from vision.schemas import (
    Catalog,
    CatalogObject,
    Detection,
    FrameMemory,
    IndexResponse,
    norm_box_to_pixels,
)
from vision.viz import annotate_detections, legend_name

logger = logging.getLogger("vision.index")


def build_catalog(
    scan_dir: str | Path,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    sweep: tuple[float, float, float] | None = None,
    angles_csv: str | Path | None = None,
    use_cache: bool = True,
    save_debug: bool | None = None,
) -> Catalog:
    """Index angle-tagged photos (or one sweep video) and write catalog.json."""
    settings = settings if settings is not None else load_settings()
    source = Path(scan_dir)
    reused: list[Detection] = []
    frame_memories: list[FrameMemory] = []
    pending: list[tuple[str, str]] = []
    if source.is_dir():
        if sweep is not None or angles_csv is not None:
            raise IngestError(
                "--sweep and --angles-csv apply to a video file, not an image folder."
            )
        frames, reused, frame_memories, pending = _load_changed_stills(
            source, settings, use_cache=use_cache
        )
    else:
        frames = load_scan(
            source,
            out_dir=settings.out_path,
            video_sample_every_s=settings.video_sample_every_s,
            blur_threshold=settings.blur_threshold,
            sweep=sweep,
            angles_csv=angles_csv,
            max_edge=settings.api_max_edge,
        )
    _log_resolution_once(frames, settings)
    for frame in frames:
        prepare_frame(frame, settings)
    tiles = [tile for frame in frames for tile in make_tiles(frame, settings)]
    fresh: list[Detection] = []
    failed: set[str] = set()
    if tiles:
        chosen = detector_lib.resolve_detector(settings, detector)
        fresh, failed = _index_tiles(
            tiles,
            settings,
            client_factory=client_factory,
            detector=chosen,
            use_cache=use_cache,
        )
    if failed:
        fresh = [item for item in fresh if item.frame_file not in failed]
        pending = [item for item in pending if item[0] not in failed]
    if pending:
        frame_memories = _remember_frames(frame_memories, pending, fresh)
    detections = reused + fresh
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scan_path = source.resolve()
    objects = merge_detections(detections, settings, created_at=created_at)
    catalog = Catalog(
        objects=objects,
        created_at=created_at,
        scan_dir=str(scan_path),
        fingerprint=scan_fingerprint(source, settings, sweep=sweep, angles_csv=angles_csv),
        frames=sorted(frame_memories, key=lambda item: item.source_file),
    )
    write_catalog(catalog, settings.out_path / "catalog.json")
    if save_debug is None:
        save_debug = settings.save_debug
    if save_debug:
        _write_debug(
            frames,
            detections,
            objects,
            settings.out_path / "debug",
            legend=legend_name(settings.provider),
        )
    logger.info("catalog objects=%s path=%s", len(objects), settings.out_path / "catalog.json")
    return catalog


def merge_detections(
    detections: list[Detection],
    settings: Settings,
    *,
    created_at: str,
) -> list[CatalogObject]:
    """Merge observations with similar labels that sit close together in angle."""
    if not detections:
        return []
    parent = list(range(len(detections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if _should_merge(detections[i], detections[j], settings):
                union(i, j)

    groups: dict[int, list[Detection]] = {}
    for index, detection in enumerate(detections):
        groups.setdefault(find(index), []).append(detection)

    representatives: list[tuple[Detection, list[Detection]]] = []
    for group in groups.values():
        representatives.append((_representative(group), group))
    representatives.sort(
        key=lambda item: (
            item[0].azimuth_deg if item[0].azimuth_deg is not None else 0.0,
            item[0].elevation_deg if item[0].elevation_deg is not None else 0.0,
            item[0].label,
            item[0].frame_file,
        )
    )
    objects: list[CatalogObject] = []
    for number, (rep, group) in enumerate(representatives, start=1):
        objects.append(_to_object(f"obj_{number:03d}", rep, group, created_at))
    return objects


def write_catalog(catalog: Catalog, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def catalog_for_query(
    scan_dir: str | Path,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    sweep: tuple[float, float, float] | None = None,
    angles_csv: str | Path | None = None,
    use_cache: bool = True,
    save_debug: bool | None = None,
) -> Catalog:
    """Reuse catalog.json when the photos and indexing settings have not changed."""
    settings = settings if settings is not None else load_settings()
    source = Path(scan_dir)
    fingerprint = scan_fingerprint(source, settings, sweep=sweep, angles_csv=angles_csv)
    catalog_path = settings.out_path / "catalog.json"
    write_debug = settings.save_debug if save_debug is None else save_debug
    if use_cache and not write_debug and catalog_path.is_file():
        existing = _read_catalog(catalog_path)
        if existing is not None and existing.fingerprint == fingerprint:
            logger.info(
                "reusing catalog objects=%s path=%s",
                len(existing.objects),
                catalog_path,
            )
            return existing
    return build_catalog(
        source,
        settings=settings,
        client_factory=client_factory,
        detector=detector,
        sweep=sweep,
        angles_csv=angles_csv,
        use_cache=use_cache,
        save_debug=save_debug,
    )


def scan_fingerprint(
    source: Path,
    settings: Settings,
    *,
    sweep: tuple[float, float, float] | None,
    angles_csv: str | Path | None,
) -> str:
    """Hash the scan files and the settings that change what the catalog contains."""
    digest = hashlib.sha256()
    digest.update(settings.fast_model.encode("utf-8"))
    digest.update(settings.image_detail.encode("utf-8"))
    digest.update(repr(tuple(settings.tile_grid)).encode("utf-8"))
    digest.update(str(settings.api_max_edge).encode("utf-8"))
    digest.update(str(settings.max_objects_per_image).encode("utf-8"))
    digest.update(grok_client.index_prompt(settings.max_objects_per_image).encode("utf-8"))
    digest.update(repr(sweep).encode("utf-8"))
    if angles_csv is not None:
        _hash_file(digest, Path(angles_csv))
    path = Path(source)
    if path.is_file():
        _hash_file(digest, path)
    elif path.is_dir():
        digest.update(str(path.resolve()).encode("utf-8"))
        names = [path / "manifest.json"]
        names.extend(
            child
            for child in sorted(path.iterdir())
            if child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES
        )
        for child in names:
            if child.is_file():
                _hash_file(digest, child)
    return digest.hexdigest()


def _hash_file(digest, path: Path) -> None:
    if not path.is_file():
        return
    stat = path.stat()
    digest.update(path.name.encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))


def frame_fingerprint(
    path: Path,
    settings: Settings,
    *,
    pan: float,
    tilt: float,
    timestamp: str | None,
) -> str:
    """Identity of one photo plus the settings that change how it is indexed."""
    digest = hashlib.sha256()
    digest.update(settings.fast_model.encode("utf-8"))
    digest.update(settings.image_detail.encode("utf-8"))
    digest.update(repr(tuple(settings.tile_grid)).encode("utf-8"))
    digest.update(str(settings.api_max_edge).encode("utf-8"))
    digest.update(str(settings.max_objects_per_image).encode("utf-8"))
    digest.update(grok_client.index_prompt(settings.max_objects_per_image).encode("utf-8"))
    digest.update(repr((pan, tilt, timestamp)).encode("utf-8"))
    _hash_file(digest, path)
    return digest.hexdigest()


def _load_changed_stills(
    source: Path,
    settings: Settings,
    *,
    use_cache: bool,
) -> tuple[list[LoadedFrame], list[Detection], list[FrameMemory], list[tuple[str, str]]]:
    """Decode and return only stills whose file or angle tag is not already remembered."""
    meta = still_frame_meta(source)
    if not meta:
        raise IngestError(no_images_message(source))
    stored = _stored_frames(source, settings) if use_cache else {}
    reused: list[Detection] = []
    memories: list[FrameMemory] = []
    pending: list[tuple[str, str]] = []
    changed_meta = []
    for path, pan, tilt, timestamp in meta:
        fingerprint = frame_fingerprint(path, settings, pan=pan, tilt=tilt, timestamp=timestamp)
        previous = stored.get(path.name)
        if previous is not None and previous.fingerprint == fingerprint:
            reused.extend(previous.detections)
            memories.append(previous)
            continue
        pending.append((path.name, fingerprint))
        changed_meta.append((path, pan, tilt, timestamp))
    frames = load_still_frames(changed_meta, settings.api_max_edge) if changed_meta else []
    logger.info("frame memory reused=%s indexing=%s", len(memories), len(changed_meta))
    return frames, reused, memories, pending


def _stored_frames(source: Path, settings: Settings) -> dict[str, FrameMemory]:
    catalog = _read_catalog(settings.out_path / "catalog.json")
    if catalog is None or not catalog.scan_dir:
        return {}
    if Path(catalog.scan_dir) != source.resolve():
        return {}
    return {item.source_file: item for item in catalog.frames}


def _remember_frames(
    memories: list[FrameMemory],
    pending: list[tuple[str, str]],
    detections: list[Detection],
) -> list[FrameMemory]:
    by_file: dict[str, list[Detection]] = {}
    for detection in detections:
        by_file.setdefault(detection.frame_file, []).append(detection)
    remembered = list(memories)
    for name, fingerprint in pending:
        remembered.append(
            FrameMemory(
                source_file=name,
                fingerprint=fingerprint,
                detections=by_file.get(name, []),
            )
        )
    return remembered


def _read_catalog(path: Path) -> Catalog | None:
    try:
        return Catalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        logger.warning("catalog %s could not be reused: %s", path, redact(str(exc)))
        return None


def _index_tiles(
    tiles, settings, *, client_factory, detector, use_cache: bool
) -> tuple[list[Detection], set[str]]:
    if not tiles:
        return [], set()
    workers = min(settings.max_concurrency, len(tiles))
    detections: list[Detection] = []
    failed: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_safe_index_tile, tile, settings, client_factory, detector, use_cache)
            for tile in tiles
        ]
        for future in futures:
            frame_file, found, ok = future.result()
            if ok:
                detections.extend(found)
            else:
                failed.add(frame_file)
    if failed:
        detections = [item for item in detections if item.frame_file not in failed]
    return detections, failed


def _safe_index_tile(
    tile: Tile, settings, client_factory, detector, use_cache: bool
) -> tuple[str, list[Detection], bool]:
    frame_file = tile.frame.source_file
    try:
        return frame_file, _index_tile(tile, settings, client_factory, detector, use_cache), True
    except GrokCallError as exc:
        logger.warning(
            "skipping tile %s offset=%s,%s error=%s",
            frame_file,
            tile.offset_x,
            tile.offset_y,
            redact(str(exc)),
        )
        return frame_file, [], False


def _index_tile(tile: Tile, settings, client_factory, detector, use_cache: bool) -> list[Detection]:
    encoded = encode_jpeg(tile.image)
    parsed, _cache = grok_client.parse_image_bytes(
        encoded,
        "image/jpeg",
        settings,
        prompt=grok_client.index_prompt(settings.max_objects_per_image),
        response_model=IndexResponse,
        model=settings.fast_model,
        client_factory=client_factory,
        use_cache=use_cache,
    )
    frame = tile.frame
    ranked = sorted(parsed.objects, key=lambda obj: _norm_area(obj.box), reverse=True)
    detections: list[Detection] = []
    for obj in ranked[: settings.max_objects_per_image]:
        grok_tile = list(norm_box_to_pixels(obj.box, tile.width, tile.height))
        grok_full = _to_camera_box(grok_tile, tile.offset_x, tile.offset_y, tile)
        try:
            hits = detector.detect(tile.image, obj.label)
        except Exception as exc:  # a broken detector must not drop the vision box
            logger.warning("detector failed on %s: %s", frame.source_file, redact(str(exc)))
            hits = []
        chosen, source, confidence = refine_box(grok_tile, hits, settings.refine_iou)
        if source == "grok":
            confidence = GROK_ONLY_CONFIDENCE
            full = grok_full
        else:
            full = _to_camera_box(list(chosen), tile.offset_x, tile.offset_y, tile)
        camera_w = frame.camera_width
        camera_h = frame.camera_height
        azimuth, elevation = box_center_angles(
            full, camera_w, camera_h, frame.pan_deg, frame.tilt_deg, settings
        )
        detections.append(
            Detection(
                label=obj.label,
                description=obj.description,
                bbox_px=full,
                grok_bbox_px=grok_full,
                count=obj.count,
                drug_name=obj.drug_name,
                expiry_text=obj.expiry_text,
                confidence=confidence,
                box_source=source,
                frame_file=frame.source_file,
                pan_deg=frame.pan_deg,
                tilt_deg=frame.tilt_deg,
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                image_width=camera_w,
                image_height=camera_h,
                timestamp=frame.timestamp,
            )
        )
    return detections


def _norm_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _to_camera_box(box_px: list[int], offset_x: int, offset_y: int, tile: Tile) -> list[int]:
    """Map a box on the working tile onto the original photo."""
    frame = tile.frame
    src_w = max(1, frame.width)
    src_h = max(1, frame.height)
    dst_w = frame.camera_width
    dst_h = frame.camera_height
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    x1, y1, x2, y2 = box_px
    scaled = [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]
    return shift_box(
        scaled,
        int(round(offset_x * scale_x)),
        int(round(offset_y * scale_y)),
        dst_w,
        dst_h,
    )


def _should_merge(left: Detection, right: Detection, settings: Settings) -> bool:
    if left.azimuth_deg is None or right.azimuth_deg is None:
        return False
    if left.elevation_deg is None or right.elevation_deg is None:
        return False
    if token_ratio(left.label, right.label) < settings.label_sim:
        return False
    if not _drugs_compatible(left.drug_name, right.drug_name, settings.label_sim):
        return False
    separation = angular_distance_deg(
        left.azimuth_deg, left.elevation_deg, right.azimuth_deg, right.elevation_deg
    )
    return separation <= settings.merge_deg


def _drugs_compatible(left: str | None, right: str | None, threshold: float) -> bool:
    if not left or not right:
        return True
    return token_ratio(left, right) >= threshold


def _representative(group: list[Detection]) -> Detection:
    return min(group, key=_rep_key)


def _rep_key(detection: Detection) -> tuple:
    return (
        _center_distance(detection),
        -detection.confidence,
        detection.frame_file,
        tuple(detection.bbox_px),
    )


def _center_distance(detection: Detection) -> float:
    x1, y1, x2, y2 = detection.bbox_px
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    dx = (center_x - detection.image_width / 2) / detection.image_width
    dy = (center_y - detection.image_height / 2) / detection.image_height
    return (dx * dx + dy * dy) ** 0.5


def _to_object(object_id: str, rep: Detection, group: list[Detection], created_at: str) -> CatalogObject:
    if rep.azimuth_deg is None or rep.elevation_deg is None:
        raise ValueError(f"detection {rep.label} is missing azimuth or elevation")
    return CatalogObject(
        object_id=object_id,
        label=rep.label,
        description=rep.description,
        count=rep.count,
        drug_name=_prefer_text(rep, group, "drug_name"),
        expiry_text=_prefer_text(rep, group, "expiry_text"),
        azimuth_deg=rep.azimuth_deg,
        elevation_deg=rep.elevation_deg,
        confidence=rep.confidence,
        frame_file=rep.frame_file,
        bbox_px=list(rep.bbox_px),
        grok_bbox_px=list(rep.grok_bbox_px) if rep.grok_bbox_px is not None else None,
        box_source=rep.box_source,
        last_seen=rep.timestamp or created_at,
        pan_deg=rep.pan_deg,
        tilt_deg=rep.tilt_deg,
        image_width=rep.image_width,
        image_height=rep.image_height,
    )


def _prefer_text(rep: Detection, group: list[Detection], field: str) -> str | None:
    value = getattr(rep, field)
    if value:
        return value
    for detection in sorted(group, key=lambda item: item.frame_file):
        other = getattr(detection, field)
        if other:
            return other
    return None


def _write_debug(frames, detections, objects, dest: Path, *, legend: str) -> None:
    by_file: dict[str, list[Detection]] = {}
    for detection in detections:
        by_file.setdefault(detection.frame_file, []).append(detection)
    ids: dict[str, dict[tuple[int, ...], str]] = {}
    for obj in objects:
        ids.setdefault(obj.frame_file, {})[tuple(obj.bbox_px)] = obj.object_id
    dest.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        frame_ids = ids.get(frame.source_file, {})
        annotate_detections(
            frame.image,
            by_file.get(frame.source_file, []),
            dest / f"{Path(frame.source_file).stem}.png",
            ids=frame_ids,
            legend=legend,
        )


def _log_resolution_once(frames: list[LoadedFrame], settings: Settings) -> None:
    if not frames or None not in (settings.fx, settings.fy, settings.cx, settings.cy):
        return
    frame = frames[0]
    if frame.camera_width == settings.image_width and frame.camera_height == settings.image_height:
        return
    logger.info(
        "frame %s is %sx%s; config placeholders are %sx%s. FOV is applied to the actual frame.",
        frame.source_file,
        frame.camera_width,
        frame.camera_height,
        settings.image_width,
        settings.image_height,
    )
