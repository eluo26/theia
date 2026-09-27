"""Run a DroidCam scan and locate the asked-for object with xAI vision."""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from camera import CameraError
from scan import (
    ScanStoreError,
    capture_scan,
    default_scans_root,
    load_saved_scan,
    save_scan,
)
from vision.aim import search_frames
from vision.config import MissingAPIKeyError
from vision.grok_client import GrokCallError, redact
from vision.ingest import IngestError

logger = logging.getLogger("object_search")

_BOX_COLOR = (64, 140, 32)
_CIRCLE_COLOR = (37, 99, 235)


def public_search(search):
    """JSON for the UI. Binary preview bytes stay on the server."""
    snapshot = {key: value for key, value in search.items() if key != "preview_jpeg"}
    snapshot["has_preview"] = bool(search.get("preview_jpeg"))
    return snapshot


def public_detection(result, located):
    aim = located.get("aim") if isinstance(located.get("aim"), dict) else None
    return {
        "status": result.get("status"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "bbox_px": result.get("bbox_px"),
        "center_px": result.get("center_px"),
        "image_width": result.get("image_width"),
        "image_height": result.get("image_height"),
        "azimuth_deg": None if aim is None else aim.get("azimuth_deg"),
        "elevation_deg": None if aim is None else aim.get("elevation_deg"),
    }


def annotate_detection(jpeg_bytes, bbox_px, label=None):
    """Draw the pixel box and center mark on the chosen scan frame."""
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jpeg_bytes
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox_px)
    except (TypeError, ValueError):
        return jpeg_bytes
    cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, 3)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = max(int(round(math.hypot(x2 - x1, y2 - y1) / 2)), 12)
    cv2.circle(image, center, radius, _CIRCLE_COLOR, 3)
    cv2.drawMarker(image, center, _BOX_COLOR, cv2.MARKER_CROSS, 28, 2)
    if label:
        text_origin = (x1, max(28, y1 - 12))
        cv2.putText(
            image,
            str(label),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return encoded.tobytes() if ok else jpeg_bytes


def run_object_search(
    search_id, query, update, *, capture=None, locate=None, save_dir=None, replay_dir=None
):
    """Capture a new handheld scan, then ask the vision module where the object is.

    Every camera search takes a fresh set of photos. replay_dir loads one
    folder and skips the camera.
    """
    using_camera = capture is None and replay_dir is None
    capture = capture_scan if capture is None else capture
    locate = search_frames if locate is None else locate

    loading_saved = replay_dir is not None

    def on_frame(taken, total, *_rest):
        action = "Loading saved photo" if loading_saved else "Capturing photo"
        update(
            search_id,
            photos_taken=taken,
            total_photos=total,
            detail=f"{action} {taken} of {total}…",
            progress_pct=int(round(100 * taken / max(total, 1))),
        )

    saved_dir = Path(replay_dir) if replay_dir is not None else None
    try:
        if replay_dir is not None:
            frames = load_saved_scan(replay_dir)
            total = len(frames)
            for index, frame in enumerate(frames, start=1):
                on_frame(index, total, frame, None)
        else:
            frames = capture(on_frame=on_frame)
    except (CameraError, ScanStoreError) as exc:
        update(search_id, stage="failed", failed_at="scanning", error=str(exc), detail=None)
        return

    if not frames:
        update(
            search_id,
            stage="failed",
            failed_at="scanning",
            error="The camera scan did not produce any photos.",
            detail=None,
        )
        return

    frame_count = len(frames)
    if saved_dir is None and (using_camera or save_dir is not None):
        root = default_scans_root() if save_dir is None else Path(save_dir)
        try:
            saved = save_scan(frames, root, query=query)
        except ScanStoreError as exc:
            logger.warning("could not save scan: %s", exc)
        else:
            saved_dir = saved
            print(f"Saved scan: {saved}")
            update(search_id, saved_scan=str(saved))
    if not update(
        search_id,
        stage="searching",
        photos_taken=frame_count,
        total_photos=frame_count,
        photos_checked=0,
        detail="Looking at the photos…",
        progress_pct=0,
    ):
        return

    def on_progress(event):
        phase = event.get("phase")
        done = int(event.get("done") or 0)
        total = max(int(event.get("total") or 0), 1)
        if phase == "indexing":
            update(
                search_id,
                photos_checked=min(done, frame_count),
                detail=f"Looking at photo {min(done, total)} of {total}…",
                progress_pct=int(round(100 * done / total * 0.85)),
            )
        elif phase == "matching":
            update(
                search_id,
                photos_checked=frame_count,
                detail="Matching your question…",
                progress_pct=int(round(85 + 15 * done / total)),
            )

    try:
        located = locate(frames, query, on_progress=on_progress, scan_dir=saved_dir)
    except TypeError:
        try:
            located = locate(frames, query, on_progress=on_progress)
        except TypeError:
            # Test doubles may not accept on_progress or scan_dir.
            located = locate(frames, query)
    except MissingAPIKeyError as exc:
        update(search_id, stage="failed", failed_at="searching", error=str(exc), detail=None)
        return
    except (IngestError, GrokCallError) as exc:
        update(
            search_id,
            stage="failed",
            failed_at="searching",
            error=redact(str(exc)),
            detail=None,
        )
        return
    except Exception:
        logger.exception("Vision processing failed for search %s", search_id)
        update(
            search_id,
            stage="failed",
            failed_at="searching",
            error="Vision processing failed.",
            detail=None,
        )
        return

    result = located.get("result") if isinstance(located.get("result"), dict) else {}
    detection = public_detection(result, located)
    preview = _preview_jpeg(located, result)
    found = located.get("fire_laser") is True and result.get("status") == "found"
    if not found:
        update(
            search_id,
            stage="failed",
            failed_at="searching",
            photos_checked=frame_count,
            progress_pct=100,
            detail=None,
            error=result.get("reason") or "Couldn't find the object.",
            detection=detection,
            preview_jpeg=preview,
        )
        return

    update(
        search_id,
        stage="on_target",
        photos_checked=frame_count,
        progress_pct=100,
        detail=None,
        distance_px=_center_offset_px(detection),
        detection=detection,
        preview_jpeg=preview,
        error=None,
    )


def _preview_jpeg(located, result):
    frame = located.get("frame_jpeg")
    if not isinstance(frame, (bytes, bytearray)):
        return None
    bbox = result.get("bbox_px")
    if isinstance(bbox, list) and len(bbox) == 4:
        return annotate_detection(frame, bbox, result.get("label"))
    return bytes(frame)


def _center_offset_px(detection):
    center = detection.get("center_px")
    width = detection.get("image_width")
    height = detection.get("image_height")
    if not (isinstance(center, list) and len(center) == 2):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    try:
        cx, cy = float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None
    return int(round(math.hypot(cx - width / 2, cy - height / 2)))
