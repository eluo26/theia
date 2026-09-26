"""Timed still capture from the shared DroidCam stream."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from camera import CameraError, SharedCamera

# 7 stills: a 180 degree arc sampled every 30 degrees. 3.5 / 0.5 rounds to 7.
DEFAULT_SCAN_SECONDS = 3.5
DEFAULT_SCAN_INTERVAL_S = 0.5

_camera = None


def get_camera():
    """One SharedCamera for the process. DroidCam only allows one viewer."""
    global _camera
    if _camera is None:
        _camera = SharedCamera()
    return _camera


def scan_photo_count(duration_s=None, interval_s=None):
    duration_s, interval_s = _scan_timing(duration_s, interval_s)
    return max(1, int(round(duration_s / interval_s)))


def encode_jpeg(frame, quality=90):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise CameraError("Could not encode a camera frame as JPEG.")
    return encoded.tobytes()


def capture_scan(camera=None, duration_s=None, interval_s=None, on_frame=None):
    """Take stills every interval_s for duration_s seconds.

    The default is 7 photos, the same count the turret will capture on a
    180 degree arc at 30 degree steps. Motors are not connected yet, so pan
    and tilt stay 0. A handheld phone sweep still gives fresh JPEGs.
    """
    duration_s, interval_s = _scan_timing(duration_s, interval_s)
    count = scan_photo_count(duration_s, interval_s)
    reader = camera if camera is not None else get_camera()
    reader.read(timeout=10.0)
    start = time.monotonic()
    frames = []
    for index in range(count):
        target = start + index * interval_s
        remaining = target - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        frame, _arrived = reader.read(newer_than=target)
        jpeg = encode_jpeg(frame)
        captured = {
            "image": jpeg,
            "pan": 0.0,
            "tilt": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        frames.append(captured)
        if on_frame is not None:
            on_frame(index + 1, count, captured, frame)
    return frames


class ScanStoreError(CameraError):
    """A saved scan folder could not be written or read."""


def default_scans_root():
    """Gitignored folder of past searches. Override with SEARCH_SCAN_DIR."""
    override = os.environ.get("SEARCH_SCAN_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / "scans"


def save_scan(frames, root=None, query=None):
    """Write one capture as JPEGs plus manifest.json for a later replay.

    The layout matches vision ingest: frame_000.jpg and
    manifest.json with file, pan, tilt, and timestamp.
    """
    if not frames:
        raise ScanStoreError("Cannot save a scan with no photos.")
    destination_root = default_scans_root() if root is None else Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = destination_root / stamp
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, frame in enumerate(frames):
        image = frame.get("image")
        if not isinstance(image, (bytes, bytearray)) or not bytes(image).startswith(b"\xff\xd8"):
            raise ScanStoreError(f"Frame {index} is not a JPEG.")
        name = f"frame_{index:03d}.jpg"
        (directory / name).write_bytes(bytes(image))
        entries.append(
            {
                "file": name,
                "pan": float(frame.get("pan") or 0.0),
                "tilt": float(frame.get("tilt") or 0.0),
                "timestamp": frame.get("timestamp"),
            }
        )
    payload = {"query": query, "frames": entries}
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def load_saved_scan(directory):
    """Read a folder written by save_scan back into capture_scan frame dicts."""
    folder = Path(directory)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ScanStoreError(f"No saved scan at {folder}.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScanStoreError(f"{manifest_path} is not valid JSON.") from exc
    entries = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ScanStoreError(f"{manifest_path} has no frames.")
    frames = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("file"):
            raise ScanStoreError(f"{manifest_path} frame {index} is missing a file name.")
        name = Path(entry["file"]).name
        image_path = folder / name
        if not image_path.is_file():
            raise ScanStoreError(f"Missing photo {name}.")
        image = image_path.read_bytes()
        if not image.startswith(b"\xff\xd8"):
            raise ScanStoreError(f"{name} is not a JPEG.")
        frames.append(
            {
                "image": image,
                "pan": float(entry.get("pan") or 0.0),
                "tilt": float(entry.get("tilt") or 0.0),
                "timestamp": entry.get("timestamp"),
            }
        )
    return frames


def resolve_saved_scan(raw):
    """Resolve a replay folder and keep it inside the saved-scans directory."""
    if not isinstance(raw, str) or not raw.strip():
        raise ScanStoreError("scan_dir must be a saved scan folder.")
    root = default_scans_root().resolve()
    requested = Path(raw.strip())
    folder = requested if requested.is_absolute() else root / requested
    folder = folder.resolve()
    if folder != root and root not in folder.parents:
        raise ScanStoreError("Scan folder must be inside the saved scans directory.")
    if folder == root or not (folder / "manifest.json").is_file():
        raise ScanStoreError(f"No saved scan at {folder.name}.")
    return folder


def _scan_timing(duration_s, interval_s):
    duration = DEFAULT_SCAN_SECONDS if duration_s is None else float(duration_s)
    interval = DEFAULT_SCAN_INTERVAL_S if interval_s is None else float(interval_s)
    if duration_s is None:
        duration = float(os.environ.get("SEARCH_SCAN_SECONDS", duration))
    if interval_s is None:
        interval = float(os.environ.get("SEARCH_SCAN_INTERVAL", interval))
    if duration <= 0 or interval <= 0:
        raise CameraError("Scan duration and interval must be positive.")
    return duration, interval
