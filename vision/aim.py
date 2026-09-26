"""Connected aim hook for the UI and the laser program.

Folder and HEIC ingest stay on answer(scan_dir, query) for local tests.
The UI calls aim(frames, query). Each frame is a JPEG from the camera plus
the pan and tilt it was taken at. The API key is read only inside this
process, from the environment or a gitignored .env. It is never returned.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vision.config import Settings, load_settings, require_api_key
from vision.ingest import IngestError
from vision.integrate import answer

_ITEM_KEYS = ("id", "label", "azimuth_deg", "elevation_deg", "confidence")


def aim(
    frames: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    query: str,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    use_cache: bool = True,
) -> dict:
    """Index live JPEG frames and return the laser payload.

    frames: one mapping, or a list of them. Each mapping has:
      image: JPEG bytes
      pan: degrees, 0 at pan home, positive to the right
      tilt: degrees, 0 horizontal, positive up
      timestamp: optional string
    Returns fire_laser, aim {azimuth_deg, elevation_deg} or null, and a short
    items list. fire_laser is true only when the query status is found.
    """
    if not isinstance(query, str) or not query.strip():
        raise IngestError("A question is required.")
    prepared = _prepare_frames(frames)
    settings = settings if settings is not None else load_settings()
    require_api_key(settings)
    with tempfile.TemporaryDirectory(prefix="theia-aim-") as directory:
        scan_dir = _write_scan(Path(directory), prepared)
        raw = answer(
            scan_dir,
            query.strip(),
            settings=settings,
            client_factory=client_factory,
            detector=detector,
            use_cache=use_cache,
        )
    return laser_payload(raw)


def laser_payload(raw: Mapping[str, Any]) -> dict:
    """Drop catalog detail the laser program does not aim with."""
    result = raw.get("result") if isinstance(raw.get("result"), Mapping) else {}
    source = raw.get("aim") if isinstance(raw.get("aim"), Mapping) else None
    found = raw.get("fire_laser") is True and result.get("status") == "found"
    aim_point = None
    if found and source is not None:
        azimuth = source.get("azimuth_deg")
        elevation = source.get("elevation_deg")
        if isinstance(azimuth, (int, float)) and isinstance(elevation, (int, float)):
            aim_point = {"azimuth_deg": azimuth, "elevation_deg": elevation}
        else:
            found = False
    items = []
    for item in raw.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        row = {key: item.get(key) for key in _ITEM_KEYS}
        if row["id"] is None:
            row["id"] = item.get("object_id")
        items.append(row)
    return {"fire_laser": found, "aim": aim_point, "items": items}


def _prepare_frames(frames: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict]:
    if isinstance(frames, Mapping):
        frames = [frames]
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise IngestError("frames must be a list of {image, pan, tilt, timestamp}.")
    if len(frames) == 0:
        raise IngestError("No frames. Each frame needs JPEG bytes, pan, and tilt.")
    prepared = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise IngestError(f"Frame {index} must include image, pan, and tilt.")
        image = frame.get("image")
        if not isinstance(image, (bytes, bytearray)) or not bytes(image).startswith(b"\xff\xd8"):
            raise IngestError(f"Frame {index} image must be JPEG bytes.")
        pan = _degrees(frame, index, "pan", "pan_deg")
        tilt = _degrees(frame, index, "tilt", "tilt_deg")
        timestamp = frame.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise IngestError(f"Frame {index} timestamp must be a string when set.")
        prepared.append(
            {
                "image": bytes(image),
                "pan": pan,
                "tilt": tilt,
                "timestamp": timestamp,
            }
        )
    return prepared


def _degrees(frame: Mapping[str, Any], index: int, *names: str) -> float:
    for name in names:
        if name in frame and frame[name] is not None:
            try:
                value = float(frame[name])
            except (TypeError, ValueError) as exc:
                raise IngestError(f"Frame {index} {names[0]} must be degrees.") from exc
            if not math.isfinite(value):
                raise IngestError(f"Frame {index} {names[0]} must be degrees.")
            return value
    raise IngestError(f"Frame {index} needs {names[0]} in degrees.")


def _write_scan(directory: Path, frames: list[dict]) -> Path:
    entries = []
    for index, frame in enumerate(frames):
        name = f"frame_{index:03d}.jpg"
        (directory / name).write_bytes(frame["image"])
        entries.append(
            {
                "file": name,
                "pan": frame["pan"],
                "tilt": frame["tilt"],
                "timestamp": frame["timestamp"],
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps({"frames": entries}),
        encoding="utf-8",
    )
    return directory
