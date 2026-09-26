"""Load stills, or sample a sweep video into frames.

Filenames such as pan060_tilt-10.jpg supply pan and tilt. A manifest.json
of {file, pan, tilt, timestamp} wins when both are present. Other images,
including .heic and .HEIC, are sorted by an integer in the name when one
exists, otherwise by filename, and assigned pan 0, 30, 60, ... with tilt 0.
Images stay at full resolution.

Video needs either a sidecar CSV (timestamp_s, pan, tilt), interpolated per
sampled frame, or a constant-speed --sweep start_pan end_pan tilt. Frames are
taken every video_sample_every_s. A frame is dropped when its Laplacian
variance is below blur_threshold. The sharpest frame in each 1-degree
pan/tilt bucket is kept.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, ConfigDict

register_heif_opener()

logger = logging.getLogger("vision.ingest")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
_FRAME_NAME = re.compile(
    r"pan(?P<pan>-?\d+(?:\.\d+)?)_tilt(?P<tilt>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_INT_IN_NAME = re.compile(r"\d+")
_UNTAGGED_PAN_STEP = 30.0


class IngestError(ValueError):
    """The scan directory or video could not be turned into frames."""


def no_images_message(scan_dir: Path) -> str:
    """What to tell the user when a folder has no accepted stills."""
    return (
        f"No images in {scan_dir}. "
        "Add photos, name files pan030_tilt-10.jpg, or add manifest.json with file, pan, tilt, timestamp. "
        "HEIC files in the folder are accepted and angled by sorted order."
    )


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file: str
    pan: float
    tilt: float
    timestamp: str | None = None


@dataclass
class LoadedFrame:
    """One capture. The array is RGB uint8 and is not written into JSON."""

    source_file: str
    path: Path
    pan_deg: float
    tilt_deg: float
    timestamp: str | None
    image: np.ndarray
    full_width: int = 0
    full_height: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def camera_width(self) -> int:
        """Pixel width of the original photo. The working image may be smaller."""
        return self.full_width or self.width

    @property
    def camera_height(self) -> int:
        return self.full_height or self.height


@dataclass
class SampledFrame:
    timestamp_s: float
    pan_deg: float
    tilt_deg: float
    image: np.ndarray
    sharpness: float


def parse_angles_from_name(name: str) -> tuple[float, float] | None:
    match = _FRAME_NAME.search(Path(name).name)
    if match is None:
        return None
    return float(match.group("pan")), float(match.group("tilt"))


def is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_SUFFIXES


def load_scan(
    source: str | Path,
    *,
    out_dir: Path,
    video_sample_every_s: float,
    blur_threshold: float,
    sweep: tuple[float, float, float] | None = None,
    angles_csv: str | Path | None = None,
    max_edge: int | None = None,
) -> list[LoadedFrame]:
    """Load stills from a directory, or sample a video file."""
    path = Path(source)
    if not path.exists():
        raise IngestError(f"Scan path does not exist: {path}")
    if path.is_file() and is_video(path):
        if sweep is not None and angles_csv is not None:
            raise IngestError("Pass either --sweep or --angles-csv, not both.")
        if sweep is None and angles_csv is None:
            raise IngestError(
                "A video needs angles. Pass --sweep START_PAN END_PAN TILT, "
                "or --angles-csv with columns timestamp_s,pan,tilt."
            )
        return ingest_video(
            path,
            out_dir=out_dir,
            sample_every_s=video_sample_every_s,
            blur_threshold=blur_threshold,
            sweep=sweep,
            angles_csv=Path(angles_csv) if angles_csv is not None else None,
        )
    if not path.is_dir():
        raise IngestError(f"Expected an image directory or a video file: {path}")
    if sweep is not None or angles_csv is not None:
        raise IngestError("--sweep and --angles-csv apply to a video file, not an image folder.")
    return ingest_directory(path, max_edge=max_edge)


def _untagged_order_key(path: Path) -> tuple[int, int | str, str]:
    """Names with no integer stay in filename order ahead of numbered names.

    Numbered names sort by the first integer, so 2.jpg precedes 10.jpg.
    """
    name = path.name
    match = _INT_IN_NAME.search(name)
    if match is None:
        return (0, name, name)
    return (1, int(match.group(0)), name)


def still_frame_meta(scan_dir: Path) -> list[tuple[Path, float, float, str | None]]:
    """Angle tags for stills, without decoding the pixels."""
    manifest = _load_manifest(scan_dir / "manifest.json")
    images = sorted(
        child
        for child in scan_dir.iterdir()
        if child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES
    )
    resolved: dict[Path, tuple[float, float, str | None]] = {}
    untagged: list[Path] = []
    for image_path in images:
        entry = manifest.get(image_path.name)
        if entry is not None:
            resolved[image_path] = (entry.pan, entry.tilt, entry.timestamp)
            continue
        parsed = parse_angles_from_name(image_path.name)
        if parsed is not None:
            resolved[image_path] = (parsed[0], parsed[1], None)
            continue
        untagged.append(image_path)
    for index, image_path in enumerate(sorted(untagged, key=_untagged_order_key)):
        pan = index * _UNTAGGED_PAN_STEP
        logger.info("%s assigned pan %s", image_path.name, int(pan))
        resolved[image_path] = (pan, 0.0, None)
    return [
        (image_path, resolved[image_path][0], resolved[image_path][1], resolved[image_path][2])
        for image_path in images
    ]


def load_still_frames(
    meta: list[tuple[Path, float, float, str | None]],
    max_edge: int | None = None,
) -> list[LoadedFrame]:
    """Decode the listed stills. Callers pass only the files that need indexing."""
    frames: list[LoadedFrame] = []
    for image_path, pan, tilt, timestamp in meta:
        image, full_w, full_h = load_working_image(image_path, max_edge)
        frames.append(
            LoadedFrame(
                source_file=image_path.name,
                path=image_path.resolve(),
                pan_deg=pan,
                tilt_deg=tilt,
                timestamp=timestamp,
                image=image,
                full_width=full_w,
                full_height=full_h,
            )
        )
    return frames


def ingest_directory(scan_dir: Path, max_edge: int | None = None) -> list[LoadedFrame]:
    frames = load_still_frames(still_frame_meta(scan_dir), max_edge)
    if not frames:
        raise IngestError(no_images_message(scan_dir))
    return frames


def ingest_video(
    video_path: Path,
    *,
    out_dir: Path,
    sample_every_s: float,
    blur_threshold: float,
    sweep: tuple[float, float, float] | None,
    angles_csv: Path | None,
) -> list[LoadedFrame]:
    rows = _load_angles_csv(angles_csv) if angles_csv is not None else None
    sampled, duration = _sample_video(video_path, sample_every_s)
    if not sampled:
        raise IngestError(f"Could not read frames from {video_path}")
    stamped: list[SampledFrame] = []
    for timestamp_s, image in sampled:
        pan, tilt = _angles_at(timestamp_s, duration, sweep, rows)
        stamped.append(
            SampledFrame(
                timestamp_s=timestamp_s,
                pan_deg=pan,
                tilt_deg=tilt,
                image=image,
                sharpness=laplacian_variance(image),
            )
        )
    kept = keep_sharpest_per_bucket(stamped, blur_threshold)
    if not kept:
        raise IngestError(
            f"Every sampled frame in {video_path} was below blur_threshold {blur_threshold}. "
            "Use a slower sweep or lower blur_threshold in config.yaml."
        )
    destination = out_dir / "frames"
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[LoadedFrame] = []
    for sample in sorted(kept, key=lambda item: (item.pan_deg, item.tilt_deg, item.timestamp_s)):
        name = (
            f"{video_path.stem}_t{sample.timestamp_s:06.2f}"
            f"_pan{sample.pan_deg:.1f}_tilt{sample.tilt_deg:.1f}.png"
        )
        path = destination / name
        Image.fromarray(sample.image).save(path)
        frames.append(
            LoadedFrame(
                source_file=name,
                path=path.resolve(),
                pan_deg=sample.pan_deg,
                tilt_deg=sample.tilt_deg,
                timestamp=f"{sample.timestamp_s:.3f}",
                image=sample.image,
            )
        )
    logger.info(
        "video ingest kept %s of %s sampled frames from %s",
        len(frames),
        len(stamped),
        video_path.name,
    )
    return frames


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def keep_sharpest_per_bucket(samples: list[SampledFrame], blur_threshold: float) -> list[SampledFrame]:
    """Drop soft frames, then keep the sharpest sample in each 1-degree bucket."""
    kept: dict[tuple[int, int], SampledFrame] = {}
    for sample in samples:
        if sample.sharpness < blur_threshold:
            continue
        key = (int(round(sample.pan_deg)), int(round(sample.tilt_deg)))
        current = kept.get(key)
        if current is None or sample.sharpness > current.sharpness:
            kept[key] = sample
    return list(kept.values())


_SWAP_ORIENTATION = {5, 6, 7, 8}


def load_rgb(path: Path) -> np.ndarray:
    image, _full_w, _full_h = load_working_image(path, None)
    return image


def load_working_image(path: Path, max_edge: int | None) -> tuple[np.ndarray, int, int]:
    """RGB array for indexing, plus the original photo size after EXIF orientation.

    Phone photos, including HEIC, are decoded toward max_edge so a 12 MP still is
    not held in full while the vision call runs. Boxes are mapped back onto the
    original size. pillow-heif registers the HEIC opener; the RGB, EXIF, and
    resize steps below are the same as for JPEG.
    """
    with Image.open(path) as image:
        full_w, full_h = image.size
        orientation = image.getexif().get(274)
        if orientation in _SWAP_ORIENTATION:
            full_w, full_h = full_h, full_w
        if (
            max_edge
            and max_edge > 0
            and image.format == "JPEG"
            and max(image.size) > max_edge
        ):
            image.draft("RGB", (max_edge, max_edge))
        turned = ImageOps.exif_transpose(image)
        if turned is None:
            turned = image
        if max_edge and max_edge > 0 and max(turned.size) > max_edge:
            scale = max_edge / float(max(turned.size))
            turned = turned.resize(
                (
                    max(1, int(round(turned.size[0] * scale))),
                    max(1, int(round(turned.size[1] * scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        array = np.asarray(turned.convert("RGB"))
    return array, int(full_w), int(full_h)


def _load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("frames", raw.get("images"))
    if not isinstance(raw, list):
        raise IngestError(f"{path} must be a list of {{file, pan, tilt, timestamp}} objects.")
    entries: dict[str, ManifestEntry] = {}
    for item in raw:
        entry = ManifestEntry.model_validate(item)
        entries[Path(entry.file).name] = entry
    return entries


def _load_angles_csv(path: Path) -> list[tuple[float, float, float]]:
    if not path.is_file():
        raise IngestError(f"Angles CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise IngestError(f"{path} needs a header row: timestamp_s,pan,tilt")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        missing = {"timestamp_s", "pan", "tilt"} - set(fields)
        if missing:
            raise IngestError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows: list[tuple[float, float, float]] = []
        for record in reader:
            rows.append(
                (
                    float(record[fields["timestamp_s"]]),
                    float(record[fields["pan"]]),
                    float(record[fields["tilt"]]),
                )
            )
    if not rows:
        raise IngestError(f"{path} has no angle rows.")
    rows.sort(key=lambda item: item[0])
    return rows


def _sample_video(path: Path, sample_every_s: float) -> tuple[list[tuple[float, np.ndarray]], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IngestError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 1e-3:
            fps = 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = count / fps if count > 0 else 0.0
        samples: list[tuple[float, np.ndarray]] = []
        next_t = 0.0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = index / fps
            if timestamp_s + 1e-6 >= next_t:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                samples.append((timestamp_s, rgb))
                next_t += sample_every_s
            index += 1
        if duration <= 0 and index:
            duration = index / fps
        return samples, duration
    finally:
        capture.release()


def _angles_at(
    timestamp_s: float,
    duration: float,
    sweep: tuple[float, float, float] | None,
    rows: list[tuple[float, float, float]] | None,
) -> tuple[float, float]:
    if rows is not None:
        return _interpolate(rows, timestamp_s)
    if sweep is None:
        raise IngestError("Video ingest needs a sweep or an angles CSV.")
    start_pan, end_pan, tilt = sweep
    if duration <= 0:
        return start_pan, tilt
    fraction = min(1.0, max(0.0, timestamp_s / duration))
    return start_pan + (end_pan - start_pan) * fraction, tilt


def _interpolate(rows: list[tuple[float, float, float]], timestamp_s: float) -> tuple[float, float]:
    if timestamp_s <= rows[0][0]:
        return rows[0][1], rows[0][2]
    if timestamp_s >= rows[-1][0]:
        return rows[-1][1], rows[-1][2]
    for earlier, later in zip(rows, rows[1:]):
        if earlier[0] <= timestamp_s <= later[0]:
            span = later[0] - earlier[0]
            fraction = 0.0 if span == 0 else (timestamp_s - earlier[0]) / span
            pan = earlier[1] + fraction * (later[1] - earlier[1])
            tilt = earlier[2] + fraction * (later[2] - earlier[2])
            return pan, tilt
    return rows[-1][1], rows[-1][2]
