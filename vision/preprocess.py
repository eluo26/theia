"""Undistort a frame when calibration is present, then tile it.

tile_grid is [columns, rows]. tile_overlap is the fraction of each tile
shared with its neighbor. Each tile records the pixel offset of its
top-left corner in the full frame so a 0-1000 box can be mapped back.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from vision.config import Settings
from vision.ingest import LoadedFrame
from vision.schemas import norm_box_to_pixels

logger = logging.getLogger("vision.preprocess")


@dataclass
class Tile:
    frame: LoadedFrame
    image: np.ndarray
    offset_x: int
    offset_y: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def prepare_frame(frame: LoadedFrame, settings: Settings) -> LoadedFrame:
    """Replace the image with an undistorted copy when calibration is complete."""
    frame.image = undistort(frame.image, settings)
    return frame


def undistort(image: np.ndarray, settings: Settings) -> np.ndarray:
    calibrated = None not in (settings.fx, settings.fy, settings.cx, settings.cy)
    if settings.dist_coeffs and not calibrated:
        logger.warning(
            "dist_coeffs is set but fx, fy, cx, and cy are not all set; skipping undistort"
        )
        return image
    if not calibrated or not settings.dist_coeffs:
        return image
    camera = np.array(
        [
            [settings.fx, 0.0, settings.cx],
            [0.0, settings.fy, settings.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(settings.dist_coeffs, dtype=np.float64)
    return cv2.undistort(image, camera, distortion)


def make_tiles(frame: LoadedFrame, settings: Settings) -> list[Tile]:
    columns, rows = settings.tile_grid
    windows = tile_windows(frame.width, frame.height, int(columns), int(rows), settings.tile_overlap)
    tiles: list[Tile] = []
    for offset_x, offset_y, width, height in windows:
        crop = frame.image[offset_y : offset_y + height, offset_x : offset_x + width]
        tiles.append(Tile(frame=frame, image=crop, offset_x=offset_x, offset_y=offset_y))
    return tiles


def tile_windows(
    width: int, height: int, columns: int, rows: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) for a grid. The last window in each axis ends on the image edge."""
    xs = _axis(width, columns, overlap)
    ys = _axis(height, rows, overlap)
    return [(x, y, w, h) for y, h in ys for x, w in xs]


def map_norm_box_to_frame(
    box: list[float],
    tile_width: int,
    tile_height: int,
    offset_x: int,
    offset_y: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    """Map a 0-1000 tile box onto full-frame pixels."""
    x1, y1, x2, y2 = norm_box_to_pixels(box, tile_width, tile_height)
    return shift_box([x1, y1, x2, y2], offset_x, offset_y, frame_width, frame_height)


def shift_box(
    box: list[int] | tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    x1, y1, x2, y2 = box
    return _clamp_box(
        [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        frame_width,
        frame_height,
    )


def encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def encode_jpeg(image: np.ndarray, *, quality: int = 80) -> bytes:
    """JPEG payload for a vision call. The array is already the working size."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()


def _axis(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    if count <= 1:
        return [(0, length)]
    tile = length / ((1 - overlap) * (count - 1) + 1)
    stride = tile * (1 - overlap)
    windows: list[tuple[int, int]] = []
    for index in range(count):
        start = int(round(index * stride))
        end = length if index == count - 1 else int(round(index * stride + tile))
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        windows.append((start, end - start))
    return windows


def _clamp_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]
