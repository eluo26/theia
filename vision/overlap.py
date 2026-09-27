"""Link the same physical object across overlapping sweep photos.

Pan and tilt are still 0 on a handheld scan, so the same speaker lands at a
different pixel azimuth in each frame and the angle merge treats it as a new
object. Reference points between consecutive photos put the previous box back
onto the next photo. A box that lands on a detection is the same object. A
box that does not is a new one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from vision.detector import box_iou
from vision.matching import same_label_family
from vision.schemas import Detection

logger = logging.getLogger("vision.overlap")

_MIN_INLIERS = 12
_MIN_IOU = 0.15
_MAX_EDGE = 640


def overlapping_view_pairs(
    detections: list[Detection],
    settings,
    scan_dir: Path | None,
) -> list[tuple[int, int]]:
    """Index pairs that are one object seen in two consecutive photos."""
    if scan_dir is None or len(detections) < 2:
        return []
    by_frame: dict[str, list[int]] = {}
    for index, detection in enumerate(detections):
        by_frame.setdefault(detection.frame_file, []).append(index)
    frames = sorted(by_frame, key=_frame_sort_key)
    pairs: list[tuple[int, int]] = []
    for previous_name, current_name in zip(frames, frames[1:]):
        homography = _homography(Path(scan_dir), previous_name, current_name)
        previous_ids = by_frame[previous_name]
        current_ids = by_frame[current_name]
        if homography is None:
            pairs.extend(_singleton_links(previous_ids, current_ids, detections, settings))
            continue
        pairs.extend(
            _matched_links(previous_ids, current_ids, detections, settings, homography)
        )
    if pairs:
        logger.info("overlap linked %s repeated views", len(pairs))
    return pairs


def _matched_links(previous_ids, current_ids, detections, settings, homography) -> list[tuple[int, int]]:
    ranked: list[tuple[float, int, int]] = []
    for previous in previous_ids:
        warped = _warp_box(detections[previous].bbox_px, homography)
        if warped is None:
            continue
        for current in current_ids:
            if not _labels_compatible(detections[previous], detections[current], settings):
                continue
            overlap = box_iou(warped, detections[current].bbox_px)
            if overlap >= _MIN_IOU:
                ranked.append((overlap, previous, current))
    ranked.sort(reverse=True)
    used_previous: set[int] = set()
    used_current: set[int] = set()
    links: list[tuple[int, int]] = []
    for _overlap, previous, current in ranked:
        if previous in used_previous or current in used_current:
            continue
        used_previous.add(previous)
        used_current.add(current)
        links.append((previous, current))
    return links


def _singleton_links(previous_ids, current_ids, detections, settings) -> list[tuple[int, int]]:
    """When the photos have no usable reference points, keep a label that appears once.

    Two speakers in one photo stay separate. One speaker in both photos is the
    same speaker.
    """
    links: list[tuple[int, int]] = []
    used_previous: set[int] = set()
    used_current: set[int] = set()
    for previous in previous_ids:
        if previous in used_previous:
            continue
        current_matches = [
            current
            for current in current_ids
            if current not in used_current
            and _labels_compatible(detections[previous], detections[current], settings)
        ]
        previous_matches = [
            other
            for other in previous_ids
            if other not in used_previous
            and _labels_compatible(detections[previous], detections[other], settings)
        ]
        if len(current_matches) != 1 or len(previous_matches) != 1:
            continue
        current = current_matches[0]
        current_family = [
            other
            for other in current_ids
            if other not in used_current
            and _labels_compatible(detections[current], detections[other], settings)
        ]
        if len(current_family) != 1:
            continue
        links.append((previous, current))
        used_previous.add(previous)
        used_current.add(current)
    return links


def _labels_compatible(left: Detection, right: Detection, settings) -> bool:
    if left.drug_name and right.drug_name:
        if fuzz.token_ratio(left.drug_name, right.drug_name) < settings.label_sim:
            return False
    if same_label_family(left.label, right.label, settings.label_sim):
        return True
    return float(fuzz.token_set_ratio(left.label, right.label)) >= settings.label_sim


def _homography(scan_dir: Path, previous_name: str, current_name: str):
    previous = _read_gray(scan_dir, previous_name)
    current = _read_gray(scan_dir, current_name)
    if previous is None or current is None:
        return None
    previous_small, previous_scale = _downscale(previous)
    current_small, current_scale = _downscale(current)
    orb = cv2.ORB_create(nfeatures=800)
    prev_points, prev_desc = orb.detectAndCompute(previous_small, None)
    curr_points, curr_desc = orb.detectAndCompute(current_small, None)
    if prev_desc is None or curr_desc is None or len(prev_points) < _MIN_INLIERS:
        return None
    if len(curr_points) < _MIN_INLIERS:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(prev_desc, curr_desc, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good.append(best)
    if len(good) < _MIN_INLIERS:
        return None
    src = np.float32([prev_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    dst = np.float32([curr_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
        return None
    return _full_homography(homography, previous_scale, current_scale)


def _full_homography(homography, previous_scale: float, current_scale: float):
    """Map full-resolution pixels. The estimate was made on the downscaled photos."""
    to_small = np.array(
        [[previous_scale, 0, 0], [0, previous_scale, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    to_full = np.array(
        [[1 / current_scale, 0, 0], [0, 1 / current_scale, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return to_full @ homography @ to_small


def _warp_box(bbox_px: list[int], homography) -> list[int] | None:
    x1, y1, x2, y2 = (float(value) for value in bbox_px)
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, homography)
    except cv2.error:
        return None
    if warped is None or not np.isfinite(warped).all():
        return None
    xs = warped[:, 0, 0]
    ys = warped[:, 0, 1]
    left, right = float(xs.min()), float(xs.max())
    top, bottom = float(ys.min()), float(ys.max())
    if right - left < 1 or bottom - top < 1:
        return None
    return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]


def _read_gray(scan_dir: Path, frame_file: str):
    path = _frame_path(scan_dir, frame_file)
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    return image


def _frame_path(scan_dir: Path, frame_file: str) -> Path | None:
    raw = Path(frame_file)
    if raw.is_file():
        return raw
    candidate = scan_dir / raw.name
    if candidate.is_file():
        return candidate
    return None


def _downscale(image):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= _MAX_EDGE:
        return image, 1.0
    scale = _MAX_EDGE / longest
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _frame_sort_key(name: str) -> tuple:
    stem = Path(name).stem
    digits = "".join(character for character in stem if character.isdigit())
    return (int(digits) if digits else 0, name)
