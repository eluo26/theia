"""Red laser localization and pixel-space correction. Coordinates use the ON frame.

No servo commands are issued here: pixel offsets need camera/motor calibration first.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


class DotNotFound(ValueError):
    pass


@dataclass(frozen=True)
class Correction:
    dx_px: float
    dy_px: float
    distance_px: float
    on_target: bool

    @property
    def instruction(self):
        if self.on_target:
            return "On target; stop correcting."
        horizontal = f"{abs(self.dx_px):.1f} px {'right' if self.dx_px >= 0 else 'left'}"
        vertical = f"{abs(self.dy_px):.1f} px {'down' if self.dy_px >= 0 else 'up'}"
        return f"Move the dot {horizontal}, {vertical}."


def correction(dot, target, tolerance_px=5.0):
    """One correction-loop step: target minus dot; x right, y down."""
    if len(dot) != 2 or len(target) != 2:
        raise ValueError("Dot and target must each contain x and y.")
    if not all(math.isfinite(v) for v in (*dot, *target, tolerance_px)) or tolerance_px < 0:
        raise ValueError("Coordinates must be finite and tolerance must be nonnegative.")
    dx, dy = target[0] - dot[0], target[1] - dot[1]
    distance = math.hypot(dx, dy)
    return Correction(dx, dy, distance, distance <= tolerance_px)


def align_off_to_on(off, on):
    """Estimate a small camera rotation/translation/scale using robust feature matches."""
    height, width = on.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    size = (round(width * scale), round(height * scale))
    gray = [cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY) for frame in (off, on)]
    detector = cv2.ORB_create(nfeatures=4000)
    kp0, d0 = detector.detectAndCompute(gray[0], None)
    kp1, d1 = detector.detectAndCompute(gray[1], None)
    if d0 is None or d1 is None:
        raise DotNotFound("Not enough scene detail to align the images. Use a fixed camera or --no-align.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    if len(good) < 10:
        raise DotNotFound("Too few matching features to align the images reliably.")
    # Use exact per-axis resize factors, then estimate directly in original pixels.
    factors = np.array([width / size[0], height / size[1]], dtype=np.float32)
    src = np.float32([kp0[m.queryIdx].pt for m in good]) * factors
    dst = np.float32([kp1[m.trainIdx].pt for m in good]) * factors
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3 / scale)
    if matrix is None or inliers.sum() < 10 or inliers.mean() < 0.35:
        raise DotNotFound("Camera alignment is unreliable; capture another off/on pair.")
    aligned = cv2.warpAffine(off, matrix, (width, height))
    valid = cv2.warpAffine(np.full((height, width), 255, np.uint8), matrix, (width, height))
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) == 255
    return aligned, valid, {"inliers": int(inliers.sum()), "matches": len(good), "off_to_on": matrix.tolist()}


def find_laser_dot(off, on, align=True):
    """Return ((x, y), diagnostics) for a red dot newly present in the ON image.

    Reject missing or similarly strong competing spots rather than guessing.
    This detector is for a red laser; other colors require a different color signal.
    """
    if off is None or on is None or off.shape != on.shape or on.ndim != 3 or on.shape[2] != 3:
        raise ValueError("Provide two readable BGR images with the same dimensions.")
    if align:
        reference, valid, alignment = align_off_to_on(off, on)
    else:
        reference, valid, alignment = off, np.ones(on.shape[:2], dtype=bool), None

    def redness(frame):
        b, g, r = cv2.split(frame.astype(np.float32))
        return r - np.maximum(b, g)

    red_on = redness(on)
    gain = red_on - redness(reference)
    baseline = float(np.median(gain[valid]))
    noise = float(np.median(np.abs(gain[valid] - baseline))) * 1.4826
    threshold = max(25.0, 8 * noise)
    gain -= baseline
    mask = ((gain > threshold) & (red_on > 30) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 3 or area > max(100, on.shape[0] * on.shape[1] * 0.001) or max(w, h) > 4 * min(w, h):
            continue
        region = labels[y:y+h, x:x+w] == label
        weights = gain[y:y+h, x:x+w] * region
        score = float(weights.sum())
        yy, xx = np.indices(weights.shape)
        point = (float(x + (xx * weights).sum() / score), float(y + (yy * weights).sum() / score))
        candidates.append((score, point, int(area)))
    candidates.sort(reverse=True)
    if not candidates:
        raise DotNotFound("No new red laser dot detected. Check frame order, visibility, and exposure.")
    if len(candidates) > 1 and candidates[1][0] > 0.65 * candidates[0][0]:
        raise DotNotFound("Multiple similarly strong red spots detected; cannot choose a dot reliably.")
    score, point, area = candidates[0]
    return point, {"alignment": alignment, "threshold": threshold, "area_px": area,
                   "candidate_count": len(candidates), "signal_score": score}


def evaluate_pair(off, on, target, tolerance_px=5.0, align=True):
    dot, diagnostics = find_laser_dot(off, on, align=align)
    result = correction(dot, target, tolerance_px)
    return {"dot_px": dot, "target_px": target, **asdict(result),
            "instruction": result.instruction, "diagnostics": diagnostics}
