"""Local open-vocabulary detector used to tighten Grok boxes.

The backend comes from config.yaml `detector`: owlv2 or grounding_dino.
Weights download on the first real call (about 1 GB). Unit tests inject a
fake detector and must not download weights. If torch is missing or the
download fails, callers keep the Grok box and continue.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger("vision.detector")

GROK_ONLY_CONFIDENCE = 0.45


class DetectorUnavailable(RuntimeError):
    """Torch, transformers, or the weight files are not available."""


@dataclass
class DetectorHit:
    bbox_px: list[int]
    score: float


class ObjectDetector(Protocol):
    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        """Boxes are pixels of image_rgb, the tile or crop that was passed in."""


class NullDetector:
    """Used when the local detector cannot be loaded. Every query returns nothing."""

    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        return []


class ScriptedDetector:
    """Test double. Maps a text query to hits in the image's pixel space."""

    def __init__(self, hits_by_text: dict[str, list[DetectorHit]] | None = None):
        self.hits_by_text = hits_by_text or {}
        self.calls: list[str] = []

    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        self.calls.append(text)
        return list(self.hits_by_text.get(text, []))


class FallbackDetector:
    """Turn a missing-weight failure into an empty result and stop trying."""

    def __init__(self, inner: ObjectDetector):
        self.inner = inner
        self.disabled = False

    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        if self.disabled:
            return []
        try:
            return self.inner.detect(image_rgb, text)
        except DetectorUnavailable as exc:
            if not self.disabled:
                logger.warning("%s", exc)
            self.disabled = True
            return []


class _LazyModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def ensure(self) -> tuple[object, object]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model
        with self._lock:
            if self._model is not None and self._processor is not None:
                return self._processor, self._model
            try:
                processor, model = self.load()
            except DetectorUnavailable:
                raise
            except Exception as exc:
                raise DetectorUnavailable(
                    f"Could not load detector weights for {self.model_id}. "
                    "Install them with: pip install -r requirements-detector.txt. "
                    "The first successful load downloads about 1 GB. "
                    f"Details: {exc}"
                ) from exc
            self._processor = processor
            self._model = model
            return processor, model

    def load(self) -> tuple[object, object]:
        raise NotImplementedError


class OwlV2Detector(_LazyModel):
    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        try:
            processor, model = self.ensure()
        except DetectorUnavailable:
            raise
        try:
            return _owl_infer(processor, model, image_rgb, text)
        except DetectorUnavailable:
            raise
        except Exception as exc:
            logger.warning("detector inference failed: %s", exc)
            return []

    def load(self) -> tuple[object, object]:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise DetectorUnavailable(
                "The local detector needs torch and transformers. "
                "Install them with: pip install -r requirements-detector.txt. "
                "The first run downloads about 1 GB of weights."
            ) from exc
        processor = Owlv2Processor.from_pretrained(self.model_id)
        model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        model.eval()
        self._torch = torch
        return processor, model


class GroundingDinoDetector(_LazyModel):
    def detect(self, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
        try:
            processor, model = self.ensure()
        except DetectorUnavailable:
            raise
        try:
            return _dino_infer(processor, model, image_rgb, text)
        except DetectorUnavailable:
            raise
        except Exception as exc:
            logger.warning("detector inference failed: %s", exc)
            return []

    def load(self) -> tuple[object, object]:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise DetectorUnavailable(
                "The local detector needs torch and transformers. "
                "Install them with: pip install -r requirements-detector.txt. "
                "The first run downloads about 1 GB of weights."
            ) from exc
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        model.eval()
        self._torch = torch
        return processor, model


def build_detector(settings) -> ObjectDetector:
    """Construct the configured backend. Weights are not loaded until detect()."""
    if settings.detector == "owlv2":
        return OwlV2Detector(settings.owlv2_model)
    if settings.detector == "grounding_dino":
        return GroundingDinoDetector(settings.grounding_dino_model)
    return NullDetector()


def resolve_detector(settings, override: ObjectDetector | None = None) -> ObjectDetector:
    inner = override if override is not None else build_detector(settings)
    if isinstance(inner, FallbackDetector):
        return inner
    return FallbackDetector(inner)


def refine_box(
    grok_bbox: list[int],
    hits: list[DetectorHit],
    refine_iou: float,
) -> tuple[list[int], str, float]:
    """Use the detector box when it overlaps the Grok box. Otherwise keep Grok and lower confidence."""
    best: DetectorHit | None = None
    best_iou = 0.0
    for hit in hits:
        overlap = box_iou(grok_bbox, hit.bbox_px)
        if overlap >= refine_iou and (best is None or overlap > best_iou):
            best = hit
            best_iou = overlap
    if best is not None:
        score = float(min(1.0, max(0.0, best.score)))
        return [int(v) for v in best.bbox_px], "detector", score
    return [int(v) for v in grok_bbox], "grok", GROK_ONLY_CONFIDENCE


def tighter_agreeing_box(
    first: list[int] | None,
    second: list[int] | None,
    min_iou: float,
) -> tuple[list[int] | None, str]:
    """Return the smaller box when both exist and their IoU is high enough.

    The source name is "detector" when second wins, and "grok" when first wins.
    """
    if first is None and second is None:
        return None, "grok"
    if first is None:
        return second, "detector"
    if second is None:
        return first, "grok"
    if box_iou(first, second) < min_iou:
        return first, "grok"
    if _area(second) < _area(first):
        return second, "detector"
    return first, "grok"


def box_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in first)
    bx1, by1, bx2, by2 = (float(v) for v in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    union = _area(first) + _area(second) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _area(box) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _owl_infer(processor, model, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
    import torch
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(image_rgb))
    inputs = processor(text=[[text]], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target, threshold=0.1
    )[0]
    return _hits_from_result(results)


def _dino_infer(processor, model, image_rgb: np.ndarray, text: str) -> list[DetectorHit]:
    import torch
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(image_rgb))
    query = text if text.endswith(".") else f"{text}."
    inputs = processor(images=image, text=query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_grounded_object_detection
    try:
        results = post(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = post(outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
    return _hits_from_result(results)


def _hits_from_result(results) -> list[DetectorHit]:
    boxes = results["boxes"]
    scores = results["scores"]
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    hits: list[DetectorHit] = []
    for box, score in zip(boxes, scores):
        hits.append(DetectorHit(bbox_px=[int(round(v)) for v in box], score=float(score)))
    return hits
