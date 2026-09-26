"""Image-conditioned search against data/references.

TODO: this is not on the index or ask path. image_guided_boxes is the OWLv2
hook (query image in, boxes out). It needs the detector weights, about 1 GB,
and was not run in this environment. Call it only after
`pip install -r requirements-detector.txt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TODO_REFERENCE_SEARCH = (
    "Image-conditioned OWLv2 search is implemented in image_guided_boxes "
    "but is not called during index or ask. It needs the detector weights "
    "(about 1 GB) and has not been run in this environment."
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}


def reference_note(references_dir: Path) -> str | None:
    """Return the manual note when reference images are present, otherwise None.

    This does not load weights and does not pretend a match was found.
    """
    if not references_dir.is_dir():
        return None
    images = [
        path
        for path in references_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not images:
        return None
    return TODO_REFERENCE_SEARCH


def image_guided_boxes(
    image_rgb: np.ndarray,
    query_rgb: np.ndarray,
    model_id: str,
    *,
    threshold: float = 0.6,
) -> list[tuple[list[float], float]]:
    """OWLv2 image-guided detection. Downloads weights on first use.

    Not called by build_catalog or locate. See TODO_REFERENCE_SEARCH.
    """
    from vision.detector import DetectorUnavailable

    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as exc:
        raise DetectorUnavailable(
            "Reference search needs torch and transformers. "
            "pip install -r requirements-detector.txt"
        ) from exc

    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.eval()
    image = Image.fromarray(np.ascontiguousarray(image_rgb))
    query = Image.fromarray(np.ascontiguousarray(query_rgb))
    inputs = processor(images=image, query_images=query, return_tensors="pt")
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_image_guided_detection(
        outputs=outputs,
        threshold=threshold,
        nms_threshold=0.3,
        target_sizes=target_sizes,
    )[0]
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    return [(box, float(score)) for box, score in zip(boxes, scores)]
