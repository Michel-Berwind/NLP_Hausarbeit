"""src.ocr.pipeline.price_region_detection

Contour-based Price Region (Price Box) Detection
================================================

This module implements *only* classical computer vision using OpenCV.
It detects price regions ("price boxes") from a morphologically processed
(binary) image where price tags appear as connected rectangular regions.

Expected inputs (from your preprocessing):
- `original_bgr`: original page image (BGR)
- `morph`: binary-ish morph mask, where candidate price regions are white (255)

Outputs:
- bounding boxes as `(x, y, w, h)`
- cropped BGR regions ready for OCR
- debug visualizations saved to a folder (e.g. `data/debug_price_boxes/...`)

Example usage (after preprocessing):

    from pathlib import Path
    import cv2

    from src.ocr.pipeline.image_preprocessing import load_image, preprocess_for_text
    from src.ocr.pipeline.price_region_detection import (
        detect_price_regions_from_morph,
        PriceRegionDetectionConfig,
    )

    img_path = Path("data/images/aldi/KW40_25_ebeae15a-90e5-4975-a5cd-ddd640c8c977_page4.png")
    img = load_image(img_path)

    stages = preprocess_for_text(img)
    morph = stages["morph"]

    bboxes, crops = detect_price_regions_from_morph(
        morph=morph,
        original_bgr=img,
        config=PriceRegionDetectionConfig(),
        debug_dir=Path("data/debug_price_boxes") / img_path.stem / "contour_detector",
    )

    print("Detected", len(bboxes), "price regions")

Notes on the heuristics
----------------------
This detector uses simple, configurable filters:
- area range (relative to image area): removes tiny noise and huge sections
- width/height constraints: ensures regions are OCR-readable and not full-width banners
- aspect ratio: price boxes are typically wider than tall, but allow some variation

No ML/DL models are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class PriceRegionDetectionConfig:
    """Configuration for contour-based price region detection.

    All thresholds are intentionally easy to tune and are documented with *why*
    they exist.

    Most values are expressed relative to image size to make the detector more
    robust across resolutions.
    """

    # --- Area constraints (relative to image area) ---
    # Why: remove tiny specks (noise) and huge blobs (whole sections).
    min_area_ratio: float = 0.00010  # 0.01% of image area
    max_area_ratio: float = 0.08000  # 8% of image area

    # --- Absolute minimum size (pixels) ---
    # Why: OCR needs enough pixels; very small boxes are rarely readable.
    min_width_px: int = 25
    min_height_px: int = 15

    # --- Maximum size (relative to image width/height) ---
    # Why: filter banners / headers / full-width separators.
    max_width_ratio: float = 0.40
    max_height_ratio: float = 0.30

    # --- Aspect ratio (w / h) ---
    # Why: price tags are typically rectangular and not extremely skinny.
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 5.00

    # --- Crop padding (pixels) ---
    # Why: include a small margin so digits at the edge aren’t cut off.
    crop_padding_px: int = 2

    # --- Non-maximum suppression ---
    # Why: morphological masks can create overlapping contours for the same tag.
    nms_iou_threshold: float = 0.30

    # --- Contour retrieval ---
    # Why: external contours are usually what we want for region proposals.
    contour_retrieval_mode: int = cv2.RETR_EXTERNAL


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce input mask to a single-channel 0/255 uint8 image."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.dtype != np.uint8:
        # In case upstream returns bool/float; keep it predictable for OpenCV.
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Ensure strict binary.
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _clip_bbox_to_image(bbox: BBox, H: int, W: int) -> BBox:
    x, y, w, h = bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter
    return inter / union


def _nms_by_area(bboxes: List[BBox], iou_threshold: float) -> List[BBox]:
    """Simple NMS: keep larger boxes when overlap is high."""
    if not bboxes:
        return []
    if iou_threshold <= 0:
        return bboxes

    bboxes_sorted = sorted(bboxes, key=lambda bb: bb[2] * bb[3], reverse=True)
    kept: List[BBox] = []

    for bb in bboxes_sorted:
        if all(_bbox_iou(bb, kk) <= iou_threshold for kk in kept):
            kept.append(bb)

    return kept


def detect_price_regions_from_morph(
    *,
    morph: np.ndarray,
    original_bgr: np.ndarray,
    config: PriceRegionDetectionConfig = PriceRegionDetectionConfig(),
    debug_dir: Optional[Path] = None,
) -> Tuple[List[BBox], List[np.ndarray]]:
    """Detect price regions (price boxes) from a morph mask using contours.

    Requirements satisfied:
    - Uses `cv2.findContours` on the morph image
    - Computes bounding boxes for contours
    - Filters using configurable heuristics (area ratio, aspect ratio, width/height)
    - Returns `(x, y, w, h)` bboxes and cropped regions
    - Writes debug visualizations to `debug_dir` when provided

    Args:
        morph: Morphologically processed binary-ish mask (white blobs = candidates)
        original_bgr: Original BGR page image (used for cropping + drawing)
        config: Tunable heuristics
        debug_dir: If set, saves debug images and crops into this folder

    Returns:
        (bboxes, crops)
    """
    H, W = original_bgr.shape[:2]
    img_area = float(H * W)

    binary = _ensure_binary_mask(morph)

    # --- Step 1: contour detection on morph mask ---
    contours, _ = cv2.findContours(
        binary,
        config.contour_retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: List[BBox] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)

        # Filter: area relative to page size
        # Why: drop tiny specks (noise) and huge regions (sections/background blocks).
        if area < config.min_area_ratio * img_area:
            continue
        if area > config.max_area_ratio * img_area:
            continue

        # Filter: absolute min size
        # Why: ensures the region has enough pixels for OCR.
        if w < config.min_width_px:
            continue
        if h < config.min_height_px:
            continue

        # Filter: max size relative to the page
        # Why: avoids selecting banners/headers that are not prices.
        if w > config.max_width_ratio * W:
            continue
        if h > config.max_height_ratio * H:
            continue

        # Filter: aspect ratio typical for price boxes
        # Why: prices are usually in compact rectangles; extreme ratios are unlikely.
        aspect = w / float(h)
        if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
            continue

        candidates.append((x, y, w, h))

    # Optional but useful: remove duplicates/overlaps
    final_bboxes = _nms_by_area(candidates, config.nms_iou_threshold)

    # Stable ordering for debugging and downstream OCR batching
    final_bboxes = sorted(final_bboxes, key=lambda bb: (bb[1], bb[0]))

    # Create crops
    crops: List[np.ndarray] = []
    for (x, y, w, h) in final_bboxes:
        pad = int(config.crop_padding_px)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        crops.append(original_bgr[y0:y1, x0:x1].copy())

    # Debug outputs
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_dir / "00_morph_input.png"), binary)

        # Draw candidates (pre-NMS) in yellow
        dbg_candidates = original_bgr.copy()
        for (x, y, w, h) in candidates:
            cv2.rectangle(dbg_candidates, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imwrite(str(debug_dir / "01_all_candidates.png"), dbg_candidates)

        # Draw final boxes in green with index
        dbg_final = original_bgr.copy()
        for idx, (x, y, w, h) in enumerate(final_bboxes):
            cv2.rectangle(dbg_final, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                dbg_final,
                str(idx),
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(debug_dir / "02_final_boxes.png"), dbg_final)

        # Also show boxes on the morph mask (useful to see what the mask produced)
        morph_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in final_bboxes:
            cv2.rectangle(morph_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "03_morph_with_boxes.png"), morph_vis)

        crops_dir = debug_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        for idx, crop in enumerate(crops):
            cv2.imwrite(str(crops_dir / f"crop_{idx:03d}.png"), crop)

    return final_bboxes, crops


if __name__ == "__main__":
    # Minimal runnable demo using your existing preprocessing.
    from src.ocr.pipeline.image_preprocessing import load_image, preprocess_for_text

    test_path = Path("data/images/aldi/KW40_25_ebeae15a-90e5-4975-a5cd-ddd640c8c977_page4.png")
    img = load_image(test_path)
    stages = preprocess_for_text(img)

    out_dir = Path("data/debug_price_boxes") / test_path.stem / "contour_detector"
    bboxes, _ = detect_price_regions_from_morph(
        morph=stages["morph"],
        original_bgr=img,
        config=PriceRegionDetectionConfig(),
        debug_dir=out_dir,
    )

    print(f"Detected {len(bboxes)} price regions")
    print(f"Debug outputs written to: {out_dir}")
