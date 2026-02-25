from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.image_preprocessing import (
    configure_tesseract,
    load_image,
    preprocess_for_text,
    preprocess_for_detection,
    save_preprocess_debug,
)
from src.utils.json_utils import save_results, to_result_record
from src.ocr.ocr_product_text import extract_product_text, extract_product_text_with_nlp
from src.detection.pricebox_detection import detect_prices


def _is_valid_product_name(product_name: str) -> bool:
    if not product_name:
        return False

    text = product_name.strip()
    if len(text) < 5:
        return False

    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return False

    lower_text = text.lower()
    reject_fragments = {
        "aktionsartikel", "sortiment", "vormittag", "nachmittag", "dekoration",
        "beachtung", "ausverkauft", "garantie", "made in",
    }
    if any(fragment in lower_text for fragment in reject_fragments):
        return False

    non_content_tokens = {
        "GERMANY", "MADE", "IN", "GO", "HOME", "CREATION", "ALDI",
    }
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha_words:
        return False

    if all(w.upper().strip("-.,") in non_content_tokens for w in alpha_words):
        return False

    has_content_word = any(
        len(re.sub(r"[^A-Za-zÄÖÜäöüß]", "", w)) >= 4 and w.upper().strip("-.,") not in non_content_tokens
        for w in alpha_words
    )
    return has_content_word


def _is_valid_offer_detection(det: dict) -> bool:
    price = det.get("price")
    product = det.get("product")
    product_nlp = det.get("product_nlp", {})

    if not price:
        return False

    method = product_nlp.get("method", "")
    if method in {"none", "rejected_by_nlp", "rejected_after_cleaning"}:
        return False

    return _is_valid_product_name(product)


def process_image(image_path: Path, debug_root: Path = None) -> dict:
    img = load_image(image_path)
    img_debug = debug_root / image_path.stem if debug_root else None

    text_stages = preprocess_for_text(img)
    detect_stages = preprocess_for_detection(img, relaxed=False)
    stages = {**text_stages, **detect_stages}
    if debug_root:
        save_preprocess_debug(image_path, debug_root, stages)

    detections = detect_prices(img, img_debug, relaxed=False, preprocess_stages=stages)
    if not detections:
        relaxed_detect = preprocess_for_detection(img, relaxed=True)
        relaxed_stages = {**text_stages, **relaxed_detect}
        save_preprocess_debug(image_path, debug_root, relaxed_stages)
        detections = detect_prices(img, img_debug / "relaxed", relaxed=True, preprocess_stages=relaxed_stages)

    # Filter out detections without prices (not real price boxes)
    detections_with_prices = [det for det in detections if det.get("price")]
    detections_without_prices = len(detections) - len(detections_with_prices)

    if detections_without_prices > 0:
        print(f"  Filtered out {detections_without_prices} boxes without prices")

    print(f"  Extracting product text from {len(detections_with_prices)} boxes using NLP...")
    for i, det in enumerate(detections_with_prices, 1):
        print(f"    Processing box {i}/{len(detections_with_prices)}...", end=" ", flush=True)
        
        # Extract with full NLP analysis
        product_result = extract_product_text_with_nlp(img, det["box"])
        
        # Store full NLP result in detection
        det["product"] = product_result.get("product_name") or None
        det["product_nlp"] = product_result  # Full NLP analysis
        
        # Print method used
        method = product_result.get("method", "unknown")
        conf = product_result.get("confidence", 0.0)
        print(f"done ({method}, conf={conf:.2f})")

    valid_detections = [det for det in detections_with_prices if _is_valid_offer_detection(det)]
    filtered_quality = len(detections_with_prices) - len(valid_detections)
    if filtered_quality > 0:
        print(f"  Filtered out {filtered_quality} low-quality offer candidates")

    return to_result_record(image_path, valid_detections)


def run_all(input_dir: Path, debug_root: Path, output: Path, pattern: str) -> None:
    images = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    if not images:
        raise FileNotFoundError(f"No images matching {pattern} in {input_dir}")

    configure_tesseract()

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"Processing {img_path}...")
        result = process_image(img_path, debug_root)

        # Save individual result file for each image
        output_file = output.parent / f"{img_path.stem}.json"
        save_results([result], output_file)
        print(f"  Saved to {output_file}")

    print(f"\nProcessed {len(images)} images")


def run_one(input_file: Path, debug_root: Path, output: Path) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    configure_tesseract()

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_file}...")
    result = process_image(input_file, debug_root)

    # If --output is a directory, write <stem>.json there; else write exactly to --output
    if output.suffix.lower() == ".json":
        output_file = output
    else:
        output.mkdir(parents=True, exist_ok=True)
        output_file = output / f"{input_file.stem}.json"

    save_results([result], output_file)
    print(f"  Saved to {output_file}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Detect price boxes and extract product texts from poster images.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input-file", help="Path to a single poster image")
    group.add_argument("--input-dir", default="data/images/aldi", help="Directory containing poster images")

    parser.add_argument("--pattern", default="*.png", help="Glob pattern for images (used with --input-dir)")
    parser.add_argument("--debug-root", default="data/debug_price_boxes", help="Directory for debug outputs")
    parser.add_argument("--output", default="data/annotations/predictions.json", help="Path to write JSON results (or directory for single-file mode)")
    args = parser.parse_args(argv)

    if args.input_file:
        run_one(Path(args.input_file), Path(args.debug_root), Path(args.output))
    else:
        run_all(Path(args.input_dir), Path(args.debug_root), Path(args.output), args.pattern)


if __name__ == "__main__":
    main()