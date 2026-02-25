from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import numpy as np

# PriceDetection is just a Dict - no need to import
PriceDetection = Dict[str, object]
Result = Dict[str, object]


def convert_numpy_types(obj):
	"""Convert numpy types to native Python types for JSON serialization."""
	if isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	elif isinstance(obj, dict):
		return {key: convert_numpy_types(value) for key, value in obj.items()}
	elif isinstance(obj, (list, tuple)):
		return [convert_numpy_types(item) for item in obj]
	else:
		return obj


def to_result_record(image_path: Path, detections: List[PriceDetection]) -> Result:
	return {
		"page_id": image_path.stem,
		"image": str(image_path),
		"items": convert_numpy_types(detections),
	}


def _to_ground_truth_like_record(result: Result) -> Dict[str, object]:
	page_id = result.get("page_id")
	if not page_id:
		image = result.get("image", "")
		page_id = Path(str(image)).stem if image else "unknown_page"

	items = result.get("items", []) or []
	offers = []
	for item in items:
		if not isinstance(item, dict):
			continue
		product = item.get("product")
		price = item.get("price")
		if product and price:
			offers.append({"product": product, "price": price})

	return {
		"page_id": page_id,
		"offers": offers,
	}


def save_results(results: List[Result], output: Path) -> None:
	output.parent.mkdir(parents=True, exist_ok=True)
	formatted = [_to_ground_truth_like_record(r) for r in results]
	if len(formatted) == 1:
		payload = formatted[0]
	else:
		payload = formatted
	output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
