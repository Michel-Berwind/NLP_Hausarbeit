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
		"image": str(image_path),
		"items": convert_numpy_types(detections),
	}


def save_results(results: List[Result], output: Path) -> None:
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
