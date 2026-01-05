"""
Price Box Detection Module

Detects rectangular price label regions on supermarket flyer pages using
OpenCV contour analysis with explainable, deterministic heuristics.

This module is designed for Master thesis work, emphasizing:
- Explainability: all thresholds are named and justified
- Determinism: no ML, no randomness
- Robustness: sensible defaults for ~2000x3000 ALDI-style flyers
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION: Tunable Detection Parameters
# =============================================================================
# All thresholds are configurable here. Defaults are calibrated for ALDI flyers.

class PriceBoxConfig:
	"""Configuration for price box detection heuristics."""
	
	# Area constraints (absolute pixels)
	MIN_AREA_PX = 8000       # Minimum box area in pixels (increased to filter small text)
	
	# Absolute size constraints (pixels)
	MIN_WIDTH_PX = 120       # Minimum width to be a valid price box (increased)
	MIN_HEIGHT_PX = 80       # Minimum height to be a valid price box (increased)
	MAX_WIDTH_RATIO = 0.25   # Max width as fraction of image width (25% - even tighter)
	MAX_HEIGHT_RATIO = 0.20  # Max height as fraction of image height (20% - even tighter)
	
	# Aspect ratio ranges (width/height)
	MIN_ASPECT_RATIO = 0.8   # Minimum aspect ratio (more square)
	MAX_ASPECT_RATIO = 2.5   # Maximum aspect ratio (less elongated)
	
	# Color-based filters (for BLUE background detection - inverted logic)
	WHITE_HSV_LOW = (85, 100, 100)   # HSV lower bound for BLUE boxes (wider to include cyan-ish blues)
	WHITE_HSV_HIGH = (135, 255, 255) # HSV upper bound for BLUE boxes
	WHITE_LAB_L_MIN = 50             # Minimum LAB L-channel for blue (darker than white)
	
	# Text detection (WHITE text on blue background)
	BLUE_HSV_LOW = (0, 0, 180)       # HSV lower bound for WHITE text
	BLUE_HSV_HIGH = (180, 50, 255)   # HSV upper bound for WHITE text
	DARK_HSV_LOW = (0, 0, 200)       # HSV lower bound for bright text
	DARK_HSV_HIGH = (180, 30, 255)   # HSV upper bound for bright text
	
	# Text ratio filter
	MIN_TEXT_RATIO = 0.05    # At least 5% of box should be text pixels
	
	# Brightness filter (inverted - now looking for darker blue boxes)
	MIN_MEAN_LAB_L = 40      # Minimum mean LAB L for blue background (darker)
	
	# Contrast filter (inverted - text lighter than background)
	MIN_CONTRAST = 30        # Minimum L-channel contrast between bg and text
	
	# Box merging parameters
	IOU_MERGE_THRESHOLD = 0.3  # Merge boxes with >30% overlap (IoU)
	
	# Debug visualization
	DEBUG_BOX_COLOR = (0, 255, 0)      # Green for bounding boxes
	DEBUG_BOX_THICKNESS = 3
	DEBUG_CONTOUR_COLOR = (255, 0, 0)  # Blue for contours
	DEBUG_TEXT_COLOR = (255, 255, 0)   # Cyan for box labels


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
	"""
	Compute Intersection over Union (IoU) of two bounding boxes.
	
	Args:
		box1, box2: Bounding boxes as (x, y, w, h)
	
	Returns:
		IoU value in [0, 1]
	"""
	x1, y1, w1, h1 = box1
	x2, y2, w2, h2 = box2
	
	# Compute intersection rectangle
	xi1 = max(x1, x2)
	yi1 = max(y1, y2)
	xi2 = min(x1 + w1, x2 + w2)
	yi2 = min(y1 + h1, y2 + h2)
	
	inter_width = max(0, xi2 - xi1)
	inter_height = max(0, yi2 - yi1)
	inter_area = inter_width * inter_height
	
	# Compute union
	box1_area = w1 * h1
	box2_area = w2 * h2
	union_area = box1_area + box2_area - inter_area
	
	if union_area == 0:
		return 0.0
	return inter_area / union_area


def merge_overlapping_boxes(
	boxes: List[Tuple[int, int, int, int]],
	iou_threshold: float = PriceBoxConfig.IOU_MERGE_THRESHOLD
) -> List[Tuple[int, int, int, int]]:
	"""
	Merge overlapping bounding boxes using IoU-based clustering.
	
	Strategy: For each box, find all boxes that overlap significantly (IoU > threshold)
	and merge them into a single bounding box that encompasses all.
	
	Args:
		boxes: List of bounding boxes (x, y, w, h)
		iou_threshold: Minimum IoU to consider boxes as overlapping
	
	Returns:
		List of merged bounding boxes
	"""
	if not boxes:
		return []
	
	# Track which boxes have been merged
	merged = [False] * len(boxes)
	result = []
	
	for i, box1 in enumerate(boxes):
		if merged[i]:
			continue
		
		# Start a new cluster with this box
		cluster = [box1]
		merged[i] = True
		
		# Find all boxes that overlap with this cluster
		for j, box2 in enumerate(boxes):
			if merged[j]:
				continue
			
			# Check if box2 overlaps with any box in the cluster
			for cluster_box in cluster:
				if compute_iou(cluster_box, box2) > iou_threshold:
					cluster.append(box2)
					merged[j] = True
					break
		
		# Merge all boxes in the cluster into one encompassing box
		x_min = min(x for x, y, w, h in cluster)
		y_min = min(y for x, y, w, h in cluster)
		x_max = max(x + w for x, y, w, h in cluster)
		y_max = max(y + h for x, y, w, h in cluster)
		
		merged_box = (x_min, y_min, x_max - x_min, y_max - y_min)
		result.append(merged_box)
	
	return result


def sort_boxes_reading_order(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
	"""
	Sort bounding boxes in reading order: top-to-bottom, left-to-right.
	
	Strategy: Primary sort by y-coordinate (top to bottom), secondary by x (left to right).
	Groups boxes that are on roughly the same horizontal line (within 20px tolerance).
	
	Args:
		boxes: List of bounding boxes (x, y, w, h)
	
	Returns:
		Sorted list of bounding boxes
	"""
	if not boxes:
		return []
	
	# Sort primarily by y, secondarily by x
	# Group boxes with similar y values (same row)
	sorted_boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))
	return sorted_boxes


def filter_white_box(
	stat: Tuple[int, int, int, int, int],
	img: np.ndarray,
	lab: np.ndarray,
	text_mask: np.ndarray,
	config: PriceBoxConfig = PriceBoxConfig(),
	is_white_background: bool = False
) -> Tuple[bool, str, Tuple[int, int, int, int]]:
	"""
	Apply color-based heuristics to filter a white region candidate.
	
	This targets price labels with bright backgrounds and visible text.
	
	Args:
		stat: Connected component stats (x, y, w, h, area)
		img: Original BGR image
		lab: LAB colorspace conversion of img
		text_mask: Binary mask of text pixels (blue/red/dark for white bg, white for blue bg)
		config: Detection configuration
		is_white_background: If True, validates white boxes with dark text; if False, validates blue boxes with light text
	
	Returns:
		(is_valid, rejection_reason, bounding_box)
	"""
	x, y, w, h, area = stat
	H, W = img.shape[:2]
	
	# Filter 1: Area constraint
	if area < config.MIN_AREA_PX:
		return False, f"too_small (area={area} < {config.MIN_AREA_PX})", (x, y, w, h)
	
	# Filter 2: Absolute size constraints
	if w < config.MIN_WIDTH_PX or h < config.MIN_HEIGHT_PX:
		return False, f"size_too_small ({w}x{h})", (x, y, w, h)
	
	max_w = config.MAX_WIDTH_RATIO * W
	max_h = config.MAX_HEIGHT_RATIO * H
	if w > max_w or h > max_h:
		return False, f"size_too_large ({w}x{h})", (x, y, w, h)
	
	# Filter 3: Aspect ratio
	aspect = w / max(h, 1)
	if aspect < config.MIN_ASPECT_RATIO or aspect > config.MAX_ASPECT_RATIO:
		return False, f"bad_aspect_ratio ({aspect:.2f})", (x, y, w, h)
	
	# Filter 4: Require text pixels
	roi_text = text_mask[y:y+h, x:x+w]
	text_ratio = float((roi_text > 0).mean())
	if text_ratio < config.MIN_TEXT_RATIO:
		return False, f"no_text (text_ratio={text_ratio:.3f})", (x, y, w, h)
	
	# Filter 5: Brightness check (different logic for white vs blue backgrounds)
	roi_lab = lab[y:y+h, x:x+w, :]
	mean_L = roi_lab[:, :, 0].mean()
	
	if is_white_background:
		# For white boxes: background should be bright (L > 160)
		if mean_L < 160:
			return False, f"not_bright_enough (mean_L={mean_L:.1f})", (x, y, w, h)
	else:
		# For blue boxes: background should be darker (L between MIN and 180)
		if mean_L < config.MIN_MEAN_LAB_L:
			return False, f"too_dark (mean_L={mean_L:.1f})", (x, y, w, h)
		if mean_L > 180:
			return False, f"too_bright (mean_L={mean_L:.1f})", (x, y, w, h)
	
	# Filter 6: Contrast check (inverted for white backgrounds)
	text_mask_roi = (roi_text > 0)
	if text_mask_roi.sum() == 0:
		return False, "no_text_pixels", (x, y, w, h)
	
	text_L = roi_lab[:, :, 0][text_mask_roi].mean()
	
	if is_white_background:
		# White background with dark text: text should be DARKER than background
		contrast = mean_L - text_L
	else:
		# Blue background with white text: text should be LIGHTER than background
		contrast = text_L - mean_L
	
	if contrast < config.MIN_CONTRAST:
		return False, f"low_contrast ({contrast:.1f})", (x, y, w, h)
	
	# All filters passed
	return True, "", (x, y, w, h)


def detect_price_boxes(
	img: np.ndarray,
	morph: np.ndarray,
	debug_dir: Path = None,
	config: PriceBoxConfig = PriceBoxConfig(),
	detect_white_boxes: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
	"""
	Detect price box regions using color-based approach.
	
	Can detect either:
	- Blue boxes with white text (default)
	- White boxes with blue/red text (when detect_white_boxes=True)
	
	Pipeline:
	1. Convert to HSV and LAB colorspaces
	2. Find colored regions (blue or white backgrounds)
	3. Find text regions (white or blue/red text)
	4. For each box, check if it contains text and has good contrast
	5. Merge overlapping boxes
	6. Sort boxes in reading order
	7. Extract cropped ROIs for OCR
	8. Generate debug visualizations
	
	Args:
		img: Original BGR image
		morph: Unused (kept for API compatibility)
		debug_dir: Directory to save debug images (optional)
		config: Detection configuration parameters
		detect_white_boxes: If True, detect white boxes with colored text instead
	
	Returns:
		(boxes, crops)
		- boxes: List of bounding boxes (x, y, w, h) in reading order
		- crops: List of cropped ROI images (BGR) for OCR
	"""
	H, W = img.shape[:2]
	
	# Step 1: Convert to HSV and LAB colorspaces
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	L = lab[:, :, 0]
	
	if detect_white_boxes:
		# Detect WHITE boxes with BLUE/RED text
		# Step 2: Find WHITE regions (bright background)
		mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
		mask_white2 = (L > 180).astype(np.uint8) * 255
		mask_white = cv2.bitwise_and(mask_white, mask_white2)
		
		# Step 3: Find BLUE/RED text regions
		blue_text = cv2.inRange(hsv, (100, 50, 50), (135, 255, 255))  # Blue text
		red_text1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))     # Red text (low hue)
		red_text2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))  # Red text (high hue)
		dark_text = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))       # Dark text
		text_mask = cv2.bitwise_or(cv2.bitwise_or(blue_text, cv2.bitwise_or(red_text1, red_text2)), dark_text)
	else:
		# Detect BLUE boxes with WHITE text (original behavior)
		# Step 2: Find BLUE regions (price boxes with blue background)
		mask_white = cv2.inRange(hsv, config.WHITE_HSV_LOW, config.WHITE_HSV_HIGH)
		mask_white2 = (L > config.WHITE_LAB_L_MIN).astype(np.uint8) * 255
		mask_white = cv2.bitwise_and(mask_white, mask_white2)
		
		# Step 3: Find WHITE text regions (white text on blue background)
		blue_mask = cv2.inRange(hsv, config.BLUE_HSV_LOW, config.BLUE_HSV_HIGH)
		dark_mask = cv2.inRange(hsv, config.DARK_HSV_LOW, config.DARK_HSV_HIGH)
		text_mask = cv2.bitwise_or(blue_mask, dark_mask)
	
	# Use smaller morphology kernel to preserve sharp edges of price boxes
	mask_white = cv2.morphologyEx(
		mask_white,
		cv2.MORPH_CLOSE,
		cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
		iterations=1
	)
	
	# Debug: Save masks
	if debug_dir:
		debug_dir.mkdir(parents=True, exist_ok=True)
		suffix = "_white" if detect_white_boxes else ""
		cv2.imwrite(str(debug_dir / f"01_box_mask{suffix}.png"), mask_white)
		cv2.imwrite(str(debug_dir / f"01_text_mask{suffix}.png"), text_mask)
	
	# Step 4: Find connected components in mask
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		mask_white, connectivity=8
	)
	
	print(f"  Found {num_labels - 1} white regions")
	
	# Filter each white region
	valid_boxes = []
	rejected_boxes = []
	
	for i in range(1, num_labels):  # Skip background (label 0)
		stat = tuple(stats[i])
		is_valid, reason, box = filter_white_box(stat, img, lab, text_mask, config, is_white_background=detect_white_boxes)
		
		if is_valid:
			valid_boxes.append(box)
		else:
			rejected_boxes.append((box, reason))
	
	print(f"  Filtered to {len(valid_boxes)} valid boxes ({len(rejected_boxes)} rejected)")
	
	# Debug: Draw boxes before merging
	if debug_dir:
		debug_raw = img.copy()
		for x, y, w, h in valid_boxes:
			cv2.rectangle(debug_raw, (x, y), (x + w, y + h), config.DEBUG_BOX_COLOR, 2)
		cv2.imwrite(str(debug_dir / "02_boxes_before_merge.png"), debug_raw)
	
	# Step 5: Merge overlapping boxes
	merged_boxes = merge_overlapping_boxes(valid_boxes, config.IOU_MERGE_THRESHOLD)
	print(f"  Merged to {len(merged_boxes)} final boxes")
	
	# Step 6: Sort in reading order
	sorted_boxes = sort_boxes_reading_order(merged_boxes)
	
	# Step 7: Extract crops
	crops = []
	for x, y, w, h in sorted_boxes:
		# Ensure coordinates are within image bounds
		x = max(0, x)
		y = max(0, y)
		w = min(w, W - x)
		h = min(h, H - y)
		
		crop = img[y:y+h, x:x+w].copy()
		crops.append(crop)
	
	# Debug: Draw final boxes with labels
	if debug_dir:
		debug_final = img.copy()
		for i, (x, y, w, h) in enumerate(sorted_boxes):
			cv2.rectangle(debug_final, (x, y), (x + w, y + h), config.DEBUG_BOX_COLOR, config.DEBUG_BOX_THICKNESS)
			# Label with index
			label = f"#{i}"
			label_pos = (x + 5, y + 25)
			cv2.putText(debug_final, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.DEBUG_TEXT_COLOR, 2)
		cv2.imwrite(str(debug_dir / "03_boxes_final.png"), debug_final)
		
		# Save individual crops
		crops_dir = debug_dir / "crops"
		crops_dir.mkdir(exist_ok=True)
		for i, crop in enumerate(crops):
			cv2.imwrite(str(crops_dir / f"box_{i:02d}.png"), crop)
	
	return sorted_boxes, crops


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

def detect_prices(
	img: np.ndarray,
	debug_dir: Path,
	relaxed: bool = False,
	preprocess_stages: Dict[str, np.ndarray] = None
) -> List[dict]:
	"""
	Detect price boxes from preprocessed image stages.
	
	This function integrates with the pipeline runner by accepting
	preprocessed stages and returning structured detections.
	
	Args:
		img: Original BGR image
		debug_dir: Directory for debug outputs
		relaxed: Whether relaxed detection mode is used (for config adjustment)
		preprocess_stages: Dictionary of preprocessed images from image_preprocessing
	
	Returns:
		List of detection dictionaries with keys:
		- "box": (x, y, w, h) bounding box
		- "confidence": always 1.0 for deterministic detection
		- "crop": cropped image (BGR) for OCR
	"""
	if preprocess_stages is None or "morph" not in preprocess_stages:
		raise ValueError("preprocess_stages must contain 'morph' image")
	
	morph = preprocess_stages["morph"]
	
	# Optionally adjust config for relaxed mode
	config = PriceBoxConfig()
	if relaxed:
		# Relax constraints slightly for relaxed mode
		config.MIN_AREA_RATIO = 0.00005  # Half the strict minimum
		config.MIN_RECTANGULARITY = 0.4  # Allow slightly less rectangular shapes
	
	# Detect boxes
	boxes_blue, crops_blue = detect_price_boxes(img, morph, debug_dir, config, detect_white_boxes=False)
	boxes_white, crops_white = detect_price_boxes(img, morph, debug_dir / "white_boxes" if debug_dir else None, config, detect_white_boxes=True)
	
	# Combine results from both detection passes
	boxes = boxes_blue + boxes_white
	crops = crops_blue + crops_white
	
	print(f"  Detected {len(boxes_blue)} blue boxes and {len(boxes_white)} white boxes")
	
	# Convert to pipeline format (don't include crop images in JSON)
	detections = []
	for box, crop in zip(boxes, crops):
		# Extract price from the box crop
		price = extract_price_from_crop(crop)
		
		detections.append({
			"box": box,
			"confidence": 1.0,  # Deterministic detection = full confidence
			"price": price,
			# Note: crop is saved to debug_dir/crops/ but not included in JSON
		})
	
	return detections


def extract_price_from_crop(crop: np.ndarray) -> str:
	"""
	Extract price text from a blue price box crop using OCR.
	
	Args:
		crop: BGR image of the price box
	
	Returns:
		Extracted price string (e.g., "2.99") or empty string if not found
	"""
	import pytesseract
	import re
	
	if crop.size == 0:
		return ""
	
	# Preprocess: resize larger for better OCR
	resized = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	
	# Collect all digit sequences from different preprocessing attempts
	all_digits = []
	
	# Try multiple preprocessing approaches
	inverted = 255 - gray
	_, binary_otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, binary_manual = cv2.threshold(inverted, 120, 255, cv2.THRESH_BINARY)
	_, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	binaries = [binary_otsu, binary_manual, binary_inv]
	
	# Try each binary with multiple PSM modes
	for binary in binaries:
		for psm in [6, 7, 11, 13]:
			# First try to get price with decimal
			text_full = pytesseract.image_to_string(
				binary, 
				lang="eng",
				config=f"--oem 3 --psm {psm}"
			)
			
			# Extract price pattern (digits with decimal separator)
			price_pattern = r'\d{1,2}[.,]\d{2}'
			matches = re.findall(price_pattern, text_full)
			
			if matches:
				price = matches[0].replace(',', '.')
				try:
					price_val = float(price)
					if 0.10 <= price_val <= 99.99:
						return price
				except ValueError:
					pass
			
			# Collect digit sequences
			text_numeric = pytesseract.image_to_string(
				binary, 
				lang="eng",
				config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
			)
			
			digits = re.sub(r'\D', '', text_numeric)
			if digits:
				all_digits.append(digits)
	
	# Try to reconstruct price from collected digits
	# Look for patterns: 3-4 consecutive digits (e.g., "299", "1499", "599")
	for digits in all_digits:
		if 3 <= len(digits) <= 4:
			price = digits[:-2] + '.' + digits[-2:]
			try:
				price_val = float(price)
				if 0.10 <= price_val <= 99.99:
					# Special case: OCR often misreads X.99 as X27 or X29
					# If we see X.27 or X.29 and also see "99" elsewhere, prefer X.99
					if digits[-2:] in ['27', '29']:
						# Check if we have "99" in our readings
						has_99 = any('99' in d for d in all_digits)
						if has_99:
							corrected_price = digits[:-2] + '.99'
							return corrected_price
					return price
			except ValueError:
				pass
	
	# Look for 3-digit patterns within longer digit strings
	# (e.g., "500599" contains "599", "500529" might be misread "599")
	for digits in all_digits:
		if len(digits) > 4:
			# Check if we have "99" readings to correct misreads
			has_99 = any('99' in d for d in all_digits)
			
			# Try to find reasonable price patterns
			# Priority 1: Look for X99 patterns (most common price ending)
			for i in range(len(digits) - 2):
				candidate = digits[i:i+3]
				if candidate.endswith('99') or candidate.endswith('49'):
					price = candidate[0] + '.' + candidate[1:]
					try:
						price_val = float(price)
						if 0.99 <= price_val <= 99.99 and candidate[0] != '0':
							return price
					except ValueError:
						pass
				# Correct common OCR errors: X29 -> X.99, X27 -> X.99
				if (candidate.endswith('29') or candidate.endswith('27')) and has_99:
					price = candidate[0] + '.99'
					try:
						price_val = float(price)
						if 0.99 <= price_val <= 99.99 and candidate[0] != '0':
							return price
					except ValueError:
						pass
			
			# Priority 2: Any 3-digit pattern
			for i in range(len(digits) - 2):
				candidate = digits[i:i+3]
				price = candidate[0] + '.' + candidate[1:]
				try:
					price_val = float(price)
					if 0.99 <= price_val <= 99.99 and candidate[0] != '0':
						return price
				except ValueError:
					pass
	
	# Special case: combine separate reads (e.g., "1" and "49" -> "1.49")
	# Group by similar lengths
	single_digit = [d for d in all_digits if len(d) == 1]
	double_digit = [d for d in all_digits if len(d) == 2]
	
	if single_digit and double_digit:
		# Take most common of each
		from collections import Counter
		single = Counter(single_digit).most_common(1)[0][0]
		double = Counter(double_digit).most_common(1)[0][0]
		price = single + '.' + double
		try:
			price_val = float(price)
			if 0.10 <= price_val <= 99.99:
				return price
		except ValueError:
			pass
	
	return ""


# =============================================================================
# EXAMPLE USAGE (for testing)
# =============================================================================

if __name__ == "__main__":
	"""
	Process all images in the input directory.
	
	This demonstrates the typical workflow:
	1. Load images from directory
	2. Preprocess each image
	3. Detect price boxes
	4. Print summary statistics
	"""
	from pathlib import Path
	import sys
	
	# Add parent directory to path for imports
	sys.path.append(str(Path(__file__).resolve().parents[2]))
	
	from src.ocr.pipeline.image_preprocessing import (
		load_image,
		preprocess_for_text,
		save_preprocess_debug
	)
	
	# Process all images in directory
	image_dir = Path("data/images/aldi")
	debug_root = Path("data/debug_price_boxes")
	
	if not image_dir.exists():
		print(f"Image directory not found: {image_dir}")
		sys.exit(1)
	
	# Find all PNG images
	image_paths = sorted(image_dir.glob("*.png"))
	
	if not image_paths:
		print(f"No PNG images found in {image_dir}")
		sys.exit(1)
	
	print(f"Found {len(image_paths)} images to process\n")
	print("=" * 80)
	
	all_results = []
	
	for image_path in image_paths:
		print(f"\nProcessing: {image_path.name}")
		print("-" * 80)
		
		# Load and preprocess
		img = load_image(image_path)
		stages = preprocess_for_text(img)
		
		# Save preprocessing debug outputs
		preprocess_out = save_preprocess_debug(image_path, debug_root, stages)
		
		# Detect price boxes
		debug_dir = debug_root / image_path.stem
		detections = detect_prices(img, debug_dir, relaxed=False, preprocess_stages=stages)
		
		# Store results
		all_results.append({
			"image": image_path.name,
			"num_boxes": len(detections),
			"debug_dir": str(debug_dir)
		})
		
		print(f"  → Detected {len(detections)} price boxes")
		print(f"  → Debug: {debug_dir}")
	
	# Print summary
	print("\n" + "=" * 80)
	print("SUMMARY")
	print("=" * 80)
	
	for result in all_results:
		print(f"  {result['image']}: {result['num_boxes']} boxes")
	
	total_boxes = sum(r["num_boxes"] for r in all_results)
	avg_boxes = total_boxes / len(all_results) if all_results else 0
	
	print(f"\nTotal images: {len(all_results)}")
	print(f"Total boxes: {total_boxes}")
	print(f"Average boxes per image: {avg_boxes:.1f}")
	print(f"\nDebug outputs saved to: {debug_root}")
