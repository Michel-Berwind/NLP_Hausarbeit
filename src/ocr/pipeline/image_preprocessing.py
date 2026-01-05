from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pytesseract


# Allow overriding the tesseract binary via env var if needed; fall back to the
# user specific installation that was used for the other scripts.
DEFAULT_TESSERACT = Path(r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")


def configure_tesseract(custom_binary: Optional[Path] = None) -> None:
	"""Configure the tesseract binary if a local install is present."""
	binary = custom_binary or DEFAULT_TESSERACT
	if binary.exists():
		pytesseract.pytesseract.tesseract_cmd = str(binary)


def load_image(image_path: Path) -> np.ndarray:
	"""Read an image from disk and fail fast if missing."""
	img = cv2.imread(str(image_path))
	if img is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
	"""Convert BGR to grayscale for downstream OCR steps."""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
	"""Apply CLAHE to boost local contrast without over-amplifying noise."""
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
	return clahe.apply(gray)


def adaptive_binarize(gray: np.ndarray, block_size: int = 21, c: int = 8) -> np.ndarray:
	"""Adaptive thresholding to keep text strokes while handling uneven lighting."""
	return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def morph_refine(bin_img: np.ndarray) -> np.ndarray:
	"""Clean binary image: close gaps then open small noise."""
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
	opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
	return opened


def morph_for_pricebox_detection(bin_img: np.ndarray) -> np.ndarray:
	"""
	Morphological processing to connect text into rectangular price box regions.
	
	This is different from morph_refine which cleans text for OCR.
	Here we want to merge nearby characters/words into solid rectangular blocks
	that represent price tags or labels.
	
	Steps:
	1. Invert binary (text becomes white on black background)
	2. Dilate horizontally to connect characters into words
	3. Dilate vertically to connect lines into blocks
	4. Close to fill gaps
	"""
	# Invert: adaptive threshold gives white bg / dark text, we need white text
	inverted = 255 - bin_img
	
	# Horizontal dilation: connect characters into price numbers (e.g., "2.99")
	# Smaller kernel to avoid connecting separate price tags
	kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
	dilated_h = cv2.dilate(inverted, kernel_h, iterations=1)
	
	# Vertical dilation: small amount to connect multi-line labels
	kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
	dilated_v = cv2.dilate(dilated_h, kernel_v, iterations=1)
	
	# Close to fill small holes within the detected regions
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	closed = cv2.morphologyEx(dilated_v, cv2.MORPH_CLOSE, kernel_close, iterations=1)
	
	return closed


def preprocess_for_text(img: np.ndarray) -> Dict[str, np.ndarray]:
	"""Full text-focused preprocessing pipeline to highlight rectangular text regions."""
	gray = to_grayscale(img)
	enhanced = enhance_contrast(gray)
	binary = adaptive_binarize(enhanced)
	morph_text = morph_refine(binary)  # For OCR (cleaned text)
	morph = morph_for_pricebox_detection(binary)  # For detection (connected blocks)
	return {
		"gray": gray,
		"enhanced": enhanced,
		"binary": binary,
		"morph_text": morph_text,  # Use this for OCR
		"morph": morph,  # Use this for price box detection
	}


def preprocess_for_detection(img: np.ndarray, relaxed: bool = False) -> Dict[str, np.ndarray]:
	"""Compute detection-oriented masks (white, blue, dark, combined text) upfront."""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	L = lab[:, :, 0]

	white_low = (0, 0, 170) if relaxed else (0, 0, 180)
	white_high = (180, 70, 255) if relaxed else (180, 60, 255)
	L_min = 150 if relaxed else 180

	mask_white = cv2.inRange(hsv, white_low, white_high)
	mask_white2 = (L > L_min).astype(np.uint8) * 255
	mask_white = cv2.bitwise_and(mask_white, mask_white2)
	mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=2)

	blue_low = (75, 5, 80) if relaxed else (90, 20, 120)
	blue_high = (165, 255, 255) if relaxed else (150, 255, 255)
	dark_low = (0, 0, 0)
	dark_high = (180, 255, 140) if relaxed else (180, 255, 110)

	blue_mask = cv2.inRange(hsv, blue_low, blue_high)
	dark_mask = cv2.inRange(hsv, dark_low, dark_high)
	text_mask = cv2.bitwise_or(blue_mask, dark_mask)

	return {
		"lab": lab,
		"L": L,
		"hsv": hsv,
		"white_mask": mask_white,
		"blue_mask": blue_mask,
		"dark_mask": dark_mask,
		"text_mask": text_mask,
	}


def save_preprocess_debug(image_path: Path, debug_root: Path, stages: Dict[str, np.ndarray]) -> Path:
	"""Persist any preprocessing stages (text + detection masks) for inspection."""
	out_dir = debug_root / "preprocess" / image_path.stem
	out_dir.mkdir(parents=True, exist_ok=True)
	for name, img in stages.items():
		cv2.imwrite(str(out_dir / f"{name}.png"), img)
	return out_dir
