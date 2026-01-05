from __future__ import annotations

from typing import List, Tuple
import re

import cv2
import numpy as np
import pytesseract


# Known brand names commonly found on German supermarket flyers
KNOWN_BRANDS = [
	"BARISSIMO", "BARISTA", "NESCAFE", "JACOBS", "DALLMAYR", "MELITTA",
	"LAVAZZA", "TCHIBO", "CREMESSO", "SENSEO", "DOLCE GUSTO", "TASSIMO",
	"MILKA", "LINDT", "RITTER SPORT", "HARIBO", "NUTELLA",
]


def clean_line(text: str) -> str:
	"""Clean OCR text, keeping German characters and common punctuation."""
	text = text.strip()
	# Keep German umlauts, letters, digits, and common punctuation
	text = re.sub(r'[^A-Za-zÄÖÜäöüß0-9 ,./&\-]', " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def score_line(text: str) -> float:
	"""Score a text line for quality as a product name."""
	if not text:
		return 0.0
	
	# Count character types
	letters = len(re.findall(r"[A-Za-zÄÖÜäöüß]", text))
	digits = len(re.findall(r"[0-9]", text))
	spaces = text.count(" ")
	length = len(text)
	
	# Calculate ratios
	alpha_ratio = letters / max(length, 1)
	digit_ratio = digits / max(length, 1)
	space_ratio = spaces / max(length, 1)
	
	# Filter out bad candidates
	if length < 4:  # Too short
		return 0.0
	if alpha_ratio < 0.40:  # Not enough letters (relaxed from 0.30)
		return 0.0
	if digit_ratio > 0.50:  # Too many digits (relaxed slightly)
		return 0.0
	if space_ratio > 0.5:  # Too many spaces (fragmented)
		return 0.0
	
	# Penalize single character words (OCR noise)
	words = text.split()
	if len(words) > 1:
		single_char_words = sum(1 for w in words if len(w) == 1)
		if single_char_words / len(words) > 0.6:  # Relaxed from 0.5
			return 0.0
	
	# Base score from length and letter ratio
	score = length * (0.7 * alpha_ratio + 0.3)
	score -= 3.0 * digit_ratio * length  # Reduced penalty for digits
	
	# Bonus for reasonable length product names (8-60 characters)
	if 8 <= length <= 60:
		score += 5.0
	
	# Bonus for having multiple words (more likely to be product names)
	if len(words) >= 2:
		score += 4.0
	
	# Strong bonus for recognized brand names
	text_upper = text.upper()
	for brand in KNOWN_BRANDS:
		if brand in text_upper:
			score += 15.0
			break
	
	# Bonus for German product-related words
	product_keywords = ["Kaffee", "Espresso", "Bohnen", "Packung", "Kapseln", 
						"Lungo", "Ristretto", "Cremoso", "Gustoso", "Glas",
						"Mahlkaffee", "Gold", "Bester", "Aluminium", "Plastik"]
	for keyword in product_keywords:
		if keyword.lower() in text.lower():
			score += 8.0
			break
	
	return score


def extract_product_text(img: np.ndarray, box: List[int]) -> str:
	x, y, w, h = box
	H, W = img.shape[:2]
	price_like = re.compile(r"\d+[.,]\d{2}")

	def best_text_from_crop(crop_bgr: np.ndarray, region_name: str = "") -> Tuple[str, float]:
		"""Extract best text from a crop using multiple preprocessing strategies."""
		if crop_bgr.size == 0:
			return "", 0
		
		# Resize for better OCR - use higher scale for small regions
		scale = 3.0 if min(crop_bgr.shape[:2]) < 100 else 2.5
		resized = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		
		# Apply denoising - reduce aggressiveness to preserve text
		denoised = cv2.bilateralFilter(gray, 9, 50, 50)
		
		# Enhance contrast
		clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
		enhanced = clahe.apply(denoised)
		
		# Try multiple binarization strategies
		thr_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		thr_adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									cv2.THRESH_BINARY, 31, 10)
		thr_adapt2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
									cv2.THRESH_BINARY, 25, 12)
		
		# Inverted versions for white text on dark background
		thr_otsu_inv = cv2.bitwise_not(thr_otsu)
		thr_adapt_inv = cv2.bitwise_not(thr_adapt)
		
		# Also try on gray directly with adaptive threshold
		thr_gray_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									cv2.THRESH_BINARY, 21, 8)
		
		candidates = [thr_otsu, thr_adapt, thr_adapt2, thr_otsu_inv, thr_adapt_inv, thr_gray_adapt]
		best_combined, best_score = "", 0.0

		def extract_all_good_lines(bin_img: np.ndarray, cfg: str) -> Tuple[str, float]:
			"""Extract and combine all good lines from OCR output."""
			good_lines = []
			total_score = 0.0
			try:
				text = pytesseract.image_to_string(bin_img, lang="deu", config=cfg)
				for ln in text.splitlines():
					ln = clean_line(ln)
					if not ln or len(ln) < 3:
						continue
					# Skip lines that look like prices
					if price_like.search(ln):
						continue
					# Skip lines that are mostly digits
					if sum(c.isdigit() for c in ln) > len(ln) * 0.5:
						continue
					# Skip very short fragments
					if len(ln) < 4:
						continue
					score = score_line(ln)
					if score > 5.0:  # Only include lines with decent score
						good_lines.append((ln, score))
						total_score += score
			except Exception:
				pass
			
			if not good_lines:
				return "", 0.0
			
			# Sort by score and take top lines
			good_lines.sort(key=lambda x: x[1], reverse=True)
			
			# Combine top lines (max 3) that together make a product name
			combined_lines = []
			for ln, score in good_lines[:3]:
				# Don't duplicate content
				if not any(ln.lower() in existing.lower() or existing.lower() in ln.lower() 
						   for existing in combined_lines):
					combined_lines.append(ln)
			
			# Join lines with space
			combined = " ".join(combined_lines)
			
			# Calculate combined score (highest single + bonus for multi-line)
			if combined_lines:
				best_single_score = good_lines[0][1]
				combined_score = best_single_score + (len(combined_lines) - 1) * 3.0
				return combined, combined_score
			
			return "", 0.0

		# Try different PSM modes for different text layouts
		psm_configs = [
			r"--oem 3 --psm 6",  # Uniform block of text
			r"--oem 3 --psm 4",  # Single column of text
			r"--oem 3 --psm 11", # Sparse text
			r"--oem 3 --psm 3",  # Fully automatic
		]
		
		for bin_img in candidates:
			for cfg in psm_configs:
				combined, combined_score = extract_all_good_lines(bin_img, cfg)
				if combined_score > best_score:
					best_combined, best_score = combined, combined_score
			
			# Early exit if we found really good text (brand + keyword)
			if best_score > 40:
				break

		return best_combined, best_score

	# Define multiple search regions around the price box
	# Adjusted for ALDI flyer layout where product text is often to the left
	regions = []
	
	# Region 1: LEFT - Primary region for this flyer type
	# Product text is typically to the left of the price box
	# Make this region wider and taller to capture full product descriptions
	x1_left = max(0, x - int(0.05 * w))  # Small gap from price box
	x0_left = max(0, x - int(3.5 * w))   # Extend far left (increased from 2.0)
	y0_left = max(0, y - int(1.0 * h))   # Above price box center
	y1_left = min(H, y + h + int(1.0 * h))  # Below price box center
	if x1_left - x0_left >= 60 and y1_left - y0_left >= 30:
		regions.append(("left", img[y0_left:y1_left, x0_left:x1_left]))
	
	# Region 2: ABOVE - Also common, text above the price
	x0_above = max(0, x - int(0.8 * w))
	x1_above = min(W, x + w + int(0.8 * w))
	y1_above = max(0, y - int(0.05 * h))
	y0_above = max(0, y - int(2.5 * h))
	if y1_above - y0_above >= 30 and x1_above - x0_above >= 60:
		regions.append(("above", img[y0_above:y1_above, x0_above:x1_above]))
	
	# Region 3: LEFT-ABOVE diagonal - for corner layouts
	x0_la = max(0, x - int(2.5 * w))
	x1_la = max(0, x - int(0.1 * w))
	y0_la = max(0, y - int(1.5 * h))
	y1_la = max(0, y + int(0.3 * h))
	if x1_la - x0_la >= 60 and y1_la - y0_la >= 30:
		regions.append(("left-above", img[y0_la:y1_la, x0_la:x1_la]))
	
	# Region 4: RIGHT (for some layouts)
	x0_right = min(W, x + w + int(0.05 * w))
	x1_right = min(W, x + w + int(2.5 * w))
	y0_right = max(0, y - int(0.5 * h))
	y1_right = min(H, y + h + int(0.5 * h))
	if x1_right - x0_right >= 60 and y1_right - y0_right >= 30:
		regions.append(("right", img[y0_right:y1_right, x0_right:x1_right]))
	
	# Region 5: BELOW (less common but possible)
	x0_below = max(0, x - int(0.5 * w))
	x1_below = min(W, x + w + int(0.5 * w))
	y0_below = min(H, y + h + int(0.05 * h))
	y1_below = min(H, y + h + int(1.5 * h))
	if y1_below - y0_below >= 30 and x1_below - x0_below >= 60:
		regions.append(("below", img[y0_below:y1_below, x0_below:x1_below]))
	
	# Extract text from all regions and pick the best one
	best_line, best_score, best_region = "", 0.0, ""
	all_candidates = []
	
	for region_name, reg in regions:
		line, score = best_text_from_crop(reg, region_name)
		all_candidates.append((region_name, line, score))
		if score > best_score:
			best_line, best_score, best_region = line, score, region_name
	
	# Debug output for troubleshooting
	# print(f"    Candidates: {[(r, l[:30] if l else '', f'{s:.1f}') for r, l, s in all_candidates]}")
	
	# If we found good text, clean it up and return
	if best_score > 0:
		# print(f"    Best text from {best_region}: {best_line} (score: {best_score:.1f})")
		return postprocess_product_text(best_line)
	return ""


def postprocess_product_text(text: str) -> str:
	"""Clean up and format the extracted product text."""
	if not text:
		return ""
	
	# Remove common OCR noise patterns
	noise_patterns = [
		r"^[^A-Za-zÄÖÜäöüß]+",  # Remove leading non-letters
		r"[^A-Za-zÄÖÜäöüß0-9]+$",  # Remove trailing noise
		r"\s+[a-zA-Z]\s+",  # Remove isolated single letters (OCR noise)
		r"Aus unserem S\w*",  # Remove "Aus unserem Sortiment" header noise
		r"Aus unsere\w*",
		r"\bSe$",  # Trailing "Se" from "Sortiment"
		r"\s+ä\s+",  # Single 'ä' often OCR noise
		r"\s+E\s+",  # Single 'E' often OCR noise
		r"\s+f\s+",  # Single 'f' often OCR noise
		r"\s+Y\s+",  # Single 'Y' often OCR noise  
		r"\bmrm\b",  # Common OCR misread
		r"\bsster\b",  # Common OCR misread
		r"^[\s,.\-]+",  # Leading punctuation/spaces
		r"^\d+\s+",  # Leading numbers
		r"^[d\.]+\s*[\.\-]+\s*",  # Pattern like "d. ...-.. "
		r"je\s+20\s+Aluminium-\s*\d*\s*",  # Remove package details
		r"\s+\d\s+",  # Remove isolated digits (noise)
	]
	
	result = text
	for pattern in noise_patterns:
		result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
	
	# Clean up multiple spaces
	result = re.sub(r"\s+", " ", result).strip()
	
	# Fix common OCR errors
	ocr_corrections = [
		(r"500-\s*-Packung", "500-g-Packung"),
		(r"Kaffeeka\s*Pseln", "Kaffeekapseln"),
		(r"\bRISSIMO\b", "BARISSIMO"),
		(r",\.-+", ""),
		(r"2x\s*250\s*500-g-Packung", "2x 250g = 500-g-Packung"),  # Fix weight info
	]
	for pattern, replacement in ocr_corrections:
		result = re.sub(pattern, replacement, result)
	
	# Remove duplicate brand mentions (keep first occurrence)
	for brand in KNOWN_BRANDS:
		# Count occurrences
		count = len(re.findall(rf"\b{brand}\b", result, flags=re.IGNORECASE))
		if count > 1:
			# Keep only first occurrence
			first_match = re.search(rf"\b{brand}\b", result, flags=re.IGNORECASE)
			if first_match:
				before = result[:first_match.end()]
				after = result[first_match.end():]
				after = re.sub(rf"\b{brand}\b", "", after, flags=re.IGNORECASE)
				result = before + after
	
	# Clean up multiple spaces again
	result = re.sub(r"\s+", " ", result).strip()
	
	# Remove trailing punctuation and noise
	result = re.sub(r"[,.\-]+$", "", result).strip()
	
	# Remove leading punctuation
	result = re.sub(r"^[,.\-\d]+\s*", "", result).strip()
	
	return result
