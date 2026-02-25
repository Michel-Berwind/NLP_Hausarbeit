from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import re

import cv2
import numpy as np
from src.preprocessing.image_preprocessing import ocr_image

# Import NLP module for entity extraction
try:
    from src.nlp.ner_model import extract_product_entities
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("Warning: NLP module not available. Install spacy with: python -m pip install spacy && python -m spacy download de_core_news_sm")

# Import text quality analysis
try:
    from src.utils.text_quality_analysis import TextQualityAnalyzer
    TEXT_QUALITY_AVAILABLE = True
except ImportError:
    TEXT_QUALITY_AVAILABLE = False
    print("Warning: Text quality analysis not available")


# Known brand names commonly found on German supermarket flyers
KNOWN_BRANDS = [
	"BARISSIMO", "BARISTA", "NESCAFE", "JACOBS", "DALLMAYR", "MELITTA",
	"LAVAZZA", "TCHIBO", "CREMESSO", "SENSEO", "DOLCE GUSTO", "TASSIMO",
	"MILKA", "LINDT", "RITTER SPORT", "HARIBO", "NUTELLA",
	"WORKZONE", "TOPCRAFT", "FERREX", "UPAFASHION", "UP2FASHION", "CRANE",
	"RIDE", "HOME", "CREATION", "HOME CREATION", "GARDENLINE", "LIVE IN STYLE",
	"BACK FAMILY", "FARMER NATURALS",
	"ALDI", "REWE", "EDEKA",
]

# Common non-product words to filter out (stopwords for products)
NON_PRODUCT_WORDS = [
	"aktionsartikel", "sortiment", "vormittag", "nachmittag", "unterschied",
	"garantie", "dekoration", "angebot", "aktion", "preis", "euro",
	"stück", "packung", "glas", "dose", "flasche",  # Units, not products
	"aus", "unserem", "unser", "diese", "dieser", "alle",  # Articles
	"siehe", "seite", "abbildung", "ohne", "mit",  # Instructions
	"klinge", "steckdose", "schalter",  # Generic parts
]

# Product category keywords (should contain at least one)
PRODUCT_KEYWORDS = [
	"kaffee", "espresso", "cappuccino", "bohnen", "kapseln", "lungo",
	"schokolade", "milch", "kakao", "riegel", "tafel",
	"saft", "tee", "getränk", "wasser", "cola", "limo",
	"müsli", "cerealien", "haferflocken", "cornflakes",
	"brot", "brötchen", "kuchen", "gebäck",
	"wurst", "käse", "butter", "joghurt", "quark",
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
	
	text_lower = text.lower()
	text_upper = text.upper()
	
	# Immediate rejection for non-product words
	for non_prod in NON_PRODUCT_WORDS:
		if text_lower.startswith(non_prod) or text_lower == non_prod:
			return 0.0
		# Reject if it's mostly this word (e.g., "ese Aktionsartikel")
		if non_prod in text_lower and len(non_prod) / max(len(text), 1) > 0.5:
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
	if length < 5:  # Too short (increased from 4)
		return 0.0
	if alpha_ratio < 0.50:  # Not enough letters (stricter)
		return 0.0
	if digit_ratio > 0.40:  # Too many digits (stricter)
		return 0.0
	if space_ratio > 0.4:  # Too many spaces (fragmented)
		return 0.0
	
	# Reject if contains too many special chars
	special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
	if special_chars / max(length, 1) > 0.3:
		return 0.0
	
	# Penalize single character words (OCR noise)
	words = text.split()
	if len(words) > 1:
		single_char_words = sum(1 for w in words if len(w) == 1)
		if single_char_words / len(words) > 0.4:  # Stricter
			return 0.0
	
	# Reject if no substantial words (all words < 3 chars)
	if not any(len(w) >= 3 for w in words):
		return 0.0
	
	# Base score from length and letter ratio
	score = length * (0.8 * alpha_ratio + 0.2)
	score -= 5.0 * digit_ratio * length  # Stronger penalty for digits
	
	# Bonus for reasonable length product names (10-50 characters)
	if 10 <= length <= 50:
		score += 8.0
	elif 6 <= length <= 60:
		score += 3.0
	
	# Bonus for having multiple words (more likely to be product names)
	if len(words) >= 2:
		score += 5.0
	
	# Strong bonus for recognized brand names
	brand_found = False
	for brand in KNOWN_BRANDS:
		if brand in text_upper:
			score += 20.0
			brand_found = True
			break
	
	# Bonus for product category keywords
	category_found = False
	for keyword in PRODUCT_KEYWORDS:
		if keyword in text_lower:
			score += 12.0
			category_found = True
			break
	
	# If no brand AND no category keyword, heavily penalize
	if not brand_found and not category_found:
		score -= 10.0
	
	# Bonus for starting with capital letter (brand names)
	if text[0].isupper():
		score += 3.0
	
	# Bonus for having all-caps words (brand names like BARISSIMO)
	if any(w.isupper() and len(w) > 2 for w in words):
		score += 5.0
	
	return score


def extract_product_text(img: np.ndarray, box: List[int]) -> str:
	"""
	Extract product text using OCR + NLP.
	
	Returns just the cleaned product name string for backward compatibility.
	Use extract_product_text_with_nlp() for full NLP analysis.
	"""
	result = extract_product_text_with_nlp(img, box)
	return result.get("product_name", "")


def extract_product_text_with_nlp(img: np.ndarray, box: List[int]) -> Dict[str, any]:
	"""
	Extract product text using OCR + NLP entity extraction.
	
	Args:
		img: Original BGR image
		box: Bounding box [x, y, w, h]
	
	Returns:
		Dictionary with:
		- product_name: Cleaned product name (string)
		- ocr_text: Raw OCR text
		- nlp_entities: Entities extracted by spaCy (if available)
		- brands: Detected brands
		- quantities: Extracted quantities
		- confidence: Extraction confidence score
	"""
	x, y, w, h = box
	H, W = img.shape[:2]
	price_like = re.compile(r"\d+[.,]\d{2}")

	def best_text_from_crop(crop_bgr: np.ndarray, region_name: str = "") -> Tuple[str, float, Dict]:
		"""Extract best text from a crop using multiple preprocessing strategies."""
		if crop_bgr.size == 0:
			return "", 0, {}
		
		# Analyze visual text quality
		text_quality = {}
		if TEXT_QUALITY_AVAILABLE:
			try:
				analyzer = TextQualityAnalyzer()
				text_quality = analyzer.analyze_text_region(crop_bgr)
			except Exception as e:
				print(f"      Text quality analysis failed: {e}")
		
		# Resize for better OCR
		scale = 3.0
		resized = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		
		# Minimal preprocessing - let Tesseract handle it
		# Just basic thresholding
		thr_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		
		# Also try inverted (white text on dark background)
		thr_inv = cv2.bitwise_not(thr_otsu)
		
		# Try direct grayscale (Tesseract can handle this)
		candidates = [gray, thr_otsu, thr_inv]
		best_combined, best_score = "", 0.0

		def extract_all_good_lines(bin_img: np.ndarray, cfg: str) -> Tuple[str, float]:
			"""Extract ALL lines from OCR output - let NLP decide what's the product."""
			all_lines = []
			try:
				# Use unified OCR (EasyOCR preferred, no timeout issues)
				text = ocr_image(bin_img, lang='de', config=cfg)
				for ln in text.splitlines():
					ln = clean_line(ln)
					if not ln or len(ln) < 3:
						continue
					# Skip obvious non-product patterns
					if price_like.search(ln):
						continue
					# Skip lines that are mostly digits
					if sum(c.isdigit() for c in ln) > len(ln) * 0.7:
						continue
					all_lines.append(ln)
			except (Exception, RuntimeError):
				# Catch timeout and other Tesseract errors
				pass
			
			if not all_lines:
				return "", 0.0
			
			# Return ALL text combined - NLP will extract the product name
			combined_text = " ".join(all_lines)
			# Score based on length and content quality
			score = len(combined_text) * 0.5 + sum(1 for c in combined_text if c.isalpha()) * 0.1
			return combined_text, score

		# Try different PSM modes - reduced to prevent timeout issues
		psm_configs = [
			r"--oem 3 --psm 7",  # Single text line (best for titles)
			r"--oem 3 --psm 6",  # Uniform block of text
		]
		
		for bin_img in candidates:
			for cfg in psm_configs:
				combined, combined_score = extract_all_good_lines(bin_img, cfg)
				if combined_score > best_score:
					best_combined, best_score = combined, combined_score
			
			# Early exit if we found really good text (brand + keyword)
			if best_score > 40:
				break

		# Apply text quality bonus if available
		if TEXT_QUALITY_AVAILABLE and text_quality:
			title_confidence = text_quality.get("confidence", 0)
			if text_quality.get("is_title", False):
				# Boost score for bold/large text
				best_score *= (1 + title_confidence)
			else:
				# Penalize thin/small text
				best_score *= (1 - title_confidence * 0.5)

		return best_combined, best_score, text_quality

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
	
	best_text_quality = {}
	all_candidates = []
	# Extract text from all regions and pick the best one
	best_line, best_score, best_region = "", 0.0, ""
	best_text_quality = {}
	all_candidates = []
	
	for region_name, reg in regions:
		line, score, quality = best_text_from_crop(reg, region_name)
		all_candidates.append((region_name, line, score))
		
		# QUALITY PENALTIES: Penalize obviously bad text
		# Starts with lowercase or special char (not a proper product title)
		if line and (line[0].islower() or not line[0].isalnum()):
			score -= 100.0
		
		# Too many special characters (garbled OCR)
		special_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / max(len(line), 1)
		if special_ratio > 0.2:
			score -= 50.0
		
		# BRAND BONUS: Strongly prefer regions with known brand names at START
		brand_at_start = False
		for brand in KNOWN_BRANDS:
			if line.upper().startswith(brand) or line.upper().startswith("O " + brand):
				brand_at_start = True
				score += 300.0  # Huge bonus for brand at start
				break
		
		# Smaller bonus if brand appears anywhere (but not at start)
		if not brand_at_start:
			has_brand = any(brand in line.upper() for brand in KNOWN_BRANDS)
			if has_brand:
				score += 100.0
		
		if score > best_score:
			best_line, best_score, best_region = line, score, region_name
			best_text_quality = quality
	
	# Debug output for troubleshooting - enable to see which regions are selected
	import os
	if os.getenv('DEBUG_OCR'):
		print(f"    Candidates: {[(r, l[:40] if l else '', f'{s:.1f}') for r, l, s in all_candidates]}")
		print(f"    Selected: {best_region} with score {best_score:.1f}")
	
	# If we found good text, apply NLP extraction
	if best_score > 0:
		ocr_text = postprocess_product_text(best_line)
		
		# Reject empty results after postprocessing
		if not ocr_text or len(ocr_text) < 5:
			return {
				"product_name": "",
				"ocr_text": "",
				"nlp_entities": [],
				"brands": [],
				"quantities": [],
				"noun_chunks": [],
				"pos_filtered": "",
				"confidence": 0.0,
				"method": "rejected_after_cleaning"
			}
		
		# PHASE 2 IMPROVEMENT: Limit text to first 2-3 lines for better product name extraction
		# Product titles are typically in the first few lines - sending all text to NLP
		# causes it to pick description details instead of the main product name
		limited_text = ocr_text
		
		# If visual quality indicates title text (bold/large), use just first sentence
		if best_text_quality.get("is_title", False):
			parts = ocr_text.split('.')
			if parts:
				limited_text = parts[0].strip()
		else:
			# Otherwise, keep first 2-3 sentences
			parts = ocr_text.split('.')
			if len(parts) > 3:
				# Keep first 3 parts
				limited_text = '. '.join(parts[:3]).strip()
		
		# Safety: ensure we have some text
		if not limited_text:
			limited_text = ocr_text
		
		# Apply NLP entity extraction if available
		if NLP_AVAILABLE and limited_text:
			try:
				nlp_result = extract_product_entities(limited_text)
				
				# Additional validation: reject if NLP confidence is too low
				nlp_confidence = nlp_result.get("confidence", 0.0)
				nlp_product = nlp_result.get("product_name") or ""
				
				# If NLP rejected the text (returned empty), respect that decision
				# Fallback policy (conservative): if NLP returns empty but OCR text looks
				# like a plausible product title, keep it to avoid losing recall.
				if not nlp_product or len(nlp_product) < 5:
					fallback_score = score_line(ocr_text)
					visual_is_title = bool(best_text_quality.get("is_title", False))
					visual_conf = float(best_text_quality.get("confidence", 0.0) or 0.0)

					# Heuristics: accept OCR fallback only when text is strong or visually title-like.
					allow_fallback = (fallback_score >= 25.0) or (visual_is_title and visual_conf >= 0.35)
					if allow_fallback:
						return {
							"product_name": ocr_text,
							"ocr_text": ocr_text,
							"nlp_entities": nlp_result.get("entities", []),
							"brands": nlp_result.get("brands", []),
							"quantities": nlp_result.get("quantities", []),
							"noun_chunks": nlp_result.get("noun_chunks", []),
							"pos_filtered": nlp_result.get("pos_filtered", ""),
							"confidence": 0.35,
							"method": "ocr_fallback_after_nlp_reject",
							"visual_quality": best_text_quality,
							"nlp_rejected": True,
							"fallback_score": fallback_score,
						}

					return {
						"product_name": "",
						"ocr_text": ocr_text,
						"nlp_entities": [],
						"brands": [],
						"quantities": [],
						"noun_chunks": [],
						"pos_filtered": "",
						"confidence": 0.0,
						"method": "rejected_by_nlp",
						"visual_quality": best_text_quality
					}
				
				# Use NLP-identified product name
				final_product = nlp_product
				
				# Apply text quality bonus to confidence
				final_confidence = max(0.3, nlp_confidence)
				if best_text_quality.get("is_title", False):
					visual_confidence = best_text_quality.get("confidence", 0)
					final_confidence = min(1.0, final_confidence * (1 + visual_confidence))
				
				# Return comprehensive result
				return {
					"product_name": final_product,
					"ocr_text": ocr_text,
					"nlp_entities": nlp_result.get("entities", []),
					"brands": nlp_result.get("brands", []),
					"quantities": nlp_result.get("quantities", []),
					"noun_chunks": nlp_result.get("noun_chunks", []),
					"pos_filtered": nlp_result.get("pos_filtered", ""),
					"confidence": final_confidence,
					"method": "nlp",
					"visual_quality": best_text_quality
				}
			except Exception as e:
				print(f"    NLP extraction failed: {e}, falling back to OCR-only")
		
		# Fallback: return OCR-only result
		return {
			"product_name": ocr_text,
			"ocr_text": ocr_text,
			"nlp_entities": [],
			"brands": [],
			"quantities": [],
			"noun_chunks": [],
			"pos_filtered": "",
			"confidence": 0.5,
			"method": "ocr_only"
		}
	
	# No text found
	return {
		"product_name": "",
		"ocr_text": "",
		"nlp_entities": [],
		"brands": [],
		"quantities": [],
		"noun_chunks": [],
		"pos_filtered": "",
		"confidence": 0.0,
		"method": "none"
	}


def postprocess_product_text(text: str) -> str:
	"""Clean up and format the extracted product text."""
	if not text:
		return ""
	
	# Final validation - reject clearly bad products
	text_lower = text.lower()
	
	# Remove leading "Aus unserem Sortiment" header if present
	if text_lower.startswith("aus unserem"):
		# Remove the header line but keep the rest
		lines = text.split('\n')
		if len(lines) > 1:
			text = '\n'.join(lines[1:])  # Keep everything after first line
			text_lower = text.lower()
		else:
			# Single line starting with "aus unserem" - try to keep brand after it
			# e.g., "Aus unserem Sortiment BARISSIMO Kaffee" -> "BARISSIMO Kaffee"
			text = re.sub(r"^aus\s+unsere[mn]?\s+\w*\s*", "", text, flags=re.IGNORECASE).strip()
			text_lower = text.lower()
			if not text:
				return ""
	
	# Reject if it starts with other non-product words
	bad_starts = ["aktionsartikel", "sortiment", "vormittag", "nachmittag", 
	              "unterschied", "garantie", "dekoration",
	              "siehe", "abbildung", "für", "bei",
	              "ese ", "rantie", "nen ", "llmenge"]
	if any(text_lower.startswith(bad) for bad in bad_starts):
		return ""
	
	# Reject if it's just generic parts
	generic_parts = ["aluminium-klinge", "klinge", "steckdose", "steckdosenleiste",
	                 "schalter", "kabel", "batterie", "akku"]
	if text_lower in generic_parts or text_lower.strip() in generic_parts:
		return ""
	
	# Reject if too short after cleaning
	if len(text.strip()) < 5:
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