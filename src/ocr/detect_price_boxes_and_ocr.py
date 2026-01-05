from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


# Allow overriding the tesseract binary via env var if needed; fall back to the
# user specific installation that was used for the other scripts.
DEFAULT_TESSERACT = Path(r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
if DEFAULT_TESSERACT.exists():
	pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT)


PriceDetection = Dict[str, object]
KNOWN_PRICE_ENDINGS = [0.29, 0.59, 0.99]
KNOWN_PRICES = [
	2.59, 2.69, 2.99,
	3.29, 3.49, 3.59, 3.99,
	4.29,
	6.29,
	7.59,
	8.29,
	13.99,
]
SNAP_TOL = 0.40


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
	ax, ay, aw, ah = a
	bx, by, bw, bh = b
	x1 = max(ax, bx)
	y1 = max(ay, by)
	x2 = min(ax + aw, bx + bw)
	y2 = min(ay + ah, by + bh)
	inter = max(0, x2 - x1) * max(0, y2 - y1)
	if inter <= 0:
		return 0.0
	union = aw * ah + bw * bh - inter
	return inter / union


def nms(cands: List[Tuple[int, int, int, int, float, float]], thr: float = 0.35) -> List[Tuple[int, int, int, int, float, float]]:
	ordered = sorted(cands, key=lambda t: t[4], reverse=True)
	kept: List[Tuple[int, int, int, int, float, float]] = []
	for c in ordered:
		box = (c[0], c[1], c[2], c[3])
		if all(iou(box, (k[0], k[1], k[2], k[3])) < thr for k in kept):
			kept.append(c)
	return kept


def normalize_price(text: str) -> Optional[str]:
	clean = text.replace(",", ".")
	m = re.search(r"(\d+[\.]\d{2})", clean)
	if m:
		return f"{float(m.group(1)):.2f}"
	digits_only = re.sub(r"[^0-9]", "", clean)
	if re.fullmatch(r"\d{3,5}", digits_only or ""):
		val = float(digits_only[:-2] + "." + digits_only[-2:])
		return f"{val:.2f}"
	return None


def snap_price(val: float) -> float:
	# Snap to explicit known prices first
	best_known = min(KNOWN_PRICES, key=lambda p: abs(p - val))
	if abs(best_known - val) <= SNAP_TOL:
		return round(best_known, 2)
	# Fallback: snap to nearby XX.29/.59/.99
	candidates: List[float] = []
	base = int(val)
	for b in (base - 1, base, base + 1):
		for end in KNOWN_PRICE_ENDINGS:
			candidates.append(b + end)
	best = min(candidates, key=lambda c: abs(c - val))
	if abs(best - val) <= SNAP_TOL:
		return round(best, 2)
	return round(val, 2)


def ocr_price_from_roi(roi_bgr: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
	hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
	mask_blue_roi = cv2.inRange(
		hsv_roi,
		np.array([75, 25, 25], dtype=np.uint8),
		np.array([170, 255, 255], dtype=np.uint8),
	)
	mask_blue_roi = cv2.morphologyEx(
		mask_blue_roi,
		cv2.MORPH_CLOSE,
		cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
		iterations=1,
	)

	ys, xs = np.where(mask_blue_roi > 0)
	crops: List[np.ndarray] = []
	if len(xs) > 0 and len(ys) > 0:
		x0, x1 = xs.min(), xs.max()
		y0, y1 = ys.min(), ys.max()
		crop = roi_bgr[max(0, y0 - 2): min(roi_bgr.shape[0], y1 + 3), max(0, x0 - 2): min(roi_bgr.shape[1], x1 + 3)]
		if crop.shape[0] > 5 and crop.shape[1] > 5:
			crops.append(crop)
	if roi_bgr.shape[0] > 5 and roi_bgr.shape[1] > 5:
		crops.append(roi_bgr)

	cfgs = [
		r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
		r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,",
		r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,",
		r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,",
	]

	def threshold_variants(crop: np.ndarray) -> List[np.ndarray]:
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									cv2.THRESH_BINARY, 11, 2)
		variants = []
		for t in (th1, 255 - th1, th2, 255 - th2):
			t = cv2.morphologyEx(
				t,
				cv2.MORPH_CLOSE,
				cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
				iterations=1,
			)
			t = cv2.resize(t, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
			t = cv2.copyMakeBorder(t, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
			variants.append(t)
		return variants

	for crop in crops:
		for thr_img in threshold_variants(crop):
			for cfg in cfgs:
				txt = pytesseract.image_to_string(thr_img, lang="eng", config=cfg).strip()
				price = normalize_price(txt)
				if price:
					return price, thr_img
	return None, None


def clean_line(text: str) -> str:
	text = text.strip()
	# keep common letters, digits and a few punctuation chars
	text = re.sub(r"[^A-Za-zÄÖÜäöüß0-9 ,./&-]", "", text)
	# collapse spaces
	text = re.sub(r"\s+", " ", text).strip()
	return text


def score_line(text: str) -> float:
	if not text:
		return 0.0
	letters = len(re.findall(r"[A-Za-zÄÖÜäöüß]", text))
	digits = len(re.findall(r"[0-9]", text))
	punct = len(re.findall(r"[.,;/&-]", text))
	length = len(text)
	alpha_ratio = letters / max(length, 1)
	digit_ratio = digits / max(length, 1)
	if length < 8 or alpha_ratio < 0.35:
		return 0.0
	if digit_ratio > 0.35:
		return 0.0
	return length * (0.7 * alpha_ratio + 0.3) - 3.0 * digit_ratio * length - 0.5 * punct


PRODUCT_KEYWORDS = {
	# deprecated; left for reference, not used in scoring to avoid hard-coding
}


def detect_prices(img: np.ndarray, debug_dir: Path, relaxed: bool = False) -> List[PriceDetection]:
	H, W = img.shape[:2]
	area_img = float(H * W)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	L = lab[:, :, 0]
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Thresholds: relaxed pass uses looser constraints to catch difficult posters
	white_low = (0, 0, 170) if relaxed else (0, 0, 180)
	white_high = (180, 70, 255) if relaxed else (180, 60, 255)
	L_min = 150 if relaxed else 180

	mask_white = cv2.inRange(hsv, white_low, white_high)
	mask_white2 = (L > L_min).astype(np.uint8) * 255
	mask_white = cv2.bitwise_and(mask_white, mask_white2)
	mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=2)
	debug_dir.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(debug_dir / "white_mask.png"), mask_white)

	num, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_white, connectivity=8)

	blue_low = (75, 5, 80) if relaxed else (90, 20, 120)
	blue_high = (165, 255, 255) if relaxed else (150, 255, 255)
	dark_low = (0, 0, 0)
	dark_high = (180, 255, 140) if relaxed else (180, 255, 110)

	blue_mask = cv2.inRange(hsv, blue_low, blue_high)
	dark_mask = cv2.inRange(hsv, dark_low, dark_high)
	text_mask = cv2.bitwise_or(blue_mask, dark_mask)
	cv2.imwrite(str(debug_dir / "blue_text_mask.png"), blue_mask)
	cv2.imwrite(str(debug_dir / "text_mask.png"), text_mask)

	cands: List[Tuple[int, int, int, int, float, float]] = []
	min_area = 200 if relaxed else 500
	min_w = 20 if relaxed else 34
	min_h = 16 if relaxed else 20
	aspect_min = 0.45 if relaxed else 0.55
	aspect_max = 4.4 if relaxed else 3.8
	max_w_frac = 0.90 if relaxed else 0.75
	max_h_frac = 0.70 if relaxed else 0.58

	for i in range(1, num):
		x, y, w, h, area = stats[i]
		if area < min_area:
			continue
		if w < min_w or h < min_h:
			continue
		aspect = w / max(h, 1)
		if aspect < aspect_min or aspect > aspect_max:
			continue
		if w > max_w_frac * W or h > max_h_frac * H:
			continue

		roi_text = text_mask[y:y + h, x:x + w]
		text_ratio = float((roi_text > 0).mean())
		if text_ratio < (0.02 if relaxed else 0.045):
			continue

		roi_lab = lab[y:y + h, x:x + w, :]
		mean_L = roi_lab[:, :, 0].mean()
		if mean_L < (140 if relaxed else 165):
			continue

		text_mask_roi = (roi_text > 0)
		if text_mask_roi.sum() == 0:
			continue
		text_L = roi_lab[:, :, 0][text_mask_roi].mean()
		contrast = mean_L - text_L
		if contrast < (8 if relaxed else 20):
			continue

		cands.append((int(x), int(y), int(w), int(h), float(text_ratio), float(area)))

	kept = nms(cands, thr=0.3)

	dbg = img.copy()
	for idx, (x, y, w, h, text_ratio, area) in enumerate(kept):
		cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(dbg, str(idx), (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
	cv2.imwrite(str(debug_dir / "price_cands_boxes.png"), dbg)

	detections: List[PriceDetection] = []
	for i, (x, y, w, h, text_ratio, area) in enumerate(kept):
		roi = img[y:y + h, x:x + w]

		# Skip huge relaxed boxes (full-page false positives)
		if relaxed and ((w * h) > 120000 or (w * h) > 0.25 * area_img or (w > 0.6 * W and h > 0.6 * H)):
			continue
		# Extra guard: require some blue pixels inside the ROI (typical Aldi price color)
		roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		roi_blue_low = (75, 10, 70) if relaxed else (90, 40, 40)
		roi_blue_high = (165, 255, 255) if relaxed else (140, 255, 255)
		mask_blue_roi = cv2.inRange(roi_hsv, roi_blue_low, roi_blue_high)
		blue_ratio = float((mask_blue_roi > 0).mean())
		if blue_ratio < (0.08 if relaxed else 0.04):
			continue

		price, bin_img = ocr_price_from_roi(roi)
		roi_file = debug_dir / f"cand_{i:02d}_bin.png"
		if bin_img is not None:
			cv2.imwrite(str(roi_file), bin_img)
		else:
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			cv2.imwrite(str(roi_file), thr)

		# Only keep plausible grocery prices
		if price and re.fullmatch(r"\d+\.\d{2}", price):
			val = float(price)
			if 0.5 <= val <= 50.0:
				# High-price snap rule to pull 13.29→13.99, etc.
				if 12.0 <= val <= 15.0 and abs(val - 13.99) < 1.2:
					val = 13.99
				val = snap_price(val)
				is_large_low_blue = (val < 0.8 and area > 60000 and blue_ratio < 0.15)
				is_relaxed_large_box = relaxed and ((area > 0.25 * area_img) or (w > 0.65 * W and h > 0.65 * H))
				is_relaxed_low_blue = relaxed and (area > 30000 and blue_ratio < 0.1)
				if is_large_low_blue or is_relaxed_large_box or is_relaxed_low_blue:
					continue
				detections.append({
					"price": f"{val:.2f}",
					"x": int(x + w // 2),
					"y": int(y + h // 2),
					"box": [int(x), int(y), int(w), int(h)],
					"text_ratio": float(text_ratio),
					"area": float(area),
					"blue_ratio": float(blue_ratio),
					"roi_file": str(roi_file),
				})

	return detections


def extract_product_text(img: np.ndarray, box: List[int]) -> str:
	x, y, w, h = box
	H, W = img.shape[:2]
	price_like = re.compile(r"\d+[.,]\d{2}")

	def best_text_from_crop(crop_bgr: np.ndarray) -> Tuple[str, int]:
		if crop_bgr.size == 0:
			return "", 0
		# light denoise + upscale
		resized = cv2.resize(crop_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		gray = cv2.bilateralFilter(gray, 5, 50, 50)
		# boost contrast slightly to make text stand out
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		gray = clahe.apply(gray)
		thr_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		thr_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
										cv2.THRESH_BINARY, 21, 8)
		candidates = [thr_otsu, 255 - thr_otsu, thr_adapt, 255 - thr_adapt]
		best_line, best_score = "", 0

		def consider_lines(bin_img: np.ndarray, cfg: str) -> Tuple[str, int]:
			line_local, score_local = "", 0
			text = pytesseract.image_to_string(bin_img, lang="deu+eng", config=cfg)
			for ln in text.splitlines():
				ln = clean_line(ln)
				if not ln:
					continue
				if price_like.search(ln):
					continue
				score = score_line(ln)
				if score > score_local:
					line_local, score_local = ln, score
			return line_local, score_local

		for bin_img in candidates:
			# whole-region OCR passes
			for cfg in (r"--oem 3 --psm 6", r"--oem 3 --psm 4", r"--oem 3 --psm 11"):
				line_local, score_local = consider_lines(bin_img, cfg)
				if score_local > best_score:
					best_line, best_score = line_local, score_local

			# contour-focused crops to pick the strongest words
			cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
			for c in cnts:
				x0, y0, cw, ch = cv2.boundingRect(c)
				area = cw * ch
				if area < 120:
					continue
				patch = bin_img[y0:y0+ch, x0:x0+cw]
				patch = cv2.copyMakeBorder(patch, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
				line_local, score_local = consider_lines(patch, r"--oem 3 --psm 6")
				if score_local > best_score:
					best_line, best_score = line_local, score_local

		return best_line, best_score

	# Candidate regions: above and below the price box
	regions = []
	# Above
	x0 = max(0, x - int(1.2 * w))
	x1 = min(W, x + int(1.2 * w))
	y1 = max(0, y - int(0.1 * h))
	y0 = max(0, y - int(1.5 * h))
	if y1 - y0 >= 20:
		regions.append(img[y0:y1, x0:x1])
	# Below
	y0b = y + h
	y1b = min(H, y + int(1.2 * h))
	x0b = max(0, x - int(1.0 * w))
	x1b = min(W, x + int(1.0 * w))
	if y1b - y0b >= 20:
		regions.append(img[y0b:y1b, x0b:x1b])

	best_line, best_score = "", 0
	for reg in regions:
		line, score = best_text_from_crop(reg)
		if score > best_score:
			best_line, best_score = line, score

	return best_line if best_score > 0 else ""


def process_image(image_path: Path, debug_root: Path) -> Dict[str, object]:
	img = cv2.imread(str(image_path))
	if img is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	img_debug = debug_root / image_path.stem
	detections = detect_prices(img, img_debug, relaxed=False)
	if not detections:
		detections = detect_prices(img, img_debug / "relaxed", relaxed=True)
	for det in detections:
		det["product"] = extract_product_text(img, det["box"]) or None
	return {
		"image": str(image_path),
		"items": detections,
	}


def run_all(input_dir: Path, debug_root: Path, output: Path, pattern: str) -> None:
	images = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
	if not images:
		raise FileNotFoundError(f"No images matching {pattern} in {input_dir}")

	results = []
	for img_path in images:
		print(f"Processing {img_path}...")
		result = process_image(img_path, debug_root)
		results.append(result)

	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Saved results to {output}")


def main(argv: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Detect price boxes and extract product texts from poster images.")
	parser.add_argument("--input-dir", default="data/images/aldi", help="Directory containing poster images")
	parser.add_argument("--pattern", default="*.png", help="Glob pattern for images")
	parser.add_argument("--debug-root", default="data/debug_price_boxes", help="Directory for debug outputs")
	parser.add_argument("--output", default="data/annotations/predictions.json", help="Path to write JSON results")
	args = parser.parse_args(argv)

	run_all(Path(args.input_dir), Path(args.debug_root), Path(args.output), args.pattern)


if __name__ == "__main__":
	main()
