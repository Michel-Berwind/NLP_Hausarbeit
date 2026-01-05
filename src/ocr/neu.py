from pathlib import Path
import argparse
import cv2
import numpy as np
import pytesseract
import re
import json
import sys

# Set to your tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def iou(a, b):
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


def nms(cands, thr=0.35):
    cands = sorted(cands, key=lambda t: t[4], reverse=True)
    kept = []
    for c in cands:
        box = (c[0], c[1], c[2], c[3])
        if all(iou(box, (k[0], k[1], k[2], k[3])) < thr for k in kept):
            kept.append(c)
    return kept


def clamp(x, y, w, h, W, H):
    x = int(max(0, x)); y = int(max(0, y))
    w = int(min(W - x, w)); h = int(min(H - y, h))
    return x, y, w, h


def ocr_price_from_roi(roi_bgr, debug=False):
    """
    OCR a price from a region of interest.
    Handles European stacked price format: large digit + superscript cents + dot below.
    
    Strategy:
    1. Try to detect stacked layout and OCR main digit + cents separately
    2. Fall back to full-ROI OCR with multiple preprocessing
    3. NO automatic correction - return raw OCR result
    """
    h, w = roi_bgr.shape[:2]
    
    def try_thresh(crop, scale=3.0):
        """Generate multiple thresholded versions of the crop."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Otsu
        th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Adaptive
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        # Inverted versions
        th1_inv = 255 - th1
        th2_inv = 255 - th2
        
        candidates = []
        for thr in (th1, th2, th1_inv, th2_inv):
            # Morphological cleanup
            thr_proc = cv2.morphologyEx(
                thr,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1
            )
            # Scale up for better OCR
            thr_proc = cv2.resize(thr_proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # Add white border (helps Tesseract)
            thr_proc = cv2.copyMakeBorder(thr_proc, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            candidates.append(thr_proc)
        return candidates
    
    def ocr_single_region(crop, configs, scale=3.0):
        """OCR a single region with multiple configs, return best result."""
        thr_list = try_thresh(crop, scale)
        results = []
        for thr in thr_list:
            for cfg in configs:
                txt = pytesseract.image_to_string(thr, lang="eng", config=cfg).strip()
                txt = txt.replace('\n', ' ').replace('\ufeff', '').replace(',', '.').replace(' ', '')
                digits = re.sub(r"[^\d.]", "", txt)
                if digits:
                    results.append((digits, thr))
        return results
    
    # OCR configs - prioritize single line detection
    cfgs = [
        r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
        r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,",  # Single word
        r"--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.,", # Raw line
        r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,",
        r"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,",
    ]
    
    # Config for single digits
    single_digit_cfgs = [
        r"--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",  # Single character
        r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
    ]
    
    def postprocess_price(txt):
        """Convert OCR output to price format."""
        txt = txt.replace(",", ".").replace(" ", "").replace("\n", "")
        
        # Direct match X.XX or XX.XX or XXX.XX
        m = re.search(r"(\d{1,3})\.(\d{2})", txt)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        
        # Parse consecutive digits
        digits = re.sub(r"[^\d]", "", txt)
        if len(digits) == 3:
            return f"{digits[0]}.{digits[1:]}"
        if len(digits) == 4:
            return f"{digits[:2]}.{digits[2:]}"
        if len(digits) == 5:
            return f"{digits[:3]}.{digits[3:]}"
        
        return None
    
    def validate_price(price_str):
        """Check if price is reasonable."""
        if not price_str:
            return False
        try:
            val = float(price_str)
            return 0.5 <= val <= 100
        except:
            return False
    
    # === Strategy 1: Split ROI for stacked layout ===
    # European stacked prices: large digit on left, small cents upper-right
    if w > 20 and h > 15:
        # Estimate split: main digit takes ~60% width, cents in upper-right
        split_x = int(w * 0.55)
        
        # Main digit region (left side, full height)
        main_roi = roi_bgr[:, :split_x]
        # Cents region (upper-right, top 60% of height)
        cents_roi = roi_bgr[:int(h * 0.6), split_x:]
        
        if main_roi.size > 0 and cents_roi.size > 0:
            # OCR main digit
            main_results = ocr_single_region(main_roi, single_digit_cfgs, scale=4.0)
            cents_results = ocr_single_region(cents_roi, cfgs, scale=4.0)
            
            for main_txt, _ in main_results:
                main_digits = re.sub(r"[^\d]", "", main_txt)
                if main_digits and len(main_digits) <= 2:
                    for cents_txt, thr in cents_results:
                        cents_digits = re.sub(r"[^\d]", "", cents_txt)
                        if len(cents_digits) >= 2:
                            cents_digits = cents_digits[:2]  # Take first 2
                            price = f"{main_digits}.{cents_digits}"
                            if validate_price(price):
                                if debug:
                                    print(f"[DEBUG] Split OCR: main='{main_digits}' cents='{cents_digits}' = {price}")
                                return price, thr
    
    # === Strategy 2: Full ROI OCR ===
    all_results = []
    thr_list = try_thresh(roi_bgr, scale=3.0)
    
    for thr in thr_list:
        for cfg in cfgs:
            txt = pytesseract.image_to_string(thr, lang="eng", config=cfg).strip()
            if debug:
                print(f"[DEBUG] OCR cfg: {cfg} | Output: '{txt}'")
            if not txt:
                continue
            
            price = postprocess_price(txt)
            if validate_price(price):
                all_results.append((price, thr))
    
    # Return first valid result
    for price, thr in all_results:
        return price, thr
    
    return None, None


def run(image_path: Path, debug_dir: Path, verbose: bool = True, correct_nines: bool = False):
    """
    Detect and OCR prices from a supermarket poster image.
    
    Args:
        image_path: Path to the input image
        debug_dir: Directory to write debug images
        verbose: Print progress info
        correct_nines: Apply heuristic to correct 2→9 and 3→9 misreads in cents
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    debug_dir.mkdir(parents=True, exist_ok=True)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Find white regions (high lightness, low saturation)
    # White: high V, low S, or high L in LAB
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    mask_white2 = (L > 180).astype(np.uint8) * 255
    mask_white = cv2.bitwise_and(mask_white, mask_white2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=2)
    cv2.imwrite(str(debug_dir / "white_mask.png"), mask_white)

    # 1b. Also find orange/yellow regions (for colored price boxes)
    # Orange: H in [5, 25], high S, high V
    mask_orange = cv2.inRange(hsv, (5, 100, 150), (25, 255, 255))
    # Yellow: H in [20, 35], high S, high V
    mask_yellow = cv2.inRange(hsv, (20, 80, 150), (40, 255, 255))
    mask_colored = cv2.bitwise_or(mask_orange, mask_yellow)
    mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=2)
    cv2.imwrite(str(debug_dir / "orange_mask.png"), mask_colored)

    # Combine white and colored masks for candidate regions
    mask_price_bg = cv2.bitwise_or(mask_white, mask_colored)
    cv2.imwrite(str(debug_dir / "price_bg_mask.png"), mask_price_bg)

    # 2. Find connected components (candidate price boxes)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_price_bg, connectivity=8)

    # 3. For each white box, check for blue OR dark text inside
    # Blue: moderate-high S, H in [90, 140] (OpenCV: 0-180)
    blue_mask = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
    # Also detect dark text (low V) for orange/other colored price boxes
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
    # Combine blue and dark masks
    text_mask = cv2.bitwise_or(blue_mask, dark_mask)
    cv2.imwrite(str(debug_dir / "blue_text_mask.png"), blue_mask)
    cv2.imwrite(str(debug_dir / "text_mask.png"), text_mask)

    cands = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        # Size constraints for price boxes - relaxed to catch smaller boxes
        if area < 200:  # minimum area (lowered)
            continue
        if area > 100000:  # maximum area
            continue
        if w < 15 or h < 15:  # minimum dimensions (lowered)
            continue
        if w > 500 or h > 400:  # maximum dimensions
            continue
        aspect = w / max(h, 1)
        # Price boxes can have various aspect ratios
        if aspect < 0.3 or aspect > 6.0:
            continue
        if w > 0.6 * W or h > 0.5 * H:
            continue

        # Require some text pixels (blue or dark)
        roi_text = text_mask[y:y+h, x:x+w]
        text_ratio = float((roi_text > 0).mean())
        if text_ratio < 0.01:  # at least 1% text pixels (lowered)
            continue
        if text_ratio > 0.85:  # not too much
            continue

        # Require reasonably bright background (mean LAB L) - but allow orange boxes
        roi_lab = lab[y:y+h, x:x+w, :]
        mean_L = roi_lab[:, :, 0].mean()
        if mean_L < 120:  # reasonably bright (lowered from 140)
            continue

        # Require text to be darker than background (contrast)
        text_mask_roi = (roi_text > 0)
        if text_mask_roi.sum() > 0:
            text_L = roi_lab[:, :, 0][text_mask_roi].mean()
            contrast = mean_L - text_L
            if contrast < 10:  # need some contrast (lowered from 15)
                continue
        else:
            continue

        cands.append((x, y, w, h, text_ratio, area))

    # 4. NMS to remove overlapping boxes
    def iou_box(a, b):
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union
    cands = sorted(cands, key=lambda t: t[4], reverse=True)
    kept = []
    for c in cands:
        if all(iou_box(c, k) < 0.3 for k in kept):
            kept.append(c)

    # 5. Debug draw
    dbg = img.copy()
    for idx, (x, y, w, h, blue_ratio, area) in enumerate(kept):
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(dbg, str(idx), (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "price_cands_boxes.png"), dbg)
    if verbose:
        print("Price candidates:", len(kept))

    # 6. OCR
    detections = []
    all_ocr_attempts = []  # For debugging
    for i, (x, y, w, h, blue_ratio, area) in enumerate(kept):
        roi = img[y:y + h, x:x + w]
        price, bin_img = ocr_price_from_roi(roi, debug=verbose)
        cv2.imwrite(str(debug_dir / f"cand_{i:02d}_roi.png"), roi)
        # Always save a binarized image, even if OCR fails
        if bin_img is not None:
            cv2.imwrite(str(debug_dir / f"cand_{i:02d}_bin.png"), bin_img)
        else:
            # Save a dummy binarized image for debugging
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cv2.imwrite(str(debug_dir / f"cand_{i:02d}_bin.png"), thr)
        
        all_ocr_attempts.append({
            "idx": i,
            "box": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "ocr_result": price
        })
        
        # Only accept if OCR result is a valid price format (1-2 digits before decimal)
        if price and re.fullmatch(r"\d{1,2}\.\d{2}", price):
            # Additional validation: filter out unlikely prices
            try:
                val = float(price)
                # Skip very small prices (likely OCR noise) and unrealistic prices
                if val < 0.50 or val > 50.0:
                    if verbose:
                        print(f"[SKIP] Price {price} outside typical range")
                    continue
            except:
                continue
                
            cx = x + w // 2
            cy = y + h // 2
            detections.append({
                "price": price,
                "x": int(cx),
                "y": int(cy),
                "box": [int(x), int(y), int(w), int(h)],
                "text_ratio": float(blue_ratio),
                "roi_file": str(debug_dir / f"cand_{i:02d}_bin.png"),
            })
    
    # Save all OCR attempts for debugging
    debug_json = debug_dir / "all_ocr_attempts.json"
    debug_json.write_text(json.dumps(all_ocr_attempts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional: Apply heuristic correction for common 9→2/3 misreads
    if correct_nines:
        for d in detections:
            price = d["price"]
            # Common misread: last digit 2 or 3 should be 9
            # e.g., 3.29 → 3.99, 7.32 → 7.39
            if len(price) >= 4 and price[-1] in ('2', '3'):
                corrected = price[:-1] + '9'
                if verbose:
                    print(f"[CORRECT] {price} → {corrected}")
                d["price"] = corrected
                d["original_ocr"] = price

    if verbose:
        print(f"Detections: {len(detections)}")
    prices_unique = sorted({d["price"] for d in detections}, key=lambda s: float(s))
    if verbose:
        print(f"FOUND PRICES: {prices_unique}")

    out_json = debug_dir / "price_detections.json"
    out_json.write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print("Saved:", out_json)


def main(argv=None):
    p = argparse.ArgumentParser(description="Detect price labels in a poster image")
    p.add_argument("image", nargs="?", default="data/images/aldi/pw.png",
                   help="Path to input image")
    p.add_argument("--debug-dir", default="data/debug_price_boxes",
                   help="Directory to write debug images and detections")
    p.add_argument("--quiet", action="store_true", help="Less verbose output")
    p.add_argument("--correct-nines", action="store_true",
                   help="Apply heuristic correction for 9→2/3 misreads in cents")
    args = p.parse_args(argv)

    image_path = Path(args.image)
    debug_dir = Path(args.debug_dir)
    try:
        run(image_path, debug_dir, verbose=not args.quiet, correct_nines=args.correct_nines)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
