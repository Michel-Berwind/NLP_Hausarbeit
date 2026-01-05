from pathlib import Path
import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

IMG_PATH = Path("data/images/aldi/page_coffee.png")
DEBUG_DIR = Path("data/debug_price_boxes")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(IMG_PATH))
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# stärkerer Kontrast für Prospekt-Print
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# wir probieren BEIDE Richtungen für Konturen
variants = {
    "inv_closed": None,
    "norm_closed": None
}

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
closed_norm = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
closed_inv  = 255 - closed_norm

variants["norm_closed"] = closed_norm
variants["inv_closed"]  = closed_inv

all_prices = set()
best_cnt = 0
best_name = None
best_boxes = []

config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,"

for name, binimg in variants.items():
    cnts, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch

        # deutlich lockerer als vorher:
        if area < 1200:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.3 or aspect > 12:
            continue

        boxes.append((x, y, cw, ch, area))

    if len(boxes) > best_cnt:
        best_cnt = len(boxes)
        best_name = name
        best_boxes = boxes

# Debug-Bild mit Boxen
dbg = img.copy()
best_boxes = sorted(best_boxes, key=lambda t: (t[1], t[0]))[:60]

for i, (x, y, cw, ch, area) in enumerate(best_boxes):
    cv2.rectangle(dbg, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
    cv2.putText(dbg, str(i), (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

out_dbg = DEBUG_DIR / f"boxes_{best_name}_{len(best_boxes)}.png"
cv2.imwrite(str(out_dbg), dbg)

# OCR auf den ROIs
for i, (x, y, cw, ch, area) in enumerate(best_boxes):
    pad = 6
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + cw + pad, w); y1 = min(y + ch + pad, h)

    roi = gray[y0:y1, x0:x1]
    roi_thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    txt = pytesseract.image_to_string(roi_thr, lang="deu", config=config)
    prices = re.findall(r"\b\d{1,2}[.,]\d{2}\b", txt)
    for p in prices:
        all_prices.add(p.replace(",", "."))

print("Best variant:", best_name, "boxes:", len(best_boxes))
print("FOUND PRICES:", sorted(all_prices, key=lambda x: float(x)))
print("Debug image saved to:", out_dbg)
