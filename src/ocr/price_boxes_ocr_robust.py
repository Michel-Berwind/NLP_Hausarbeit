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
gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Box detection (wie vorher) ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray0)
thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in cnts:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cw * ch
    if area < 1200:
        continue
    aspect = cw / max(ch, 1)
    if aspect < 0.3 or aspect > 12:
        continue
    boxes.append((x, y, cw, ch, area))

boxes = sorted(boxes, key=lambda t: (t[1], t[0]))[:60]

# --- OCR settings ---
price_re = re.compile(r"\b\d{1,2}[.,]\d{2}\b")
configs = [
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,",
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
    r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,",
]

def ocr_try(img_variant):
    hits = set()
    for cfg in configs:
        txt = pytesseract.image_to_string(img_variant, lang="deu", config=cfg)
        for p in price_re.findall(txt):
            hits.add(p.replace(",", "."))
    return hits

all_prices = set()
debug_hits = []

for i, (x, y, cw, ch, area) in enumerate(boxes):
    pad = 8
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + cw + pad, w); y1 = min(y + ch + pad, h)
    roi = gray0[y0:y1, x0:x1]

    # Digit-Boost: Upscale (macht bei großen Preisen viel aus)
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Varianten
    v1 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    v2 = 255 - v1
    v3 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    v4 = 255 - v3

    hits = set()
    for v in (v1, v2, v3, v4):
        hits |= ocr_try(v)

    # Sammeln & Debug speichern, wenn Treffer
    if hits:
        all_prices |= hits
        # speichere eine Variante als Beleg für die Präsentation
        out = DEBUG_DIR / f"hit_roi_{i:02d}_{'_'.join(sorted(hits))}.png"
        cv2.imwrite(str(out), v1)
        debug_hits.append((i, sorted(hits), str(out)))

print("Boxes:", len(boxes))
print("FOUND PRICES:", sorted(all_prices, key=lambda x: float(x)))
print("Saved hit ROIs:", len(debug_hits))
if debug_hits:
    print("Example hit:", debug_hits[0])
