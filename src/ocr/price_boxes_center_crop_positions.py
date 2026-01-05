from pathlib import Path
import cv2
import numpy as np
import pytesseract
import re
import json

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

IMG_PATH = Path("data/images/aldi/page_coffee.png")
DEBUG_DIR = Path("data/debug_price_boxes")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMG_PATH}")

h, w = img.shape[:2]

# --- 1) Blau-Maske (Preisziffern sind meist blau) ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Startbereich für Aldi-Preisblau (ggf. später feinjustieren)
lower_blue = np.array([90, 60, 40], dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Morphologie: Ziffern zusammenführen
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
mask_blue2 = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_blue2 = cv2.dilate(mask_blue2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

cv2.imwrite(str(DEBUG_DIR / "mask_blue.png"), mask_blue)
cv2.imwrite(str(DEBUG_DIR / "mask_blue_morph.png"), mask_blue2)

# --- 2) Contours auf Blau-Maske -> Kandidaten ---
cnts, _ = cv2.findContours(mask_blue2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cands = []
for c in cnts:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cw * ch

    # sehr kleine Flecken raus (punktuelles Blau etc.)
    if area < 400:
        continue

    aspect = cw / max(ch, 1)
    # Preis-Block: eher "breit" oder "kompakt", aber nicht extrem dünn/lang
    if aspect < 0.5 or aspect > 8.0:
        continue

    # Kandidat etwas expandieren, um weißen Preislabel-Rand mitzunehmen
    pad = int(0.25 * max(cw, ch))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + cw + pad, w)
    y1 = min(y + ch + pad, h)

    roi = img[y0:y1, x0:x1]

    # --- 3) Optional: Validierung über "weißes Label" ---
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Weiß: geringe Sättigung, hohe Helligkeit
    mask_white = cv2.inRange(roi_hsv, np.array([0, 0, 180]), np.array([179, 70, 255]))
    white_ratio = (mask_white > 0).mean()

    # Preislabel haben oft viel Weißfläche
    if white_ratio < 0.35:
        continue

    cands.append((x0, y0, x1 - x0, y1 - y0, float(white_ratio), int(area)))

# sortiert für Debug
cands = sorted(cands, key=lambda t: (t[1], t[0]))

dbg = img.copy()
for i, (x, y, cw, ch, wr, _) in enumerate(cands):
    cv2.rectangle(dbg, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
    cv2.putText(dbg, f"{i}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imwrite(str(DEBUG_DIR / "price_cands_boxes.png"), dbg)
print("Price candidates:", len(cands))

price_like = re.compile(r"^\d{1,3}([.,]\d{2})?$")  # erlaubt auch "629" -> später zu 6.29
digits = re.compile(r"\d+")

def normalize_price_token(tok: str):
    t = tok.strip()
    t = t.replace(" ", "")
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace("l", "1").replace("I", "1")

    # falls "6,29" oder "6.29"
    if re.fullmatch(r"\d{1,3}[.,]\d{2}", t):
        return t.replace(",", ".")

    # falls "629" (ohne Trennzeichen) -> letzte 2 Ziffern = Cent
    if re.fullmatch(r"\d{3,5}", t):
        return f"{int(t[:-2])}.{t[-2:]}"  # formatbasiert, nicht wertbasiert

    return None

def ocr_price_from_roi(roi_bgr):
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Blau-Maske (wie gehabt)
    mask_blue_roi = cv2.inRange(
        hsv_roi,
        np.array([90, 60, 40], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )

    # etwas schließen, damit Punkt + Ziffern zusammenhängen
    mask_blue_roi = cv2.morphologyEx(
        mask_blue_roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        iterations=1,
    )

    ys, xs = np.where(mask_blue_roi > 0)
    if len(xs) == 0:
        return None, None  # keine blauen Pixel -> kein Preis

    # 1) Tight bounding box um die (blauen) Ziffern
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Padding (relativ), damit nichts abgeschnitten wird
    pad = int(0.28 * max(x1 - x0 + 1, y1 - y0 + 1))
    x0 = max(x0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, roi_bgr.shape[1] - 1)
    y1 = min(y1 + pad, roi_bgr.shape[0] - 1)

    crop = roi_bgr[y0:y1 + 1, x0:x1 + 1]

    # 2) OCR-Input bauen: Graustufen + Otsu, schwarze Ziffern auf weiß
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Falls invertiert (weißes Objekt auf schwarz), dann invertieren:
    # Heuristik ist rein bildbasiert: Hintergrund ist überwiegend weiß
    if (thr == 0).mean() < 0.5:   # zu wenig schwarz -> vermutlich invertiert
        thr = 255 - thr

    # kleine Partikel entfernen (gegen Artefakte)
    thr = cv2.morphologyEx(
    thr,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    iterations=1
)


    # Upscale
    thr = cv2.resize(thr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # 3) OCR: lieber image_to_string + Regex (bei so cleanen Bildern stabiler)
    cfgs = [
        r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,",
        r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,",
        r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,",
    ]

    best = None
    for cfg in cfgs:
        txt = pytesseract.image_to_string(thr, lang="eng", config=cfg).strip()

        # direktes Preisformat finden
        m = re.search(r"\d{1,3}[.,]\d{2}", txt)
        if m:
            best = m.group(0).replace(",", ".")
            break

        # sonst: nur Ziffern zusammenziehen (z.B. "759" -> 7.59)
        digits_only = re.sub(r"[^\d]", "", txt)
        if re.fullmatch(r"\d{3,5}", digits_only or ""):
            best = f"{int(digits_only[:-2])}.{digits_only[-2:]}"
            break

    return best, thr

detections = []
for i, (x, y, cw, ch, wr, _) in enumerate(cands):
    roi = img[y:y+ch, x:x+cw]
    price, bin_img = ocr_price_from_roi(roi)

    # Debug speichern (auch wenn price None)
    cv2.imwrite(str(DEBUG_DIR / f"cand_{i:02d}_bin.png"), bin_img if bin_img is not None else roi)

    if price:
        cx = x + cw // 2
        cy = y + ch // 2
        detections.append({
            "price": price,
            "x": int(cx),
            "y": int(cy),
            "box": [int(x), int(y), int(cw), int(ch)],
            "white_ratio": float(wr),
            "roi_file": str(DEBUG_DIR / f"cand_{i:02d}_bin.png")
        })

print("Detections:", len(detections))
prices_unique = sorted({d["price"] for d in detections}, key=lambda s: float(s))
print("FOUND PRICES:", prices_unique)

out_json = DEBUG_DIR / "price_detections.json"
out_json.write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", out_json)
