from pathlib import Path
import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

IMG_PATH = Path("data/images/aldi/page_coffee.png")

img = cv2.imread(str(IMG_PATH))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kontrast erhöhen + binarisieren (gut für Preisboxen)
gray = cv2.GaussianBlur(gray, (3,3), 0)
thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# OCR: nur Zahlen + Punkt/Komma
config = r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,"

text = pytesseract.image_to_string(thr, lang="deu", config=config)

# Preise erkennen: 6.29 / 7.59 / 13.99 / 4.29 / 3.99 oder 6,29 etc.
prices = re.findall(r"\b\d{1,2}[.,]\d{2}\b", text)
prices = [p.replace(",", ".") for p in prices]

print("RAW OCR:", text[:500])
print("FOUND PRICES:", sorted(set(prices)))
