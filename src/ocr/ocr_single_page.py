from pathlib import Path
from PIL import Image
import pytesseract

# 1) Pfad zu tesseract.exe anpassen
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\miche\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

IMG_PATH = Path("data/images/aldi/page_coffee.png")
OUT_PATH = Path("data/ocr_text/aldi/page_coffee.txt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 2) OCR-Konfiguration: gut für Prospekttext
config = r"--oem 3 --psm 6"  # OEM 3 = default LSTM, PSM 6 = Block of text

text = pytesseract.image_to_string(Image.open(IMG_PATH), lang="deu", config=config)

OUT_PATH.write_text(text, encoding="utf-8")
print(f"Saved OCR text to: {OUT_PATH}")
