"""Test the improved NLP extraction on the coffee page."""
import sys
import io
import cv2
import json
from src.ocr.pipeline.ocr_product_text import extract_product_text_with_nlp

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the annotation file to get boxes
with open("data/annotations/page_coffee.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load the image
img_path = data[0]["image"]
img = cv2.imread(img_path)

print("=" * 80)
print("Testing Improved NLP Extraction on Coffee Page")
print("=" * 80)

expected = {
    "6.29": "Express-Kaffee Gold",
    "7.79": 'Mahlkaffee "Unser Bester"',  # Note: Price in JSON is 7.79, not 7.59
    "13.99": "Espresso Cremoso oder Caffè Gustoso",
    "4.29": "Kaffeekapseln",
    "3.99": "Kaffeekapseln"
}

for item in data[0]["items"]:
    box = item["box"]
    price = item["price"]
    
    print(f"\n{'=' * 80}")
    print(f"Preis: {price}€")
    print(f"Expected: {expected.get(price, 'UNKNOWN')}")
    print(f"Box: {box}")
    print("-" * 80)
    
    # Extract with NLP
    result = extract_product_text_with_nlp(img, box)
    
    print(f"Extracted: {result['product_name']}")
    print(f"OCR Text: {result['ocr_text'][:100]}...")
    print(f"Method: {result['method']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result.get('brands'):
        print(f"Brands: {', '.join(result['brands'][:3])}")
    
    # Check if correct
    expected_name = expected.get(price, "").upper()
    extracted_name = result['product_name'].upper()
    
    # Fuzzy match - check if key words are present
    key_words = {
        "6.29": ["EXPRESS", "KAFFEE", "GOLD"],
        "7.79": ["MAHLKAFFEE", "UNSER", "BESTER"],
        "13.99": ["ESPRESSO", "CREMOSO", "CAFFÈ", "GUSTOSO", "ODER"],
        "4.29": ["KAFFEEKAPSELN", "KAPSELN"],
        "3.99": ["KAFFEEKAPSELN", "KAPSELN"]
    }
    
    expected_keywords = key_words.get(price, [])
    matches = sum(1 for kw in expected_keywords if kw in extracted_name)
    
    if matches >= 2 or (matches >= 1 and price in ["4.29", "3.99"]):
        print("✓ MATCH (contains key product words)")
    else:
        print("✗ MISMATCH")

print(f"\n{'=' * 80}")
print("Test complete!")
