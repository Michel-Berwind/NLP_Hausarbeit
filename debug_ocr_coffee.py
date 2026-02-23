"""Debug OCR output to see what text is actually extracted."""
import cv2
import json
from src.ocr.pipeline.ocr_product_text import extract_product_text_with_nlp

# Load annotation
with open("data/annotations/page_coffee.json", "r", encoding="utf-8") as f:
    data = json.load(f)

img = cv2.imread(data[0]["image"])

for item in data[0]["items"]:
    box = item["box"]
    price = item["price"]
    
    print(f"\n{'=' * 80}")
    print(f"Preis: {price}€")
    
    result = extract_product_text_with_nlp(img, box)
    
    print(f"\nFull OCR Text:")
    print(result['ocr_text'])
    print(f"\nExtracted Product: {result['product_name']}")
    print(f"NLP Entities: {result.get('nlp_entities', [])[:3]}")
    print(f"Noun Chunks: {result.get('noun_chunks', [])[:5]}")
