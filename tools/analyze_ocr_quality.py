"""Analyze OCR quality for specific boxes"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import json
from src.ocr.pipeline.ocr_product_text import extract_product_text_with_nlp

# Load results to see which boxes had issues
page10_json = Path("data/annotations/KW40_25_ebeae15a-90e5-4975-a5cd-ddd640c8c977_page10.json")
page10_image = Path("data/images/aldi/KW40_25_ebeae15a-90e5-4975-a5cd-ddd640c8c977_page10.png")

with open(page10_json) as f:
    data = json.load(f)

# Get items list
items = data[0]["items"]

img = cv2.imread(str(page10_image))

print("=" * 80)
print("OCR Quality Analysis - Page 10")
print("=" * 80)

# Analyze problematic boxes: 2 (bad OCR), 6 (garbage)
problem_boxes = [1, 5]  # Box indices (0-based: 1=Box2, 5=Box6)

for box_idx in problem_boxes:
    item = items[box_idx]
    box = item["box"]
    
    print(f"\nBox {box_idx + 1} | €{item['price']}")
    print(f"Product: {item['product']}")
    ocr_text = item.get('product_nlp', {}).get('ocr_text', '')
    print(f"OCR Text: {ocr_text[:100]}...")
    print(f"Box coordinates: {box}")
    
    # Re-run OCR with different strategies
    print("\nTrying different OCR approaches:")
    
    # Try with original box
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    
    print(f"  Crop size: {crop.shape}")
    
    # Check if crop is valid
    if crop.size == 0:
        print("  ERROR: Empty crop!")
        continue
        
    # Try OCR
    result = extract_product_text_with_nlp(img, box)
    print(f"  Current method: '{result.get('product_name', 'NONE')}'")
    print(f"  OCR text: {result.get('ocr_text', '')[:80]}...")
    print(f"  Method: {result.get('method', 'unknown')}, Confidence: {result.get('confidence', 0):.2f}")

print("\n" + "=" * 80)
