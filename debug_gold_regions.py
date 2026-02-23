"""Visualize the boxes and search regions to debug the 6.29€ issue."""
import cv2
import json
import numpy as np

# Load annotation
with open("data/annotations/page_coffee.json", "r", encoding="utf-8") as f:
    data = json.load(f)

img = cv2.imread(data[0]["image"])
H, W = img.shape[:2]

# Focus on the 6.29€ box
item = data[0]["items"][0]  # First item with price 6.29
box = item["box"]
x, y, w, h = box

print(f"Image size: {W} x {H}")
print(f"Box: x={x}, y={y}, w={w}, h={h}")
print(f"Box position: ({x}, {y}) to ({x+w}, {y+h})")

# Draw the price box
img_vis = img.copy()
cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Draw search regions (same logic as in OCR code)
regions_info = []

# Region 1: LEFT
x1_left = max(0, x - int(0.05 * w))
x0_left = max(0, x - int(3.5 * w))
y0_left = max(0, y - int(1.0 * h))
y1_left = min(H, y + h + int(1.0 * h))
if x1_left - x0_left >= 60 and y1_left - y0_left >= 30:
    cv2.rectangle(img_vis, (x0_left, y0_left), (x1_left, y1_left), (255, 0, 0), 2)
    regions_info.append(("LEFT", x0_left, y0_left, x1_left, y1_left))
    print(f"LEFT region: ({x0_left}, {y0_left}) to ({x1_left}, {y1_left})")

# Region 2: ABOVE
x0_above = max(0, x - int(0.8 * w))
x1_above = min(W, x + w + int(0.8 * w))
y1_above = max(0, y - int(0.05 * h))
y0_above = max(0, y - int(2.5 * h))
if y1_above - y0_above >= 30 and x1_above - x0_above >= 60:
    cv2.rectangle(img_vis, (x0_above, y0_above), (x1_above, y1_above), (0, 255, 255), 2)
    regions_info.append(("ABOVE", x0_above, y0_above, x1_above, y1_above))
    print(f"ABOVE region: ({x0_above}, {y0_above}) to ({x1_above}, {y1_above})")

# Region 3: LEFT-ABOVE
x0_la = max(0, x - int(2.5 * w))
x1_la = max(0, x - int(0.1 * w))
y0_la = max(0, y - int(1.5 * h))
y1_la = max(0, y + int(0.3 * h))
if x1_la - x0_la >= 60 and y1_la - y0_la >= 30:
    cv2.rectangle(img_vis, (x0_la, y0_la), (x1_la, y1_la), (255, 255, 0), 2)
    regions_info.append(("LEFT-ABOVE", x0_la, y0_la, x1_la, y1_la))
    print(f"LEFT-ABOVE region: ({x0_la}, {y0_la}) to ({x1_la}, {y1_la})")

# Save visualization
cv2.imwrite("debug_gold_coffee_regions.png", img_vis)
print("\nVisualization saved to: debug_gold_coffee_regions.png")

# Extract and save each region
for i, (name, x0, y0, x1, y1) in enumerate(regions_info):
    region = img[y0:y1, x0:x1]
    cv2.imwrite(f"debug_region_{i}_{name}.png", region)
    print(f"Saved region {i} ({name}): {region.shape}")
