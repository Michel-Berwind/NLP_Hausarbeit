# Price Box Detection - PowerPoint Summary

## Overview
**Goal**: Detect rectangular price label regions on supermarket flyer pages using explainable, deterministic heuristics (no ML)

## Pipeline Architecture

```
Preprocessed Images → Color Segmentation → Filtering → Merging → Sorting → Crops
```

## How Preprocessed Images Are Used

### 1. **morph.png** - Box Detection Input
- **Purpose**: Find connected text regions as potential price boxes
- **Not directly used** in current pipeline (kept for API compatibility)
- Originally intended for contour-based detection

### 2. **hsv** & **lab** - Color Space Conversions
- **HSV**: Used for color-based segmentation (easier to isolate blue/white regions)
- **LAB**: Used for brightness analysis (L channel = lightness)
- **Created fresh** in `detect_price_boxes()` for each detection pass

### 3. Color-Based Masks (created on-demand)
The pipeline detects **two types of price boxes**:

#### **Blue Boxes with White Text** (Primary)
- **Box Mask**: HSV range [85-135 hue, 100-255 sat, 100-255 val] + L > 50
- **Text Mask**: White text [0-180 hue, 0-50 sat, 180-255 val]

#### **White Boxes with Colored Text** (Secondary)
- **Box Mask**: HSV [0-180 hue, 0-60 sat, 180-255 val] + L > 180
- **Text Mask**: Blue/red/dark text regions combined

## Detection Steps

### Step 1: Connected Components Analysis
```python
cv2.connectedComponentsWithStats(box_mask)
```
- Finds all connected white/blue regions in the image
- Each region = potential price box candidate

### Step 2: Multi-Stage Filtering
Each candidate box must pass **6 heuristic filters**:

| Filter | Purpose | Threshold |
|--------|---------|-----------|
| **1. Area** | Eliminate small noise | ≥ 8,000 px² |
| **2. Dimensions** | Reasonable box size | Width ≥120px, Height ≥80px |
| **3. Aspect Ratio** | Not too elongated | 0.8 ≤ w/h ≤ 2.5 |
| **4. Text Ratio** | Must contain text | ≥5% text pixels |
| **5. Brightness** | Correct background color | L in [40-180] (blue) or L>160 (white) |
| **6. Contrast** | Text visible vs background | ΔL ≥ 30 |

### Step 3: Merge Overlapping Boxes
- Calculate IoU (Intersection over Union) between all box pairs
- Merge boxes with **IoU > 0.3** into single encompassing box
- Prevents duplicate detections of same price

### Step 4: Reading Order Sort
```python
sorted(boxes, key=lambda b: (b[1] // 20, b[0]))
```
- Primary: Y-coordinate (top to bottom)
- Secondary: X-coordinate (left to right)
- Groups boxes in same row

### Step 5: Crop Extraction
- Extract image region for each box
- Save to `debug_dir/crops/box_XX.png`
- Pass to OCR for price extraction

## Detection Statistics (Example: page_coffee)

```
Found 127 white regions
  ↓
Filtered to 18 valid boxes (109 rejected)
  ↓
Merged to 12 final boxes
```

**Common rejection reasons**:
- `too_small`: Area < 8000px²
- `no_text`: Text ratio < 5%
- `bad_aspect_ratio`: Too elongated
- `low_contrast`: Text not visible enough

## Key Design Decisions

### ✅ Explainability
- **All thresholds named and justified** in `PriceBoxConfig` class
- No black-box ML models
- Deterministic: same input = same output

### ✅ Color-Based Approach
- **Why not use morph.png contours?** 
  - Color segmentation more reliable for distinguishing price boxes from product names
  - Blue/white backgrounds uniquely identify price tags
  - Contours would detect all text regions indiscriminately

### ✅ Two-Pass Detection
- **Pass 1**: Blue boxes (promotional prices)
- **Pass 2**: White boxes (standard prices)
- Combines results for comprehensive coverage

### ✅ Robust Filtering
- Multiple complementary filters catch different failure modes
- Text ratio ensures boxes contain actual text, not just colored regions
- Contrast filter ensures OCR-readable text

## Debug Outputs

For each detection run, saved to `data/debug_price_boxes/[image_name]/`:

```
01_box_mask.png          - Detected blue/white regions
01_text_mask.png         - Detected text regions
02_boxes_before_merge.png - Boxes after filtering
03_boxes_final.png       - Final merged & sorted boxes
crops/box_00.png         - Individual box crops for OCR
crops/box_01.png
...
```

## Relaxed Mode Fallback

If standard detection finds no/few boxes:
- **Loosen color thresholds** (wider HSV ranges, lower L thresholds)
- **Retry detection** with more permissive parameters
- Catches edge cases: faded prints, off-white backgrounds

---

## Summary for Slide

**3 Key Points:**

1. **Input**: Uses HSV+LAB color spaces (not morph.png) for color-based segmentation
2. **Method**: 6-stage heuristic filtering (area, size, aspect ratio, text presence, brightness, contrast)
3. **Output**: Sorted, merged bounding boxes with extracted crops ready for OCR

**Advantage**: Fully explainable, deterministic, thesis-appropriate approach
