# Page Coffee Preprocessing Analysis

## Overview
The `page_coffee` image goes through **12 distinct preprocessing stages** divided into two categories:
1. **Text-focused preprocessing** (5 stages) - optimized for OCR quality
2. **Detection-focused preprocessing** (7 stages) - optimized for finding price box regions

## Preprocessing Pipeline Flow

```
Original Image (page_coffee.png)
    ↓
┌─────────────────────────────────────────┐
│  preprocess_for_text()                  │
│  ├─ gray: grayscale conversion          │
│  ├─ enhanced: CLAHE contrast boost      │
│  ├─ binary: adaptive thresholding       │
│  ├─ morph_text: cleaned for OCR         │
│  └─ morph: connected blocks for detect  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  preprocess_for_detection()             │
│  ├─ hsv: HSV color space                │
│  ├─ lab: LAB color space                │
│  ├─ L: Lightness channel                │
│  ├─ white_mask: white regions           │
│  ├─ blue_mask: blue text regions        │
│  ├─ dark_mask: dark text regions        │
│  └─ text_mask: combined blue+dark       │
└─────────────────────────────────────────┘
```

## Detailed Stage Descriptions

### Text-Focused Stages

#### 1. **gray.png** - Grayscale Conversion
- **Function**: `to_grayscale()`
- **Method**: BGR → Grayscale conversion using `cv2.COLOR_BGR2GRAY`
- **Purpose**: Reduces color complexity, baseline for subsequent processing
- **Use Case**: Starting point for all text-based operations

#### 2. **enhanced.png** - Contrast Enhancement
- **Function**: `enhance_contrast()`
- **Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Clip limit: 2.0
  - Grid size: 8×8 tiles
- **Purpose**: Boost local contrast without over-amplifying noise
- **Use Case**: Makes faded text more readable, handles uneven lighting

#### 3. **binary.png** - Adaptive Thresholding
- **Function**: `adaptive_binarize()`
- **Method**: Gaussian adaptive thresholding
  - Block size: 21×21 pixels
  - Constant: 8
- **Purpose**: Convert to black/white while handling variable lighting across image
- **Use Case**: Isolates text strokes from background, robust to shadows

#### 4. **morph_text.png** - Morphological Refinement for OCR
- **Function**: `morph_refine()`
- **Method**: 
  - Close operation (3×3 rect kernel, 1 iteration) - fills small gaps in characters
  - Open operation (2×2 rect kernel, 1 iteration) - removes small noise
- **Purpose**: Clean text for better OCR accuracy
- **Use Case**: Used by Tesseract when reading price numbers

#### 5. **morph.png** - Morphological Processing for Detection
- **Function**: `morph_for_pricebox_detection()`
- **Method**: Complex multi-step process:
  1. Invert binary (white text on black background)
  2. Horizontal dilation (9×2 kernel) - connects characters into numbers (e.g., "2.99")
  3. Vertical dilation (2×5 kernel) - connects multi-line labels
  4. Close operation (5×5 kernel) - fills holes within regions
- **Purpose**: Merge nearby text into solid rectangular blocks representing price tags
- **Use Case**: Used by contour detection to find price box boundaries

### Detection-Focused Stages

#### 6. **hsv.png** - HSV Color Space
- **Function**: Part of `preprocess_for_detection()`
- **Method**: BGR → HSV conversion
- **Purpose**: Color-based segmentation (easier to isolate hues)
- **Use Case**: Base for white_mask, blue_mask, dark_mask

#### 7. **lab.png** - LAB Color Space
- **Function**: Part of `preprocess_for_detection()`
- **Method**: BGR → LAB conversion
- **Purpose**: Perceptually uniform color space
- **Use Case**: Provides L channel for lightness-based detection

#### 8. **L.png** - Lightness Channel
- **Function**: Extracted from LAB color space
- **Method**: LAB[:,:,0] - the L (lightness) channel
- **Purpose**: Isolate brightness independent of hue
- **Use Case**: Combined with HSV for robust white detection

#### 9. **white_mask.png** - White Region Detection
- **Function**: Part of `preprocess_for_detection()`
- **Method**: 
  - HSV range: H[0-180], S[0-60], V[180-255]
  - L channel threshold: >180
  - Morphological closing: 13×13 rect kernel, 2 iterations
- **Purpose**: Detect price boxes with white backgrounds
- **Use Case**: Most price tags have white/light backgrounds for contrast

#### 10. **blue_mask.png** - Blue Text Detection
- **Function**: Part of `preprocess_for_detection()`
- **Method**: HSV range: H[90-150], S[20-255], V[120-255]
- **Purpose**: Detect blue-colored text (common in promotional prices)
- **Use Case**: Marketing prices often use blue for emphasis

#### 11. **dark_mask.png** - Dark Text Detection
- **Function**: Part of `preprocess_for_detection()`
- **Method**: HSV range: H[0-180], S[0-255], V[0-110]
- **Purpose**: Detect dark/black text
- **Use Case**: Standard price text color

#### 12. **text_mask.png** - Combined Text Mask
- **Function**: Part of `preprocess_for_detection()`
- **Method**: Bitwise OR of blue_mask and dark_mask
- **Purpose**: Unified mask of all text regions regardless of color
- **Use Case**: Comprehensive text detection for price box validation

## How These Are Used in the Pipeline

### Stage 1: Price Box Detection
```python
# Uses: morph.png (connected blocks) + white_mask.png
1. Find contours in morph.png → potential price box boundaries
2. Filter contours by:
   - Size (width, height, area)
   - Aspect ratio
   - Position on page
3. Check overlap with white_mask → likely has white background
4. Return bounding boxes of detected price boxes
```

### Stage 2: OCR on Detected Boxes
```python
# Uses: morph_text.png (cleaned text)
1. Crop each price box from original image
2. Apply morph_text preprocessing to crop
3. Run Tesseract OCR with price-specific config
4. Parse prices from OCR text
```

### Stage 3: Product Text Extraction
```python
# Uses: enhanced.png (contrast boosted) + adaptive thresholding
1. For each price box, expand upward to find product name region
2. Apply text preprocessing to that region
3. Run Tesseract to extract product description
```

## Relaxed Mode

If initial detection finds no/few boxes, the pipeline tries again with **relaxed parameters**:

| Parameter | Normal Mode | Relaxed Mode |
|-----------|-------------|--------------|
| White HSV | V≥180, S≤60 | V≥170, S≤70 |
| White L   | L>180       | L>150        |
| Blue HSV  | H[90-150], S[20-255], V≥120 | H[75-165], S[5-255], V≥80 |
| Dark V    | V≤110       | V≤140        |

This catches price boxes with:
- Slightly off-white backgrounds
- Faded or light blue text
- Gray text instead of pure black

## Summary

**For the page_coffee image:**
- All 12 preprocessing outputs are saved in `data/debug_price_boxes/preprocess/page_coffee/`
- The pipeline first tries normal detection parameters
- If that fails, it retries with relaxed parameters and saves another set of masks
- The goal: robustly detect price boxes despite variations in printing, lighting, and scan quality
- Key innovation: Separate preprocessing for **detection** (finding boxes) vs **OCR** (reading text)
