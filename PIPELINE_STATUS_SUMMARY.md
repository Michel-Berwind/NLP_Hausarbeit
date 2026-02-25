# Pipeline Status Summary - NLP Hausarbeit Project

## Use Case / Project Goal

**Master's Thesis**: Automatic product and price extraction from German discount store promotional flyers (Aldi)

**Pipeline Architecture**: Computer Vision → OCR → NLP
1. **Detection**: Find blue/white price boxes using CV (color masks + morphology)
2. **OCR**: Extract prices from boxes + product text from surrounding regions
3. **NLP**: Extract clean product names using spaCy German model

**Evaluation**: Match extracted (product, price) pairs against manually labeled ground truth
- Metric: F1 score based on RapidFuzz fuzzy matching (threshold=0.80)
- Dataset: 25 Aldi pages (manually annotated)
- **Current F1 Score: 0.18-0.20 (18-20%)**
- **Target F1 Score: 0.65-0.75 (65-75%)**

---

## Status Update (2026-02-24)

### ✅ End-to-end use case is running again

The full thesis workflow now executes without the previous hard blockers:

1. **Batch runner fixed** (`run_pipeline_all.py`)
  - Replaced stale import `process_single_image` with current API `process_image`
  - Writes one prediction JSON per page to `data/predictions/`

2. **Price OCR root cause fixed** (`src/preprocessing/image_preprocessing.py`)
  - Added language normalization for Tesseract (`en -> eng`, `de -> deu`)
  - This fixed empty prices in detections where OCR was silently failing

3. **NLP model bootstrap fixed** (`src/nlp/ner_model.py`)
  - spaCy model download now uses `sys.executable` (active venv) instead of system Python
  - Added fail-safe to avoid repeated crashing initialization attempts

### Quick reproducible runbook

```bash
# 1) Activate your project venv
.venv-3\Scripts\activate

# 2) Ensure spaCy German model is available
python -m spacy download de_core_news_sm

# 3) Smoke test (Aldi page01)
python test_pipeline.py

# 4) Aldi 5-page benchmark
python test_5pages.py

# 5) Evaluate (example: all available predictions)
python -m src.evaluation.evaluate --predictions data/predictions --annotations data/annotations --output results/test_evaluation_latest.json
```

### Current verified metrics

- **Smoke test** (`test_pipeline.py`):
  - `aldi_page01`: 5 offers extracted

- **Aldi 5-page evaluation** (`results/aldi_5pages_eval_latest.json`):
  - Precision: **0.074**
  - Recall: **0.069**
  - F1: **0.071**
  - TP=2, FP=25, FN=27

### Interpretation

The **use case is now operational** (pipeline runs and produces structured outputs), but extraction quality is still below thesis target. The next iteration should focus on detection precision and product-name filtering quality, not infrastructure/debugging blockers.

---

## Current Implementation

### Working Components ✅

**1. Box Detection** (`src/detection/pricebox_detection.py`)
- HSV color space filtering for blue/white boxes
- Morphological operations to merge text into rectangular regions
- Aspect ratio + size filtering to remove false positives
- **Status**: Works reasonably well (detects ~80% of boxes)

**2. Price OCR** (`src/detection/pricebox_detection.py:extract_price_from_crop()`)
- Preprocessing: 5x resize + CLAHE contrast enhancement
- Multiple binary thresholding strategies (Otsu, manual thresholds, inverted)
- Tesseract OCR with PSM modes [7, 13, 6] (single line, raw line, block)
- **Recent Enhancement**: Scoring system for price candidates
  - Bonus: .99 (+5), .49 (+3), .95 (+2) endings
  - Penalty: .27/.29/.77/.79 (-5) likely OCR errors
  - Counter-based voting for prices appearing multiple times
- **Status**: ~70% accurate, but has timeout/hang issues

**3. Product Text Extraction** (`src/ocr/ocr_product_text.py`)
- Multi-region approach: left, above, left-above, right of price box
- Tesseract OCR with multiple PSM modes
- Text quality analysis (boldness, size) to prioritize titles
- Brand bonus scoring (+300 for brand at start)
- **Recent Enhancement**: Limited to first 2-3 lines to avoid description text
- **Status**: Extracts text, but NLP picks wrong parts

**4. NLP Entity Extraction** (`src/nlp/ner_model.py`)
- spaCy German model (de_core_news_sm)
- Known brands list: BARISSIMO, FERREX, TOPCRAFT, WORKZONE, etc.
- **Recent Enhancement**: Added missing Aldi brands (RIDE, HOME, CREATION, GARDENLINE, ACTIV, ENERGY, LILY, DAN, EXPERTIZ)
- **Recent Simplification**: Reduced from complex 480-line scoring to simple "brand + 2-3 nouns" strategy
- **Status**: Still rejecting valid products or extracting wrong parts

**5. Evaluation** (`src/evaluation/evaluate.py`)
- RapidFuzz token_set_ratio for fuzzy product name matching
- Exact price matching after normalization
- Per-page and overall P/R/F1 metrics
- **Status**: Working correctly, shows low accuracy

---

## Critical Problems 🔴

### Problem 1: Tesseract Timeout/Hang Issues (BLOCKING)
**Symptom**: Pipeline hangs indefinitely during OCR on certain image regions
- Python subprocess timeout doesn't work reliably on Windows
- Even with `timeout=5` parameter, pytesseract still hangs
- Happens unpredictably on ~30-40% of pages

**Attempted Fixes**:
- ❌ Added timeout parameter to pytesseract calls → Still hangs
- ❌ Reduced PSM modes from 4 to 2 → Still hangs
- ❌ Removed expensive denoising (fastNlMeansDenoising) → Helped but not enough
- ❌ Tried to integrate EasyOCR → DLL loading errors on Windows (PyTorch c10.dll issue)
- ✅ Wrapped in try/except → Catches some errors but not subprocess hangs

**Impact**: Cannot complete full test run on 5 pages, let alone 25 pages

### Problem 2: Price OCR Accuracy (~70%)
**Examples from aldi_page01**:
- Ground Truth: 2.99 → Prediction: (rejected, no product extracted)
- Ground Truth: 12.99 → Prediction: 12.29 ❌
- Ground Truth: 3.99 → Prediction: 3.27 ❌
- Ground Truth: 5.99 → Prediction: 5.99 ✓

**Root Causes**:
- Tesseract misreads: 9→2, 9→7
- Binary thresholding artifacts
- Price box has complex backgrounds (product images, patterns)

### Problem 3: Product Name Extraction (~40%)
**Examples from aldi_page01**:
```
Ground Truth: "Fahrrad-Spiralkabelschloss"
OCR Text:     "RIDE GO Fahrrad-Spiral- . kabelschloss Versch. Modelle ä/// flexibles Stahlseil..."
NLP Result:   "" (rejected by NLP)
Reason:       Simplified NLP can't find brand "RIDE" followed by nouns (hyphenation breaks it)

Ground Truth: "LED-Premium-Fahrradlampen"  
OCR Text:     "RIDE GO LED-Premium-Fahrradlarr Front-Scheinwerfer und Rücklicht..."
NLP Result:   "RIDE GO LED-Premium-Fahrradlar Front-Scheinwerfer"
Match:        No (wrong product suffix, extra text)
```

**Root Causes**:
- OCR text has artifacts: hyphens, line breaks, special chars (ä///)
- NLP simplified too much: requires brand + nouns, but brand might be missing/misspelled
- Text from product image gets mixed with actual product title
- Multi-line OCR results in concatenated text without proper sentence boundaries

### Problem 4: Scope limited to Aldi
- This workspace is intentionally scoped to Aldi only.

---

## Code Structure

```
NLP_Hausarbeit/
├── src/
│   ├── pipeline.py                          # Main orchestrator
│   ├── preprocessing/
│   │   └── image_preprocessing.py           # Grayscale, CLAHE, morphology, OCR wrapper
│   ├── detection/
│   │   └── pricebox_detection.py            # Box detection + price extraction
│   ├── ocr/
│   │   └── ocr_product_text.py              # Multi-region text extraction
│   ├── nlp/
│   │   └── ner_model.py                     # spaCy entity extraction
│   └── evaluation/
│       └── evaluate.py                      # F1 score calculation
├── data/
│   ├── images/aldi/                         # 25 PNG pages (300 DPI)
│   ├── annotations/                         # Ground truth JSON files
│   └── predictions/                         # Pipeline output (incomplete due to hangs)
└── test_5pages.py                           # Test script for first 5 Aldi pages
```

**Key Files to Review**:
1. `src/detection/pricebox_detection.py` lines 492-620 (price extraction with scoring)
2. `src/ocr/ocr_product_text.py` lines 174-460 (multi-region OCR)
3. `src/nlp/ner_model.py` lines 158-256 (simplified product extraction)
4. `src/preprocessing/image_preprocessing.py` lines 40-66 (OCR wrapper with EasyOCR fallback)

---

## Recent Code Changes (Last Session)

### Phase 1: Price Recognition Enhancement
**File**: `src/detection/pricebox_detection.py`

```python
# OLD: Simple first-match approach
text = pytesseract.image_to_string(binary, config=config)
if is_valid_price(text):
    return text

# NEW: Scoring + voting system
price_candidates = []
for psm in [7, 13, 6]:
    for binary in binaries:
        # Multiple strategies with scoring
        if cents == '99': score += 5
        elif cents in ['27', '29']: score -= 5
        price_candidates.append((price, score))

# Counter voting
price_counts = Counter([p for p, _ in price_candidates])
rescored = [(price, score + (count-1)*3) for price, score in price_candidates]
return sorted(rescored, key=lambda x: x[1], reverse=True)[0][0]
```

### Phase 2: Product Extraction Simplification
**File**: `src/nlp/ner_model.py`

```python
# OLD: Complex 480-line scoring with multiple strategies (product types, entities, chunks)
# NEW: Simple 100-line approach

def _extract_product_name(self, doc):
    # Find brand
    for token in doc:
        if token.text.upper() in self.known_brands:
            product_parts = [token.text]
            break
    
    # Collect 2-3 nouns after brand
    for token in doc[brand_position+1:]:
        if token.pos_ in ["NOUN", "PROPN"]:
            product_parts.append(token.text)
            if len(product_parts) >= 3: break
    
    return " ".join(product_parts)
```

### Added Brands
```python
self.known_brands = {
    # ... existing brands ...
    # NEW: Aldi-specific brands found in test data:
    "RIDE", "GO", "HOME", "CREATION", "GARDENLINE", 
    "ACTIV", "ENERGY", "LILY", "DAN", "EXPERTIZ"
}
```

### OCR Abstraction Layer
**File**: `src/preprocessing/image_preprocessing.py`

```python
def ocr_image(img, lang='deu', config='', method='auto'):
    """Unified OCR interface supporting EasyOCR and Tesseract."""
    if USE_EASYOCR and EASYOCR_AVAILABLE:
        reader = get_easyocr_reader()
        results = reader.readtext(img, detail=0, paragraph=True)
        return '\n'.join(results)
    # Fallback to Tesseract
    return pytesseract.image_to_string(img, lang=lang, config=config)
```

---

## Test Results

### Before Improvements (Baseline)
```
aldi_page01: P=0.20, R=0.20, F1=0.20
aldi_page02: P=0.00, R=0.00, F1=0.00
aldi_page03: P=0.00, R=0.00, F1=0.00
aldi_page04: P=0.67, R=0.40, F1=0.50
aldi_page05: P=0.25, R=0.25, F1=0.25
Overall: F1=0.18
```

### After Improvements (Partial Test - page 1 only)
```
Ground Truth (aldi_page01):
  1. Fahrradschloss                    @ 4.99
  2. Fahrrad-Spiralkabelschloss        @ 2.99
  3. LED-Premium-Fahrradlampen         @ 12.99
  4. Fahrrad-Zubehör                   @ 5.99
  5. Speichenreflektoren               @ 3.99

Predictions:
  1. (no product)                      @ 2.99  ❌
  2. RIDE GO Fahrradschloss            @ 4.99  ✓
  3. RIDE GO LED-Premium-Fahrradlar... @ 12.29 ❌ (price wrong, product partial)
  4. GERMANY RIDE                      @ 5.99  ❌
  5. RIDE GO Fahrrad-Zubehör           @ 3.27  ❌ (price wrong)

Matches: 1/5 = 20% recall (same as before)
```

**Status**: Cannot test pages 2-5 due to Tesseract hangs

---

## What We Need Help With

### Option 1: Fix Tesseract Timeout Issues
**Goal**: Make pipeline complete all 25 pages without hanging

**Possible approaches**:
- Process killer wrapper that forcefully terminates hung Tesseract processes after N seconds
- Image downscaling (reduce resolution by 50% to speed up OCR)
- Skip problematic regions after timeout and continue
- Alternative: Use subprocess.Popen with PIPE and manual timeout handling

### Option 2: Improve OCR Accuracy
**Goal**: Increase price recognition from 70% to 90%+

**Possible approaches**:
- Better binarization for price boxes (currently tries 5 strategies)
- Specialized price-only OCR (Google Vision API, open-source alternatives)
- Pre-trained digit recognition model (MNIST-style)
- Ensemble: Combine Tesseract + digit recognition

### Option 3: Improve NLP Product Extraction
**Goal**: Increase product name extraction from 40% to 70%+

**Possible approaches**:
- Better OCR text cleaning (remove artifacts: ä///, double spaces, line breaks)
- Visual text hierarchy: Prioritize bold/large text (already partially implemented)
- Rule-based extraction for common patterns: "BRAND PRODUCT_TYPE" (e.g., "RIDE Fahrradschloss")
- Fine-tuned NER model on German retail product names
- LLM-based extraction (GPT-4 API for product name extraction from noisy OCR text)

### Option 4: Alternative Architecture
**Goal**: Bypass current issues with completely different approach

**Possible approaches**:
- Use LLM vision model (GPT-4 Vision, Claude 3) to extract products+prices directly from images
- Use commercial OCR API (Google Cloud Vision, Azure Computer Vision) instead of Tesseract
- Two-stage: Crop price boxes → submit to LLM for extraction

---

## Environment Details

- **OS**: Windows 11
- **Python**: 3.12
- **Virtual Environment**: `.venv-3`
- **Key Dependencies**:
  - opencv-python 4.x
  - pytesseract 0.3.x
  - Tesseract OCR: C:\Users\miche\AppData\Local\Programs\Tesseract-OCR
  - spacy 3.x + de_core_news_sm
  - rapidfuzz 3.x
  - easyocr 1.7.2 (installed but DLL error prevents use)

**Dataset**:
- Format: PNG images, 300 DPI
- Annotations: JSON files with structure `{"page_id": "...", "offers": [{"product": "...", "price": "..."}]}`
- Example annotation: `data/annotations/aldi_page01.json`
- Example image: `data/images/aldi/aldi_page01.png`

---

## Questions for Next Steps

1. **Should we invest time fixing Tesseract issues** (timeout handling, image downscaling), or **switch to a different OCR solution**?

2. **Is the simplified NLP approach the right direction**, or should we **revert to the complex scoring system** and just fix specific bugs?

3. **Should we consider LLM-based extraction** for at least the product names (GPT-4 Vision or similar) given the noisy OCR?

4. **What's the minimum viable F1 score for a master's thesis**? Can we document F1=0.20 as "proof of concept" and focus on architecture/methodology instead of accuracy?

5. **How can we make Tesseract not hang on Windows?** This is the biggest blocker right now.

---

## Immediate Next Action

**If you can help us solve the Tesseract timeout issue**, we can:
1. Complete test run on 5 pages
2. Evaluate impact of Phase 1 + Phase 2 improvements
3. Decide whether to continue improving or pivot to different approach

**Current blocker**: `pytesseract.image_to_string()` hangs indefinitely on ~30-40% of image regions, even with `timeout=5` parameter.
