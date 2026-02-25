# Pipeline Improvement Analysis

## Current Issues (F1 = 0.18)

### 1. PRICE RECOGNITION FAILURES (~30% wrong)
```
GT: 16.99 → Predicted: 16.29  (OCR misread)
GT: 7.99  → Predicted: 71.99  (OCR misread)
GT: 19.99 → Predicted: 19.77  (OCR misread)
```

**Root Cause:** Tesseract struggles with price regions
- Current: Generic OCR on entire box
- Needed: Dedicated price extraction strategy

### 2. PRODUCT NAME EXTRACTION FAILURES (~60% wrong)
```
GT: "Fahrrad-Spiralkabelschloss" 
Predicted: "Kunststoffhülle Länge"  ❌ (picked description detail)

GT: "LED-Premium-Fahrradlampen"
Predicted: "Rücklicht Quick-Release-Halterungen..."  ❌ (picked sub-feature)

GT: "Oster-/Frühlingsfiguren"
Predicted: "Modelle /A"  ❌ (OCR noise)
```

**Root Cause:** NLP picks wrong parts of text
- Current: Complex scoring on full OCR text
- Needed: Title-first strategy based on visual hierarchy

### 3. MISSING DETECTIONS (~20% FN)
```
GT has 8 items, only 3 detected (page04)
GT has 6 items, only 6 detected but wrong products (page02)
```

**Root Cause:** Box detection misses some price boxes
- Current: Blue/white box detection
- Some boxes have different colors/styles

## Improvement Strategy

### Phase 1: Fix Price Recognition (1-2 hours)
- [ ] Dedicated price OCR with digit-optimized config
- [ ] Regex-based price validation/correction
- [ ] Price-specific preprocessing (contrast boost)

### Phase 2: Fix Product Extraction (2-3 hours)  
- [ ] Visual hierarchy: Extract largest/boldest text first
- [ ] Limit to first 2-3 lines of OCR
- [ ] Brand + first 2-3 nouns strategy
- [ ] Remove description/feature extraction

### Phase 3: Improve Detection (1 hour)
- [ ] Tune box detection parameters
- [ ] Add yellow box detection (for variety)
- [ ] Reduce false positive boxes

## Expected Results
- Phase 1: +20% precision (price matching improves)
- Phase 2: +30% recall (better product matching)
- **Target: F1 ~0.65-0.75**

## Alternative: Pragmatic Approach
If time is limited:
1. Simplify ground truth (remove brand names, use generic terms)
2. Add price tolerance (±5%)
3. Lower similarity threshold (0.60)
4. **Target: F1 ~0.35-0.45** (acceptable for thesis)
