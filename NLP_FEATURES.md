# NLP Features in the Flyer Price Extraction Pipeline

## Overview

This project now incorporates **Natural Language Processing (NLP)** techniques using **spaCy** to extract structured product information from supermarket flyers. The system combines traditional OCR with modern NLP to achieve better accuracy and explainability.

---

## NLP Techniques Implemented

### 1. **Named Entity Recognition (NER)**
- **Library**: spaCy's `de_core_news_sm` German language model
- **Purpose**: Automatically identify and classify entities in text
- **Entity Types Detected**:
  - `ORG`: Organizations (brand names like "NESCAFÉ", "JACOBS")
  - `PRODUCT`: Product names
  - `MISC`: Miscellaneous entities (product variants, flavors)
  - `PER`: Person names (sometimes misclassified product names)

**Example**:
```
Input: "NESCAFÉ Gold Kaffee 200g"
Entities: [("NESCAFÉ", "ORG")]
```

### 2. **Part-of-Speech (POS) Tagging**
- **Purpose**: Identify grammatical roles of words to filter OCR noise
- **Tags Used**:
  - `NOUN`: Common nouns (product types: "Kaffee", "Kapseln")
  - `PROPN`: Proper nouns (brand names: "JACOBS", "LINDT")
  - `ADJ`: Adjectives (product attributes: "Premium", "Vollmundig")
  - `NUM`: Numbers (quantities)

**Noise Filtering**: Removes verbs, prepositions, and conjunctions that are likely OCR errors

**Example**:
```
Input: "Aus unserem Sortiment TCHIBO Caffè"
POS Tags: [("Aus", "ADP"), ("unserem", "DET"), ("Sortiment", "NOUN"), 
           ("TCHIBO", "PROPN"), ("Caffè", "PROPN")]
Filtered: "Sortiment TCHIBO Caffè"  # Keeps only NOUN/PROPN
```

### 3. **Dependency Parsing**
- **Purpose**: Understand syntactic relationships between words
- **Applications**:
  - Extract compound product names (e.g., "Jacobs Krönung")
  - Link attributes to products (e.g., "Premium" → "Espresso")
  - Identify quantity-unit relationships

**Example**:
```
Input: "BARISSIMO Espresso Cremoso"
Dependencies: BARISSIMO ← Espresso → Cremoso
Result: "BARISSIMO Espresso Cremoso" (complete product name)
```

### 4. **Noun Chunk Extraction**
- **Purpose**: Extract multi-word noun phrases that represent complete product names
- **Method**: Uses spaCy's syntactic parser to identify noun phrases

**Example**:
```
Input: "Premium Espresso Bohnen 500g"
Noun Chunks: ["Premium Espresso Bohnen", "500g"]
```

### 5. **Token-Level Linguistic Features**
- **Features Used**:
  - `is_upper`: Detect ALL-CAPS brand names
  - `is_alpha`: Filter numeric OCR noise
  - `pos_`: Part-of-speech tag
  - `lemma_`: Base form for normalization
  - `is_stop`: Identify stop words

**Example**:
```python
token.text = "BARISSIMO"
token.is_upper = True    # → Likely a brand
token.is_alpha = True    # → Valid text (not noise)
token.pos_ = "PROPN"     # → Proper noun (brand/product)
```

### 6. **Custom Pattern Matching (spaCy Matcher)**
- **Purpose**: Define custom rules for brand and product detection
- **Patterns Implemented**:
  - Brand pattern: Capitalized words (length > 2)
  - Product pattern: [ADJ]* + NOUN + [NOUN]*

**Example Pattern**:
```python
# Pattern: Optional adjectives + Noun + Optional nouns
pattern = [{"POS": "ADJ", "OP": "*"}, 
           {"POS": "NOUN"}, 
           {"POS": "NOUN", "OP": "*"}]
Matches: "Premium Espresso Bohnen", "Vollmilch Schokolade"
```

### 7. **Semantic Scoring with Linguistic Features**
- **Purpose**: Calculate confidence scores for extracted text
- **Features**:
  - Entity presence (bonus for recognized entities)
  - POS distribution (bonus for proper nouns)
  - Brand matching (bonus for known brands)
  - Text quality (penalty for excessive digits)

**Confidence Calculation**:
```python
base_score = 0.5
+ 0.2 if entities found
+ 0.15 if proper nouns present
+ 0.15 if known brand detected
- 0.2 if digit_ratio > 0.4
```

---

## Pipeline Architecture

### OCR → NLP Integration Flow

```
┌─────────────┐
│ OCR (Raw)   │ ← Tesseract extracts text
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Text Cleaning   │ ← Regex-based preprocessing
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ spaCy NLP       │ ← Linguistic analysis
│  - Tokenization │
│  - POS Tagging  │
│  - NER          │
│  - Parsing      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Entity Extract  │ ← Extract structured data
│  - Product name │
│  - Brands       │
│  - Quantities   │
│  - Attributes   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ JSON Output     │ ← Structured results
└─────────────────┘
```

---

## Code Examples

### Basic Usage

```python
from src.extraction.ner_model import extract_product_entities

# Extract entities from OCR text
text = "BARISSIMO Espresso Cremoso 500-g-Packung"
result = extract_product_entities(text)

print(result["product_name"])  # "BARISSIMO Espresso Cremoso"
print(result["brands"])         # ["BARISSIMO", "Espresso"]
print(result["quantities"])     # [{"value": "500", "unit": "g"}]
print(result["confidence"])     # 0.95
```

### Integration with OCR Pipeline

```python
from src.ocr.pipeline.ocr_product_text import extract_product_text_with_nlp

# Extract product text with NLP from image region
product_result = extract_product_text_with_nlp(img, box)

# Access NLP results
print(product_result["product_name"])      # Clean product name
print(product_result["nlp_entities"])      # spaCy entities
print(product_result["noun_chunks"])       # Noun phrases
print(product_result["pos_filtered"])      # POS-filtered text
print(product_result["method"])            # "nlp" or "ocr_only"
```

---

## NLP Model Details

### spaCy Model: `de_core_news_sm`

- **Language**: German
- **Components**:
  - Tokenizer (linguistic tokenization)
  - Tagger (POS tagging)
  - Parser (dependency parsing)
  - NER (named entity recognition)
  - Lemmatizer (word normalization)

- **Training Data**: German news text corpus
- **Vocabulary Size**: ~20,000 unique tokens
- **Performance**: ~97% POS accuracy, ~89% NER F-score

### Installation

```bash
pip install spacy
python -m spacy download de_core_news_sm
```

---

## Comparison: Before vs After NLP

### Before (Regex-Only)
```python
Input: "Aus unserem Sortiment TCHIBO Caffè Crema Vollmundig 500g"
Output: "TCHIBO Caffè Crema Vollmundig 500g"  # Manual regex cleaning
Issues:
- Hardcoded brand list
- No structure (just string)
- No confidence score
- Can't handle variations
```

### After (With NLP)
```python
Input: "Aus unserem Sortiment TCHIBO Caffè Crema Vollmundig 500g"
Output: {
  "product_name": "TCHIBO Caffè Crema Vollmundig",
  "brands": ["TCHIBO"],
  "entities": [("TCHIBO Caffè Crema Vollmundig", "MISC")],
  "noun_chunks": ["unserem Sortiment", "TCHIBO Caffè Crema"],
  "quantities": [{"value": "500", "unit": "g"}],
  "pos_filtered": "Sortiment TCHIBO Caffè Crema Vollmundig",
  "confidence": 0.95,
  "method": "nlp"
}

Benefits:
✓ Structured entity extraction
✓ Automatic brand detection via NER
✓ Confidence scores for reliability
✓ POS-based noise filtering
✓ Quantity extraction with units
✓ Explainable with linguistic features
```

---

## Academic Justification for NLP Class

### NLP Techniques Demonstrated

1. **Token-level Processing**: Tokenization, POS tagging, linguistic features
2. **Syntactic Analysis**: Dependency parsing, noun chunking
3. **Semantic Analysis**: Named entity recognition, entity typing
4. **Pattern Matching**: Rule-based NLP with spaCy Matcher
5. **Confidence Estimation**: Feature-based scoring
6. **Multi-lingual NLP**: German language processing

### Research Relevance

- **Information Extraction**: Classic NLP task (extracting structured data from text)
- **Domain Adaptation**: Applying NLP to retail/OCR domain
- **Hybrid Approach**: Combining rule-based and ML-based NLP
- **Error Handling**: Dealing with noisy OCR input (realistic NLP challenge)

---

## Testing & Validation

Run the NLP feature demo:
```bash
python test_nlp_features.py
```

Run the full pipeline with NLP:
```bash
python -m src.ocr.pipeline.pipeline_runner --input-dir "data/images" --output "data/annotations/nlp_results.json"
```

---

## References

- **spaCy Documentation**: https://spacy.io/
- **German Models**: https://spacy.io/models/de
- **NER Paper**: Honnibal & Montani (2017) "spaCy 2: Natural language understanding with Bloom embeddings"
- **Dependency Parsing**: Universal Dependencies (UD) framework

---

## Future NLP Enhancements

Potential improvements for extended NLP functionality:

1. **Word Embeddings**: Use word2vec/FastText for semantic similarity
2. **BERT-based NER**: Fine-tune transformer models on retail domain
3. **Relation Extraction**: Identify price-product relationships
4. **Text Classification**: Categorize products (coffee, chocolate, dairy, etc.)
5. **Coreference Resolution**: Link pronouns to product mentions
6. **Multi-lingual Support**: Extend to English, French flyers

---

## Summary

The pipeline now uses **genuine NLP techniques** including:
- ✅ Named Entity Recognition (spaCy NER)
- ✅ Part-of-Speech Tagging (POS filtering)
- ✅ Dependency Parsing (syntactic structure)
- ✅ Noun Chunking (phrase extraction)
- ✅ Custom Pattern Matching (linguistic rules)
- ✅ Token-level Features (morphology, lexical properties)

This transforms the project from a **pure OCR + regex system** into a **true NLP application** suitable for an NLP university course.
