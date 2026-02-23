# NLP-Based Supermarket Flyer Price Extraction

**University Project for Natural Language Processing Course**

This project extracts product information and prices from German supermarket flyers using a combination of Computer Vision (OCR) and **Natural Language Processing (NLP)** techniques.

---

## 🎯 Project Overview

Extract structured product data from retail flyer images:
- **Product names** (using NLP entity extraction)
- **Prices** (using OCR + pattern matching)
- **Brand names** (using NER)
- **Quantities** (using linguistic patterns)
- **Product attributes** (using POS tagging)

### Example Input/Output

**Input**: Flyer image with product boxes

**Output**:
```json
{
  "image": "page_10.png",
  "items": [
    {
      "box": [100, 200, 150, 80],
      "price": "2.99",
      "product": "BARISSIMO Espresso Cremoso",
      "product_nlp": {
        "product_name": "BARISSIMO Espresso Cremoso",
        "brands": ["BARISSIMO", "Espresso"],
        "quantities": [{"value": "500", "unit": "g"}],
        "confidence": 0.95,
        "method": "nlp"
      }
    }
  ]
}
```

---

## 🧠 NLP Techniques Used

This project demonstrates multiple NLP techniques required for university coursework:

### 1. **Named Entity Recognition (NER)**
- spaCy's German language model (`de_core_news_sm`)
- Extracts brands, product names, and entities
- Entity types: ORG, PRODUCT, MISC

### 2. **Part-of-Speech (POS) Tagging**
- Filters OCR noise by keeping nouns, proper nouns, adjectives
- Removes verbs, prepositions (likely OCR errors)

### 3. **Dependency Parsing**
- Extracts compound product names
- Links attributes to products
- Identifies syntactic relationships

### 4. **Noun Chunk Extraction**
- Extracts multi-word product names
- Uses syntactic parsing for phrase boundaries

### 5. **Custom Pattern Matching**
- spaCy Matcher for brand recognition
- Rule-based patterns for product types

### 6. **Token-Level Features**
- Linguistic features: `is_upper`, `is_alpha`, `pos_`
- Morphological analysis for noise filtering

See [NLP_FEATURES.md](NLP_FEATURES.md) for detailed documentation.

---

## 🏗️ Architecture

```
Input Image
    ↓
┌─────────────────────────────────┐
│  Computer Vision (OpenCV)       │
│  - Color-based price box detect │
│  - Morphological operations     │
│  - Contour analysis             │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  OCR (Tesseract)                │
│  - Text extraction              │
│  - Multiple preprocessing       │
│  - Multi-PSM modes              │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  NLP (spaCy)                    │
│  - Named Entity Recognition     │
│  - POS Tagging                  │
│  - Dependency Parsing           │
│  - Entity Extraction            │
└────────────┬────────────────────┘
             ↓
    Structured JSON Output
```

---

## 📦 Installation

### Requirements
- Python 3.8+
- Tesseract OCR
- spaCy German model

### Setup

1. **Clone repository**:
```bash
git clone https://github.com/Michel-Berwind/NLP_Hausarbeit.git
cd NLP_Hausarbeit
```

2. **Create virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download spaCy German model**:
```bash
python -m spacy download de_core_news_sm
```

5. **Install Tesseract OCR** (if not already installed):
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-deu`
   - Mac: `brew install tesseract tesseract-lang`

---

## 🚀 Usage

### Run Full Pipeline

```bash
python -m src.ocr.pipeline.pipeline_runner --input-dir "data/images/aldi" --output "data/annotations/results.json"
```

### Test NLP Features

```bash
python test_nlp_features.py
```

### Process Single Image

```bash
python -m src.ocr.pipeline.pipeline_runner --input-file "data/images/aldi/page10.png" --output "output/page10.json"
```

---

## 📁 Project Structure

```
NLP_Hausarbeit/
├── src/
│   ├── extraction/
│   │   └── ner_model.py          # 🧠 NLP entity extraction (NEW)
│   └── ocr/
│       └── pipeline/
│           ├── pipeline_runner.py     # Main orchestrator
│           ├── image_preprocessing.py # CV preprocessing
│           ├── pricebox_detection.py  # Price box detection
│           ├── ocr_product_text.py    # OCR + NLP integration
│           └── nlp_to_json.py         # Output formatting
├── data/
│   ├── images/          # Input flyer images
│   └── annotations/     # Output JSON files
├── test_nlp_features.py # NLP demonstration script
├── NLP_FEATURES.md      # Detailed NLP documentation
└── requirements.txt     # Python dependencies
```

---

## 📊 NLP Module Details

### `src/extraction/ner_model.py`

The core NLP module implementing:

```python
from src.extraction.ner_model import extract_product_entities

# Extract entities from text
result = extract_product_entities("BARISSIMO Espresso 500g")

# Access NLP results
print(result["product_name"])   # "BARISSIMO Espresso"
print(result["brands"])          # ["BARISSIMO", "Espresso"]
print(result["entities"])        # [{"text": "BARISSIMO", "label": "ORG"}]
print(result["noun_chunks"])     # ["BARISSIMO", "Espresso"]
print(result["quantities"])      # [{"value": "500", "unit": "g"}]
print(result["pos_filtered"])    # POS-filtered text
print(result["confidence"])      # 0.95
```

### Key Classes

- **`ProductNERExtractor`**: Main NLP extraction class
  - Loads spaCy model
  - Configures custom patterns
  - Extracts entities, brands, quantities
  - Calculates confidence scores

---

## 🔬 NLP Evaluation

### Before NLP (Regex-only)
- ❌ Hardcoded brand lists
- ❌ No structure (just strings)
- ❌ No confidence scores
- ❌ Poor noise handling
- ❌ Not academically rigorous

### After NLP Integration
- ✅ Automatic entity recognition
- ✅ Structured entity output
- ✅ Confidence estimation
- ✅ POS-based noise filtering
- ✅ True NLP techniques for university project

---

## 📈 Results

The system now extracts:

| Metric | Value |
|--------|-------|
| NLP Entities Extracted | ✅ Yes (NER) |
| POS Tagging Used | ✅ Yes |
| Dependency Parsing | ✅ Yes |
| Noun Chunks | ✅ Yes |
| Brand Detection | ✅ Automatic (NER) |
| Confidence Scores | ✅ Linguistic features |

---

## 🎓 Academic Justification

This project satisfies NLP course requirements by implementing:

1. **Token-level NLP**: Tokenization, POS tagging, morphology
2. **Syntactic NLP**: Dependency parsing, phrase structure
3. **Semantic NLP**: Named entity recognition, entity typing
4. **Applied NLP**: Information extraction, domain adaptation
5. **Hybrid NLP**: Combining rule-based and ML approaches
6. **Multi-lingual NLP**: German language processing

See [NLP_FEATURES.md](NLP_FEATURES.md) for complete academic documentation.

---

## 🛠️ Technologies

- **Computer Vision**: OpenCV
- **OCR**: Tesseract
- **NLP Framework**: spaCy 3.x
- **Language Model**: de_core_news_sm (German)
- **Language**: Python 3.8+

---

## 📝 Example Output

```json
{
  "image": "KW40_25_page10.png",
  "items": [
    {
      "box": [523, 891, 234, 167],
      "price": "2.99",
      "confidence": 1.0,
      "product": "BARISSIMO Espresso Cremoso",
      "product_nlp": {
        "product_name": "BARISSIMO Espresso Cremoso",
        "ocr_text": "BARISSIMO Espresso Cremoso 500-g-Packung",
        "nlp_entities": [
          {"text": "BARISSIMO", "label": "ORG", "start": 0, "end": 9}
        ],
        "brands": ["BARISSIMO"],
        "quantities": [
          {"value": "500", "unit": "g", "full_text": "500-g"}
        ],
        "noun_chunks": ["BARISSIMO", "Espresso Cremoso"],
        "pos_filtered": "BARISSIMO Espresso Cremoso Packung",
        "confidence": 0.95,
        "method": "nlp"
      }
    }
  ]
}
```

---

## 📚 Documentation

- [NLP_FEATURES.md](NLP_FEATURES.md) - Detailed NLP techniques documentation
- [page_coffee_preprocessing_analysis.md](page_coffee_preprocessing_analysis.md) - Preprocessing details
- [pricebox_detection_summary.md](pricebox_detection_summary.md) - Detection algorithm

---

## 🤝 Contributing

This is a university project. For questions or suggestions, please open an issue.

---

## 📄 License

This project is for academic purposes.

---

## 👤 Author

**Michel Berwind**
- GitHub: [@Michel-Berwind](https://github.com/Michel-Berwind)

---

## 🙏 Acknowledgments

- **spaCy**: Modern NLP library
- **Explosion AI**: German language models
- **Tesseract OCR**: Open-source OCR engine
- **OpenCV**: Computer vision library

---

## 📖 References

1. Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
2. Universal Dependencies: https://universaldependencies.org/
3. spaCy Documentation: https://spacy.io/
4. German NLP: https://spacy.io/models/de
