# Automatische Extraktion von Produkt- und Preisinformationen aus Discounter-Prospekten

**Masterarbeit - Natural Language Processing**

Dieses Repository enthält eine klassische CV + OCR + NLP Pipeline zur automatischen Extraktion von Produktinformationen und Preisen aus deutschen Discounter-Prospekten (Aldi).

---

## 🎯 Projektübersicht

Die Pipeline extrahiert strukturierte Produktdaten aus Prospekt-Bildern:
- **Produktnamen** (mittels NLP-Entitätserkennung)
- **Preise** (mittels OCR + Pattern Matching)
- **Markennamen** (mittels Named Entity Recognition)
- **Mengenangaben** (mittels linguistischer Patterns)
- **Produktattribute** (mittels POS-Tagging)

### Beispiel Input/Output

**Input**: Prospekt-Bild mit Produktboxen

**Output** (JSON):
```json
{
  "image": "KW40_25_ebeae15a-90e5-4975-a5cd-ddd640c8c977_page10.png",
  "items": [
    {
      "box": [100, 200, 150, 80],
      "price": "2.99",
      "product": "BARISSIMO Espresso Cremoso",
      "product_nlp": {
        "product_name": "BARISSIMO Espresso Cremoso",
        "brands": ["BARISSIMO"],
        "quantities": [{"value": "500", "unit": "g"}],
        "confidence": 0.95,
        "method": "nlp"
      }
    }
  ]
}
```

---

## 🏗️ Pipeline-Architektur

```
Input: PDF/PNG
    ↓
┌─────────────────────────────────┐
│  1. Bildvorverarbeitung         │
│     - Farbraumkonvertierung     │
│     - Morphologische Operationen│
│     - Kontrastanpassung         │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  2. Preisbox-Detektion (CV)     │
│     - Farbbasierte Segmentierung│
│     - Konturanalyse (OpenCV)    │
│     - Heuristische Filterung    │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  3. OCR (Tesseract)             │
│     - Textextraktion            │
│     - Multi-PSM-Modi            │
│     - Noise Filtering           │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  4. NLP-Verarbeitung (spaCy)    │
│     - Named Entity Recognition  │
│     - POS-Tagging               │
│     - Dependency Parsing        │
│     - Pattern Matching          │
└────────────┬────────────────────┘
             ↓
  Strukturiertes JSON-Output
```

---

## 📦 Installation & Setup

### Voraussetzungen
- Python 3.8+
- Tesseract OCR
- Git

### Schritt-für-Schritt Anleitung

**1. Repository klonen:**
```bash
git clone https://github.com/Michel-Berwind/NLP_Hausarbeit.git
cd NLP_Hausarbeit
```

**2. Virtuelle Umgebung erstellen:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# oder
source .venv/bin/activate  # Linux/Mac
```

**3. Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

**4. spaCy German Model herunterladen:**
```bash
python -m spacy download de_core_news_sm
```

**5. Tesseract OCR installieren:**
- **Windows:** https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt-get install tesseract-ocr tesseract-ocr-deu`
- **Mac:** `brew install tesseract tesseract-lang`

**6. Poppler installieren (für PDF-Konvertierung):**
- **Windows:** https://github.com/oschwartz10612/poppler-windows/releases/ (poppler-xx.xx.x/Library/bin/ zum PATH hinzufügen)
- **Linux:** `sudo apt-get install poppler-utils`
- **Mac:** `brew install poppler`

---

## 📄 PDF zu PNG konvertieren

Die Pipeline arbeitet mit PNG-Bildern. Konvertieren Sie Ihre PDFs zuerst:

### Einzelne PDF-Datei

```bash
python -m src.preprocessing.pdf_to_images --input-file "data/raw/aldi/prospekt.pdf" \
                                          --output-dir "data/images/aldi" \
                                          --dpi 300
```

### Gesamtes Verzeichnis

```bash
# Alle Aldi-PDFs konvertieren
python -m src.preprocessing.pdf_to_images --input-dir "data/raw/aldi" \
                                          --output-dir "data/images/aldi" \
                                          --dpi 300


```

**Hinweis:** DPI=300 ist empfohlen für beste OCR-Qualität. Jede PDF-Seite wird als separate PNG-Datei gespeichert (z.B. `prospekt_page1.png`, `prospekt_page2.png`, ...).

---

## 🚀 Pipeline ausführen

### Einzelnes Bild verarbeiten

```bash
python -m src.pipeline --input-file "data/images/aldi/KW40_25_page10.png" \
                       --output "results/page10.json"
```

### Gesamtes Verzeichnis verarbeiten

```bash
python -m src.pipeline --input-dir "data/images/aldi" \
                       --output "results/aldi_results.json" \
                       --pattern "*.png"
```

### Parameter

| Parameter | Beschreibung | Standard |
|-----------|--------------|----------|
| `--input-file` | Einzelne Bilddatei | - |
| `--input-dir` | Verzeichnis mit Bildern | - |
| `--output` | Ausgabepfad für JSON | `results/output.json` |
| `--pattern` | Dateimuster (bei --input-dir) | `*.png` |
| `--debug-root` | Optional: Debug-Ausgaben | - |

---

## 📊 Evaluation

### Evaluation ausführen

```bash
python -m src.evaluation.evaluate --predictions "results/" \
                                  --annotations "data/annotations/" \
                                  --iou-threshold 0.5 \
                                  --output "evaluation_results.json"
```

### Metriken

Die Evaluation berechnet:
- **Precision**: Anteil korrekt erkannter Boxen an allen erkannten
- **Recall**: Anteil gefundener Ground-Truth-Boxen
- **F1-Score**: Harmonisches Mittel von Precision und Recall
- **IoU-basiertes Matching**: Bounding-Box-Überlappung (Standard: 50%)

---

## 📁 Projektstruktur

```
NLP_HAUSARBEIT/
│
├── data/
│   ├── raw/              # Original PDFs
│   │   ├── aldi/
│   │
│   ├── images/           # PDF → PNG Konvertierungen
│   │   ├── aldi/
│   │
│   ├── ocr_text/         # OCR Rohausgaben (optional)
│   │   ├── aldi/
│   │
│   └── annotations/      # Ground Truth für Evaluation
│       └── *.json
│
├── src/
│   ├── preprocessing/
│   │   └── image_preprocessing.py    # Bildvorverarbeitung
│   ├── detection/
│   │   ├── pricebox_detection.py     # Preisbox-Detektion (OpenCV)
│   │   └── price_region_detection.py # Region-basierte Detektion
│   ├── ocr/
│   │   └── ocr_product_text.py       # Tesseract Wrapper
│   ├── nlp/
│   │   ├── ner_model.py              # spaCy NER für Produkte
│   │   └── rule_based.py             # Regelbasierte Extraktion
│   ├── evaluation/
│   │   └── evaluate.py               # Precision/Recall/F1
│   ├── utils/
│   │   ├── json_utils.py             # JSON Serialisierung
│   │   └── text_quality_analysis.py  # OCR-Qualitätsanalyse
│   └── pipeline.py                   # 🔥 HAUPTPIPELINE
│
├── results/              # Pipeline-Ausgaben (JSON)
│   └── .gitkeep
│
├── notebooks/            # Explorative Analyse
│   └── exploration.ipynb
│
├── .gitignore            # Git-Konfiguration
├── requirements.txt      # Python-Abhängigkeiten
├── README.md             # Diese Datei
└── NLP_FEATURES.md       # Detaillierte NLP-Dokumentation
```

---

## 🧠 NLP-Techniken

Dieses Projekt demonstriert folgende NLP-Methoden:

### 1. **Named Entity Recognition (NER)**
- spaCy German Model (`de_core_news_sm`)
- Extraktion von Marken, Produktnamen, Organisationen
- Entity-Typen: ORG, PRODUCT, MISC

### 2. **Part-of-Speech (POS) Tagging**
- Filterung von OCR-Rauschen durch linguistische Analyse
- Behalten: Nomen, Eigennamen, Adjektive
- Entfernen: Verben, Präpositionen (wahrscheinlich OCR-Fehler)

### 3. **Dependency Parsing**
- Extraktion von Kompositum-Produktnamen
- Verknüpfung von Attributen zu Produkten
- Analyse syntaktischer Relationen

### 4. **Noun Chunk Extraction**
- Extraktion mehrwortiger Produktnamen
- Nutzung syntaktischer Phrasengrenzen

### 5. **Custom Pattern Matching**
- spaCy Matcher für Markenerkennung
- Regelbasierte Patterns für Produkttypen und Mengen

### 6. **Token-Level Features**
- Linguistische Features: `is_upper`, `is_alpha`, `pos_`
- Morphologische Analyse zur Rauschfilterung

**Detaillierte Dokumentation:** Siehe [NLP_FEATURES.md](NLP_FEATURES.md)

---

## 📈 Reproduzierbarkeit

### Experiment wiederholen

1. **Daten vorbereiten:** PDFs in `data/raw/aldi/` ablegen

2. **PDFs zu PNGs konvertieren:**
   ```bash
   # Aldi-Prospekte
   python -m src.preprocessing.pdf_to_images --input-dir data/raw/aldi \
                                             --output-dir data/images/aldi
   

   ```

3. **Pipeline ausführen:** 
   ```bash
   python -m src.pipeline --input-dir data/images/aldi --output results/aldi.json
   ```

4. **Evaluation:** 
   ```bash
   python -m src.evaluation.evaluate --predictions results/ --annotations data/annotations/
   ```

### Erwartete Ergebnisse

Die Pipeline wurde auf ALDI-Prospekten getestet.

---

## 🔧 Technologie-Stack

| Komponente | Technologie | Zweck |
|------------|------------|-------|
| **Computer Vision** | OpenCV | Preisbox-Detektion |
| **OCR** | Tesseract | Textextraktion |
| **NLP** | spaCy (`de_core_news_sm`) | Entitätserkennung |
| **Preprocessing** | NumPy, OpenCV | Bildverarbeitung |
| **Evaluation** | Custom Python | Metriken-Berechnung |

---

## 📝 Beispiel-Output

```json
{
  "image": "KW40_25_page10.png",
  "items": [
    {
      "box": [523, 891, 234, 167],
      "price": "2.99",
      "product": "BARISSIMO Espresso Cremoso",
      "product_nlp": {
        "product_name": "BARISSIMO Espresso Cremoso",
        "ocr_text": "BARISSIMO Espresso Cremoso 500-g-Packung",
        "nlp_entities": [
          {"text": "BARISSIMO", "label": "ORG"}
        ],
        "brands": ["BARISSIMO"],
        "quantities": [{"value": "500", "unit": "g"}],
        "noun_chunks": ["BARISSIMO", "Espresso Cremoso"],
        "confidence": 0.95,
        "method": "nlp"
      }
    }
  ]
}
```

---

## 📄 Lizenz

Dieses Projekt wurde für akademische Zwecke erstellt (Masterarbeit).

---

## 👤 Autor

Michel Berwind  
Masterarbeit - Natural Language Processing  
2026
