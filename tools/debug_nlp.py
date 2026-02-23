"""Debug script to see what NLP is doing with actual OCR text"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.extraction.ner_model import ProductNERExtractor

# Real OCR texts from page 10
test_texts = [
    ("Box 1", "a FERREX 40 Akku-Laubsauger Blasen, säugen, zerkleinern, Blasge- schwindigkeit max. 210 km/h, Drehzahl ca. 8000-13000 U/min, Saugleistung max. 570 /h, Füllmenge Fangsack ca. 45 1, Jahre Garantie, je Stück"),
    ("Box 2", "zerkleinern, Blasge- - ax. 210 km/h, Drehzahl FERREX U/mm, Sauglenstungo 20 Akku-Heckenschere Füllmenge Fangsack ca. rantie, je Stück Zwei-Hand-Sicherheitsschalter, Aluminium-Klinge, Schnittlänge ca. 510 mm, Schnittkapazität max. 15 mm, integr. Handschutz, Maße ca. 90 18 22 cm, Jahre Garantie, je Stück"),
    ("Box 3", "WORKZONE Arbeitsleuchte mit Bewegungsmelder Helligkeitsmodi, 6000- 7500 K, 2x 2600 mAh, Maße ca. 9 x 9 x 15,5 cm, 3 Jahre Garantie, je Stück"),
    ("Box 4", "FERREX Asche- und Grobschmutzsauger Filterreinigungsfunktion, Doppel- filtersystem, ca. 20-I-Metallbehälter, Sicherheitsschalter, Jahre Garantie, je Stück Wir bitten um Beachtun dass diese Aktionsartike in begrenzter Anzahl zur Verfügung stehen. Sie kö Aktionsbeginn ausverkauft sein. Alle Artikel ohne"),
    ("Box 5", "Weitere Farbe im Unterschied zu unserem ständig vorhandenen Sortiment nur nen daher schon am Vormittag des ersten Aktionstages kurz nach Gutes ekoration"),
    ("Box 6", "P Laubsauger rkleinern, Blasge- 210 km/h, Drehzahl U/min, Saugleistung illmenge Fangsack ca. ä"),
]

extractor = ProductNERExtractor()

print("NLP Extraction Debug:")
print("=" * 80)
for box_name, text in test_texts:
    print(f"\n{box_name}:")
    print(f"  Input ({len(text)} chars): {text[:100]}...")
    doc = extractor.nlp(text)
    product = extractor._extract_product_name(doc, debug=True)
    if product:
        print(f"  OK Extracted: {product}")
    else:
        print(f"  X REJECTED")
    print()
