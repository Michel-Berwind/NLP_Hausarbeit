"""Test Box 4 extraction"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.extraction.ner_model import ProductNERExtractor

text = "FERREX Asche- und Grobschmutzsauger Filterreinigungsfunktion, Doppel- filtersystem, ca. 20-I-Metallbehälter, Sicherheitsschalter, Jahre Garantie, je Stück Wir bitten um Beachtun dass diese Aktionsartike in begrenzter Anzahl zur Verfügung stehen. Sie kö Aktionsbeginn ausverkauft sein. Alle Artikel ohne"

extractor = ProductNERExtractor()
doc = extractor.nlp(text)
product = extractor._extract_product_name(doc, debug=True)

print(f"\nFinal result: {product if product else '(REJECTED)'}")
