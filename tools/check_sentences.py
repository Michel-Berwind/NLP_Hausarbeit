"""Check how spaCy is segmenting sentences"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.extraction.ner_model import ProductNERExtractor

text = "a FERREX 40 Akku-Laubsauger Blasen, saugen, zerkleinern, Blasge- schwindigkeit max. 210 km/h, Drehzahl ca. 8000-13000 U/min, Saugleistung max. 570 /h, Fullmenge Fangsack ca. 45 1, Jahre Garantie, je Stuck"

extractor = ProductNERExtractor()
doc = extractor.nlp(text)

print("Sentence Segmentation:")
print("=" * 80)
for i, sent in enumerate(doc.sents):
    print(f"Sentence {i}: [{sent.text}]")
    print(f"  Tokens: {[t.text for t in sent]}")
    print()
