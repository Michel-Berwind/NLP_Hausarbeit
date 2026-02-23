"""Check POS tags for specific words"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.extraction.ner_model import ProductNERExtractor

text = "FERREX 40 Akku-Laubsauger Blasen, saugen, zerkleinern"

extractor = ProductNERExtractor()
doc = extractor.nlp(text)

print("POS Tags:")
for token in doc:
    print(f"  {token.text:20s} -> {token.pos_:10s}")
