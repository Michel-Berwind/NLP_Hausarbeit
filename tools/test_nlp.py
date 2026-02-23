"""Test NLP extraction improvements."""
from src.extraction.ner_model import ProductNERExtractor

extractor = ProductNERExtractor()

test_cases = [
    "FERREX 20V Akku-Heckenschere",
    "TOPCRAFT Arbeitshandschuhe",  
    "WORKZONE Arbeitsleuchte mit Bewegungsmelder",
    "Weitere Farbe im Unterschied zu unserem ständig vorhandenen Sortiment",
    "FERREX 40V Akku-Laubsauger Blasen, säugen, zerkleinern",
    "a FERREX 40 Akku-Laubsauger Blasen, säugen, zerkleinern, Blasge- schwindigkeit max. 210 km/h, Drehzahl ca. 8000-13000 U/min, Saugleistung max. 570 /h, Füllmenge Fangsack ca. 45 1, Jahre Garantie, je Stück"
]

print("\nNLP Extraction Tests:")
print("=" * 80)
for text in test_cases:
    result = extractor.extract_entities(text)
    product = result['product_name']
    print(f"\nInput:   {text[:70]}")
    print(f"Product: {product if product else '(REJECTED)'}")
