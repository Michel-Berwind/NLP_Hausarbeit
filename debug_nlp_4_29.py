"""Debug NLP extraction for the 4.29€ product."""
import spacy
from src.extraction.ner_model import ProductNERExtractor

# The OCR text from 4.29€
text = "N . DAKIS INMU A ESPRESSO m. RISTRETTO LUNGO n CREMA NR CoMPATIBLE S/12 3. ALUMINIUM . CAPSULES Ba W N An c DE A 11030 t ., BARISSIMO - , &R3 FA A"

print("=" * 80)
print("Debugging NLP Extraction for 4.29€ Product")
print("=" * 80)
print(f"\nOCR Text:\n{text}\n")

extractor = ProductNERExtractor()

# Call with debug=True
result = extractor.extract_entities(text)

print("\n" + "=" * 80)
print("Result:")
print("=" * 80)
print(f"Product Name: {result['product_name']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"\nBrands: {result.get('brands', [])}")
print(f"Entities: {[e['text'] for e in result.get('nlp_entities', [])]}")
print(f"Noun Chunks: {result.get('noun_chunks', [])}")
