"""
Demo script to showcase NLP features for product extraction.

This demonstrates the NLP techniques used in the pipeline:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Dependency parsing
- Entity extraction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.extraction.ner_model import extract_product_entities


def demo_nlp_extraction():
    """Demonstrate NLP-based entity extraction on sample texts."""
    
    # Sample OCR texts from supermarket flyers (with typical OCR noise)
    sample_texts = [
        "BARISSIMO Espresso Cremoso 500-g-Packung",
        "NESCAFÉ Gold Kaffee 200g Glas",
        "Jacobs Krönung Kaffeekapseln 20 Kapseln Aluminium",
        "LINDT Schokolade Vollmilch 100g Tafel",
        "Aus unserem Sortiment TCHIBO Caffè Crema Vollmundig 500g",
        "Barista Edition Premium Espresso Bohnen 1kg",
    ]
    
    print("=" * 80)
    print("NLP-BASED PRODUCT ENTITY EXTRACTION DEMO")
    print("=" * 80)
    print()
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*80}")
        print(f"Sample {i}: {text}")
        print(f"{'='*80}")
        
        # Extract entities using NLP
        result = extract_product_entities(text)
        
        print(f"\n📦 Product Name (NLP-extracted):")
        print(f"   → {result['product_name']}")
        
        print(f"\n🏢 Brands detected:")
        if result['brands']:
            for brand in result['brands']:
                print(f"   → {brand}")
        else:
            print("   → (none)")
        
        print(f"\n📊 Named Entities (spaCy NER):")
        if result['entities']:
            for ent in result['entities']:
                print(f"   → {ent['text']} [{ent['label']}]")
        else:
            print("   → (none)")
        
        print(f"\n📝 Noun Chunks (multi-word product names):")
        if result['noun_chunks']:
            for chunk in result['noun_chunks']:
                print(f"   → {chunk}")
        else:
            print("   → (none)")
        
        print(f"\n⚖️ Quantities extracted:")
        if result['quantities']:
            for qty in result['quantities']:
                print(f"   → {qty['value']} {qty['unit']} ({qty['full_text']})")
        else:
            print("   → (none)")
        
        print(f"\n🏷️ Attributes (adjectives):")
        if result['attributes']:
            for attr in result['attributes']:
                print(f"   → {attr}")
        else:
            print("   → (none)")
        
        print(f"\n🔍 POS-filtered text (nouns/proper nouns only):")
        print(f"   → {result['pos_filtered']}")
        
        print(f"\n✅ Confidence: {result['confidence']:.2%}")
    
    print(f"\n{'='*80}")
    print("NLP TECHNIQUES DEMONSTRATED:")
    print("="*80)
    print("✓ Named Entity Recognition (NER) - spaCy's de_core_news_sm model")
    print("✓ Part-of-Speech (POS) tagging - filtering noise by keeping nouns/adjectives")
    print("✓ Noun chunk extraction - multi-word product names")
    print("✓ Dependency parsing - extracting relationships between words")
    print("✓ Custom entity patterns - brand recognition with Matcher")
    print("✓ Linguistic features - token-level analysis (is_upper, is_alpha, etc.)")
    print("="*80)


if __name__ == "__main__":
    demo_nlp_extraction()
