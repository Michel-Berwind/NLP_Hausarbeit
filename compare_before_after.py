"""
Visual comparison: Before (regex-only) vs After (with NLP)

This script demonstrates the difference between the old regex-based approach
and the new NLP-enhanced extraction.
"""

import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent))

from src.extraction.ner_model import extract_product_entities


def old_regex_approach(text: str) -> dict:
    """Simulate the old regex-only approach."""
    # Just basic cleaning with regex
    text = re.sub(r'[^A-Za-zÄÖÜäöüß0-9 ,./&\-]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Hardcoded brand matching
    known_brands = ["BARISSIMO", "NESCAFE", "JACOBS", "LINDT", "TCHIBO"]
    brands_found = [b for b in known_brands if b in text.upper()]
    
    return {
        "method": "regex",
        "product_name": text[:50],  # Just truncate
        "brands": brands_found,
        "structure": "❌ No structure, just strings",
        "confidence": "❌ No confidence score",
        "nlp_features": "❌ None"
    }


def new_nlp_approach(text: str) -> dict:
    """Use the new NLP-based approach."""
    result = extract_product_entities(text)
    result["method"] = "nlp"
    result["structure"] = "✅ Structured entities"
    result["nlp_features"] = "✅ NER, POS, Parsing"
    return result


def compare(text: str):
    """Compare old vs new approach side by side."""
    print(f"\n{'='*80}")
    print(f"INPUT TEXT: {text}")
    print(f"{'='*80}\n")
    
    # Old approach
    old = old_regex_approach(text)
    print("🔴 OLD APPROACH (Regex-only):")
    print("-" * 80)
    print(f"  Method:        {old['method']}")
    print(f"  Product:       {old['product_name']}")
    print(f"  Brands:        {old['brands']}")
    print(f"  Structure:     {old['structure']}")
    print(f"  Confidence:    {old['confidence']}")
    print(f"  NLP Features:  {old['nlp_features']}")
    
    # New approach
    new = new_nlp_approach(text)
    print("\n🟢 NEW APPROACH (With NLP):")
    print("-" * 80)
    print(f"  Method:        {new['method']}")
    print(f"  Product:       {new['product_name']}")
    print(f"  Brands:        {new['brands']}")
    print(f"  Structure:     {new['structure']}")
    print(f"  Confidence:    {new['confidence']:.2%}")
    print(f"  NLP Features:  {new['nlp_features']}")
    print(f"\n  📊 NLP Analysis:")
    print(f"     Entities:       {len(new['entities'])} found")
    if new['entities']:
        for ent in new['entities']:
            print(f"                     - {ent['text']} [{ent['label']}]")
    print(f"     Noun Chunks:    {new['noun_chunks']}")
    print(f"     Quantities:     {len(new['quantities'])} found")
    if new['quantities']:
        for qty in new['quantities']:
            print(f"                     - {qty['value']} {qty['unit']}")
    print(f"     POS-filtered:   {new['pos_filtered']}")
    
    print("\n" + "="*80)
    print("🎯 IMPROVEMENT:")
    print("="*80)
    print("✅ Structured entity extraction (not just strings)")
    print("✅ Automatic brand detection via NER (no hardcoded list)")
    print("✅ Confidence scores based on linguistic features")
    print("✅ POS tagging removes OCR noise")
    print("✅ Quantity extraction with units")
    print("✅ Multiple entity types (ORG, MISC, PRODUCT)")
    print("✅ Explainable with linguistic annotations")


if __name__ == "__main__":
    samples = [
        "BARISSIMO Espresso Cremoso 500-g-Packung",
        "NESCAFÉ Gold Kaffee 200g Glas",
        "Aus unserem Sortiment TCHIBO Caffè Crema Vollmundig 500g",
        "LINDT Schokolade Vollmilch 100g Tafel",
    ]
    
    print("\n" + "="*80)
    print(" " * 20 + "BEFORE vs AFTER NLP INTEGRATION")
    print("="*80)
    
    for sample in samples:
        compare(sample)
        input("\n[Press Enter for next comparison...]")
