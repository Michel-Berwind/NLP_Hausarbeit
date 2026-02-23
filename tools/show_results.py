"""Quick script to display extraction results."""
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_results.py <json_file>")
        return
    
    with open(sys.argv[1], encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Results from: {data[0]['image']}")
    print(f"{'='*80}\n")
    
    for i, item in enumerate(data[0]['items'], 1):
        price = item['price']
        product_name = item['product_nlp']['product_name']
        full_text = item['product_nlp']['ocr_text']
        method = item['product_nlp']['method']
        conf = item['product_nlp']['confidence']
        
        status = "✓" if product_name and len(product_name) > 5 else "✗"
        
        print(f"{status} Box {i} | €{price:>6} | [{method:>8}] | conf={conf:.2f}")
        print(f"   Identified Product: {product_name if product_name else '(REJECTED)'}")
        if full_text and full_text != product_name:
            # Truncate very long text for display
            display_text = full_text if len(full_text) <= 100 else full_text[:97] + "..."
            print(f"   Full OCR Text:      {display_text}")
        print()

if __name__ == "__main__":
    main()
