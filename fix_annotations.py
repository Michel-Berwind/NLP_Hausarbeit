"""Fix annotation files: remove brands and fix encoding."""
import json
from pathlib import Path
import re

def remove_brand_from_product(product: str) -> str:
    """Remove brand names from product string."""
    # Common brand patterns
    brands = [
        r'^RIDE\+GO\s+',
        r'^RIDE\s+GO\s+',
        r'^Parkside\s+',
        r'^ESMARA\s+',
        r'^LIVERGY\s+',
        r'^CRIVIT\s+',
        r'^SILVERCREST\s+',
        r'^FLORABEST\s+',
        r'^POWERFIX\s+',
        r'^MERADISO\s+',
        r'^LUPILU\s+',
        r'^TOPMOVE\s+',
    ]
    
    for brand_pattern in brands:
        product = re.sub(brand_pattern, '', product, flags=re.IGNORECASE)
    
    return product.strip()

def fix_annotation_file(file_path: Path):
    """Fix single annotation file."""
    # Read with proper encoding
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Remove brands from products
    modified = False
    for offer in data.get('offers', []):
        original = offer['product']
        cleaned = remove_brand_from_product(original)
        if cleaned != original:
            offer['product'] = cleaned
            modified = True
    
    # Write back without BOM
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return modified

def main():
    annotations_dir = Path('data/annotations')
    
    print("Fixing annotation files...")
    fixed_count = 0
    
    for json_file in sorted(annotations_dir.glob('*.json')):
        try:
            modified = fix_annotation_file(json_file)
            if modified:
                print(f"  ✓ {json_file.name} (brands removed)")
                fixed_count += 1
            else:
                print(f"  ✓ {json_file.name} (encoding fixed)")
        except Exception as e:
            print(f"  ✗ {json_file.name}: {e}")
    
    print(f"\n✅ Fixed {len(list(annotations_dir.glob('*.json')))} files ({fixed_count} had brands removed)")

if __name__ == '__main__':
    main()
