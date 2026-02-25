"""Test pipeline improvements on 5 Aldi pages."""
from pathlib import Path
import traceback
from src.pipeline import process_image
from src.preprocessing.image_preprocessing import configure_tesseract
from src.utils.json_utils import save_results

def main():
    configure_tesseract()
    
    aldi_dir = Path("data/images/aldi")
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None  # Disable debug output to prevent hanging
    
    # Test on first 5 pages only
    test_pages = [f"aldi_page{i:02d}.png" for i in range(1, 6)]
    
    print(f"{'='*60}")
    print(f"TESTING PIPELINE IMPROVEMENTS - 5 Pages")
    print(f"{'='*60}\n")
    
    for i, page_name in enumerate(test_pages, 1):
        img_path = aldi_dir / page_name
        if not img_path.exists():
            print(f"  ⚠ {page_name} not found, skipping")
            continue
            
        print(f"[{i}/5] {page_name}...")
        
        try:
            result = process_image(img_path, debug_dir)
            output_file = output_dir / f"{img_path.stem}.json"
            save_results([result], output_file)
            
            num_items = len(result.get("items", []))
            print(f"  ✅ {num_items} offers found\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Done! Predictions in: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
