"""
Run pipeline on all images and save predictions.
"""
from pathlib import Path
from src.pipeline import process_image
from src.preprocessing.image_preprocessing import configure_tesseract
from src.utils.json_utils import save_results

def main():
    configure_tesseract()

    # Directories
    aldi_images = Path("data/images/aldi")
    output_dir = Path("data/predictions")
    debug_dir = Path("debug_all")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Aldi images
    aldi_files = sorted(aldi_images.glob("*.png"))
    print(f"Processing {len(aldi_files)} Aldi images...")
    
    for i, img_path in enumerate(aldi_files, 1):
        page_id = img_path.stem
        output_path = output_dir / f"{page_id}.json"
        
        print(f"  [{i}/{len(aldi_files)}] {page_id}...", end="", flush=True)
        
        try:
            result = process_image(img_path, debug_dir)
            save_results([result], output_path)
            print(" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
    
    print(f"\nDone! Predictions saved to {output_dir}/")

if __name__ == "__main__":
    main()
