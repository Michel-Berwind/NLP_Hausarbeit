"""Quick test: Run pipeline on 2 test images."""
from pathlib import Path
from src.pipeline import process_image
from src.preprocessing.image_preprocessing import configure_tesseract
from src.utils.json_utils import save_results

def main():
    configure_tesseract()
    
    test_images = [
        "data/images/aldi/aldi_page01.png"
    ]
    
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path("debug_test")
    
    for img_path_str in test_images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"❌ {img_path} nicht gefunden")
            continue
            
        print(f"\n{'='*60}")
        print(f"Verarbeite: {img_path.name}")
        print('='*60)
        
        try:
            result = process_image(img_path, debug_dir)
            output_file = output_dir / f"{img_path.stem}.json"
            save_results([result], output_file)
            
            num_items = len(result.get("items", []))
            print(f"\n✅ Erfolgreich! {num_items} Angebote gefunden")
            print(f"   Gespeichert in: {output_file}")
            
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
