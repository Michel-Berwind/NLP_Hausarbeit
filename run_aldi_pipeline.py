"""Run pipeline on all 25 Aldi images."""
from pathlib import Path
from src.pipeline import process_image
from src.preprocessing.image_preprocessing import configure_tesseract
from src.utils.json_utils import save_results

def main():
    configure_tesseract()
    
    aldi_dir = Path("data/images/aldi")
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path("debug_aldi")
    
    aldi_images = sorted(aldi_dir.glob("aldi_page*.png"))
    
    print(f"{'='*60}")
    print(f"ALDI PIPELINE - {len(aldi_images)} Seiten")
    print(f"{'='*60}\n")
    
    for i, img_path in enumerate(aldi_images, 1):
        print(f"[{i}/{len(aldi_images)}] {img_path.name}...")
        
        try:
            result = process_image(img_path, debug_dir)
            output_file = output_dir / f"{img_path.stem}.json"
            save_results([result], output_file)
            
            num_items = len(result.get("items", []))
            print(f"  ✅ {num_items} Angebote gefunden\n")
            
        except KeyboardInterrupt:
            print("\n❌ Abgebrochen durch Nutzer")
            break
        except Exception as e:
            print(f"  ❌ Fehler: {e}\n")
    
    print(f"\n{'='*60}")
    print(f"Fertig! Predictions in: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
