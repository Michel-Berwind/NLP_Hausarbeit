"""
PDF to Image Conversion Tool

Converts PDF files to individual PNG images, one per page.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    fmt: str = "png",
    show_progress: bool = True,
) -> list[Path]:
    """
    Convert a PDF file to individual images.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save output images
        dpi: Resolution in dots per inch (default: 300)
        fmt: Output image format (default: "png")
        show_progress: Show detailed progress information (default: True)
    
    Returns:
        List of created image file paths
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image not installed. Install with: pip install pdf2image\n"
            "Also requires poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📄 Konvertiere: {pdf_path.name}")
    print(f"📁 Ausgabe: {output_dir}")
    print(f"🎨 DPI: {dpi}")
    print(f"{'='*70}\n")
    
    # Get page count first (without converting)
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(pdf_path))
        page_count = info.get("Pages", 0)
        if page_count > 0:
            print(f"📊 Anzahl Seiten: {page_count}")
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"📦 Dateigröße: {file_size_mb:.2f} MB")
            print(f"⏱️  Geschätzte Zeit: ~{page_count * 2} Sekunden\n")
    except:
        page_count = 0
        print("⚠️  Seitenzahl konnte nicht ermittelt werden - konvertiere alle Seiten\n")
    
    # Convert page by page with real-time progress
    output_paths = []
    base_name = pdf_path.stem
    start_time = time.time()
    
    print("🔄 Konvertiere Seite für Seite...\n")
    
    if page_count > 0:
        # We know the page count - convert page by page
        for page_num in range(1, page_count + 1):
            page_start = time.time()
            
            # Convert single page
            try:
                images = convert_from_path(
                    str(pdf_path), 
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num
                )
                
                if images:
                    # Save the page
                    output_path = output_dir / f"{base_name}_page{page_num}.{fmt}"
                    images[0].save(str(output_path), fmt.upper())
                    output_paths.append(output_path)
                    
                    # Progress display
                    page_time = time.time() - page_start
                    total_elapsed = time.time() - start_time
                    
                    if show_progress:
                        bar_width = 40
                        progress = page_num / page_count
                        filled = int(bar_width * progress)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        
                        # ETA calculation
                        avg_time_per_page = total_elapsed / page_num
                        remaining_pages = page_count - page_num
                        eta_seconds = avg_time_per_page * remaining_pages
                        
                        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds >= 60 else f"{int(eta_seconds)}s"
                        
                        print(f"\r[{bar}] {page_num}/{page_count} | {progress*100:5.1f}% | {page_time:.1f}s/Seite | ETA: {eta_str}  ", end='', flush=True)
            
            except Exception as e:
                print(f"\n⚠️  Fehler bei Seite {page_num}: {e}")
                continue
        
        print("\n")  # New line after progress
    
    else:
        # Page count unknown - convert all at once (fallback)
        print("⚠️  Konvertiere alle Seiten auf einmal (kein Fortschrittsbalken)...")
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)
            print(f"✓ {len(images)} Seiten geladen, speichere jetzt...\n")
            
            for i, image in enumerate(images, start=1):
                output_path = output_dir / f"{base_name}_page{i}.{fmt}"
                image.save(str(output_path), fmt.upper())
                output_paths.append(output_path)
                print(f"\r💾 Speichere Seite {i}/{len(images)}...", end='', flush=True)
            
            print("\n")
        except Exception as e:
            print(f"\n❌ Fehler beim Konvertieren: {e}")
            raise
    
    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"✅ Fertig! {len(output_paths)} Seiten in {total_time:.1f}s konvertiert")
    if len(output_paths) > 0:
        print(f"⚡ Durchschnitt: {total_time/len(output_paths):.1f}s pro Seite")
    print(f"{'='*70}\n")
    
    return output_paths


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.pdf",
    dpi: int = 300,
) -> None:
    """
    Convert all PDF files in a directory to images.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output images
        pattern: Glob pattern for PDF files (default: "*.pdf")
        dpi: Resolution in dots per inch (default: 300)
    """
    pdf_files = sorted(input_dir.glob(pattern))
    
    if not pdf_files:
        print(f"❌ Keine PDF-Dateien in {input_dir} gefunden (Pattern: '{pattern}')")
        return
    
    print(f"\n{'#'*70}")
    print(f"🚀 PDF zu PNG Konvertierung")
    print(f"{'#'*70}")
    print(f"📂 Eingabe: {input_dir}")
    print(f"📁 Ausgabe: {output_dir}")
    print(f"📄 Gefundene PDFs: {len(pdf_files)}")
    print(f"🎨 DPI: {dpi}")
    
    # Show file list
    total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    print(f"💾 Gesamtgröße: {total_size_mb:.2f} MB")
    print(f"\nDateien:")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  {i}. {pdf.name} ({size_mb:.2f} MB)")
    print(f"{'#'*70}\n")
    
    # Convert each PDF
    total_pages = 0
    overall_start = time.time()
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📌 PDF {idx}/{len(pdf_files)}")
        output_paths = convert_pdf_to_images(pdf_path, output_dir, dpi=dpi)
        total_pages += len(output_paths)
    
    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"🎉 ALLE KONVERTIERUNGEN ABGESCHLOSSEN!")
    print(f"{'='*70}")
    print(f"✅ {len(pdf_files)} PDF(s) → {total_pages} PNG(s)")
    print(f"⏱️  Gesamtzeit: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"⚡ Durchschnitt: {total_time/total_pages:.1f}s pro Seite")
    print(f"📁 Ausgabe: {output_dir}")
    print(f"{'='*70}\n")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF files to individual page images"
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a single PDF file"
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing PDF files"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output images"
    )
    
    # Conversion options
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI (default: 300)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern for PDF files when using --input-dir (default: *.pdf)"
    )
    
    args = parser.parse_args(argv)
    
    if args.input_file:
        # Convert single file
        convert_pdf_to_images(
            args.input_file,
            args.output_dir,
            dpi=args.dpi
        )
    else:
        # Convert directory
        convert_directory(
            args.input_dir,
            args.output_dir,
            pattern=args.pattern,
            dpi=args.dpi
        )


if __name__ == "__main__":
    main()
