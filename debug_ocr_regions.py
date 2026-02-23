"""Manually run OCR on the extracted regions."""
import cv2
import pytesseract

regions = [
    "debug_region_0_LEFT.png",
    "debug_region_1_ABOVE.png",
    "debug_region_2_LEFT-ABOVE.png"
]

for region_file in regions:
    print(f"\n{'=' * 80}")
    print(f"Region: {region_file}")
    print('=' * 80)
    
    img = cv2.imread(region_file)
    if img is None:
        print("ERROR: Could not load image")
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different PSM modes
    configs = [
        ("PSM 7 (single line)", "--oem 3 --psm 7"),
        ("PSM 6 (uniform block)", "--oem 3 --psm 6"),
        ("PSM 4 (single column)", "--oem 3 --psm 4"),
    ]
    
    for name, config in configs:
        try:
            text = pytesseract.image_to_string(gray, lang="deu", config=config)
            if text.strip():
                print(f"\n{name}:")
                print(text.strip())
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
