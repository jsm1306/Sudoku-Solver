import cv2
import numpy as np
from main import auto_scan_sudoku, visualize_auto_scan_steps

# Test the auto-scan on sample images
test_images = ['photo_2025-09-06_12-20-25.jpg', 'photo_2025-09-07_12-12-27.jpg', 'sudoku_3.jpg', 'sudoku_4.jpg']

for img_path in test_images:
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load {img_path}")
            continue

        print(f"\nTesting {img_path}...")
        corrected_image, scan_success, debug_info = auto_scan_sudoku(image)

        if scan_success:
            print(f"✅ Success: Grid detected and corrected for {img_path}")
            # Save the corrected image for inspection
            output_path = f"corrected_{img_path}"
            cv2.imwrite(output_path, corrected_image)
            print(f"Saved corrected image to {output_path}")
        else:
            print(f"❌ Failed: {debug_info.get('error', 'Unknown error')} for {img_path}")

    except Exception as e:
        print(f"Error testing {img_path}: {e}")

print("\nTest completed.")
