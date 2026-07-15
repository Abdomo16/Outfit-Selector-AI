import os
import sys
from pathlib import Path
from PIL import Image
import json

# Ensure we can import from the root directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attribute_extractor.attribute_extractor import AttributeExtractor

def test_phase4():
    sample_img_path = "sample_test.jpg"
    if not os.path.exists(sample_img_path):
        print(f"Cannot find {sample_img_path}")
        return

    print("Initializing CLIP Attribute Extractor...")
    extractor = AttributeExtractor()
    
    print(f"Loading image {sample_img_path}...")
    img = Image.open(sample_img_path).convert("RGB")
    
    print("Running Zero-Shot Prediction...")
    result = extractor.predict(img)
    
    print("\n--- Extracted Attributes ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_phase4()
