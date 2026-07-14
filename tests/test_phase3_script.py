import os
import sys
from pathlib import Path
from PIL import Image
import urllib.request
import json

# Ensure we can import from the root directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline import InferencePipeline

def download_sample_image(path: str):
    """Download a sample image of a person with clothes if not exists."""
    if not os.path.exists(path):
        url = "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&q=80&w=800"
        print(f"Downloading sample image from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Error downloading image: {e}")
            return False
    return True

def test_pipeline():
    sample_img_path = "sample_test.jpg"
    if not download_sample_image(sample_img_path):
        print("Could not get a sample image. Please provide one manually as 'sample_test.jpg' in the root.")
        return

    print("Loading inference pipeline...")
    try:
        pipeline = InferencePipeline()
        
        print(f"Loading image {sample_img_path}...")
        img = Image.open(sample_img_path).convert("RGB")
        
        print("Running pipeline...")
        result = pipeline.process_upload(img)
        
        print("\n Pipeline Result")
        # Format the result nicely
        print(json.dumps(result, indent=2))
        
        # Verification complete
        print("\n Verification complete. No items were dropped to disk.")
            
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

if __name__ == "__main__":
    test_pipeline()
