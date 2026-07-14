import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

class Segmenter:
    def __init__(self, model_path="weights/segmentation.pt"):
        self.model_path = model_path
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_full_path = base_dir / self.model_path
        # Load the fine-tuned model if it exists, otherwise fall back to pure YOLOv8m-seg
        if os.path.exists(model_full_path):
            self.model = YOLO(str(model_full_path))
        else:
            self.model = YOLO("yolov8m-seg.pt")

    def segment(self, image: Image.Image) -> list:
        """
        Run YOLOv8 segmentation on the given image.
        Crop each detected item, keeping it fully in computer memory,
        and return the list of cropped images.
        """
        # Use agnostic_nms to remove overlapping boxes (like sleeves predicting as separate items)
        results = self.model(image, conf=0.5, iou=0.4, agnostic_nms=True)
        crops = []

        if not results or not results[0].boxes:
            return [image]  # fallback to the whole image
        
        result = results[0]
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            # Extract bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
            
        if not crops:
            return [image]
            
        return crops
