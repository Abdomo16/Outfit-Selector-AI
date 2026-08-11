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
        Each crop is returned as an RGBA image where the background pixels are
        transparent, so downstream color detection is not polluted by the background.
        """
        import numpy as np
        import cv2

        # Use agnostic_nms to remove overlapping boxes (like sleeves predicting as separate items)
        results = self.model(image, conf=0.5, iou=0.4, agnostic_nms=True)
        crops = []

        if not results or not results[0].boxes:
            return [image]  # fallback to the whole image

        result = results[0]
        boxes = result.boxes
        masks = result.masks  # polygon/binary masks from YOLO

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # If YOLO produced a segmentation mask for this box, apply it as
            # an alpha channel so the ColorDetector can ignore background pixels.
            if masks is not None and i < len(masks):
                img_rgba = image.convert("RGBA")
                w, h = img_rgba.size

                # masks.data is [N, H, W] in the model's scaled resolution
                mask_tensor = masks.data[i]  # shape (H', W')
                mask_np = mask_tensor.cpu().numpy()

                # Resize to match the original image dimensions
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                alpha = (mask_resized > 0.5).astype(np.uint8) * 255

                # Apply mask as alpha channel
                r, g, b, a = img_rgba.split()
                alpha_img = Image.fromarray(alpha, mode="L")
                img_masked = Image.merge("RGBA", (r, g, b, alpha_img))

                crop = img_masked.crop((x1, y1, x2, y2))
            else:
                # No mask available — plain bounding-box crop (no background removal)
                crop = image.crop((x1, y1, x2, y2))

            crops.append(crop)

        if not crops:
            return [image]

        return crops
