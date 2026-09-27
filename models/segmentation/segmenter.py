import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


class Segmenter:
    def __init__(self, model_path="weights/deepfashion2_yolov8s-seg.pt"):
        self.model_path = model_path
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_full_path = base_dir / self.model_path

        if os.path.exists(model_full_path):
            self.model = YOLO(str(model_full_path))
        else:
            self.model = YOLO("yolov8m-seg.pt")

    def segment(self, image: Image.Image) -> list:
        """
        Run YOLOv8 segmentation on the given image.

        Uses the predicted instance masks (not just bounding boxes) to cut each
        item out of its background, so the classifier and color detector only
        see the garment itself — not the floor, hangers, or other clutter.

        Returns a list of tuples (crop, class_name, confidence, mask). Each
        crop is RGB, composited on a clean white background. class_name is the
        model's detected garment category (e.g. "trousers") or None when
        nothing was detected and the full image is returned instead. mask is a
        uint8 array matching the crop size (or None).
        """
        # 1. Normalise the photo: apply EXIF rotation, force RGB.
        image = ImageOps.exif_transpose(image).convert("RGB")
        W, H = image.size

        # 2. Class-aware NMS (default) keeps overlapping *different* items,
        #    e.g. a shirt worn over pants. conf is lower than before because
        #    the min-area + mask filters below now handle the false positives
        #    that agnostic NMS was (badly) trying to remove.
        results = self.model(image, conf=0.35, iou=0.45, max_det=12)

        if not results or not results[0].boxes:
            return [{"crop": image, "label": None, "conf": 0.0,
                     "mask": None, "box": (0, 0, W, H)}]

        result = results[0]
        boxes = result.boxes
        names = result.names

        # Ignore detections smaller than 2% of the image — prints, sleeves,
        # and stray layers predicted as separate items.
        min_area = 0.02 * W * H

        # Mask tensor: (N, h, w) at model resolution, one per detection.
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        segments = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if (x2 - x1) * (y2 - y1) < min_area:
                continue

            if masks is not None and i < len(masks):
                crop, crop_mask = self._masked_crop(image, masks[i], (x1, y1, x2, y2))
            else:
                # Model produced boxes but no masks — fall back to plain crop.
                crop = image.crop((x1, y1, x2, y2))
                crop_mask = None
            segments.append({
                "crop": crop,
                "label": names[int(box.cls)],
                "conf": float(box.conf),
                "mask": crop_mask,
                "box": (x1, y1, x2, y2),
            })

        if not segments:
            return [{"crop": image, "label": None, "conf": 0.0,
                     "mask": None, "box": (0, 0, W, H)}]

        return segments

    @staticmethod
    def _masked_crop(
        image: Image.Image, mask: np.ndarray, bbox: tuple
    ) -> tuple:
        """
        Cut one item out using its instance mask.

        The mask comes at model resolution, so it is resized to the original
        image size, cropped to the mask's own extent (tighter than the
        detection box), and used to composite the garment onto a white
        background.

        Returns (crop, mask_crop) where mask_crop is a uint8 array matching
        the crop size — the pipeline uses it so color detection ignores the
        white background.
        """
        x1, y1, x2, y2 = bbox
        W, H = image.size

        # Resize mask to full image size (nearest keeps it binary).
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((W, H), Image.NEAREST)

        mask_arr = np.array(mask_img) > 127
        # Only keep mask pixels inside this detection's box — masks from other
        # detections can overlap this box.
        box_mask = np.zeros((H, W), dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        item_mask = mask_arr & box_mask

        # Tighten the crop to the mask's actual extent.
        rows = np.any(item_mask, axis=1)
        cols = np.any(item_mask, axis=0)
        if rows.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            y1, y2, x1, x2 = r0, r1 + 1, c0, c1 + 1

        arr = np.array(image)
        white = np.full_like(arr, 255)
        composited = np.where(item_mask[..., None], arr, white)

        crop = Image.fromarray(composited[y1:y2, x1:x2])
        mask_crop = (item_mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        return crop, mask_crop
