import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks

# Extra room around the detection box so hems and cuffs stay in the crop.
BOX_PAD = 0.04
# A mask covering less than this fraction of the box is misaligned. Keep the box.
MIN_MASK_COVER = 0.15


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
        # retina_masks scales the mask to the photo before the box crops it.
        results = self.model(
            image, conf=0.35, iou=0.45, max_det=12, retina_masks=True,
        )

        if not results or not results[0].boxes:
            return [{"crop": image, "label": None, "conf": 0.0,
                     "mask": None, "box": (0, 0, W, H)}]

        result = results[0]
        boxes = result.boxes
        names = result.names

        # Ignore detections smaller than 2% of the image — prints, sleeves,
        # and stray layers predicted as separate items.
        min_area = 0.02 * W * H

        # Masks are in the letterboxed network frame. Scale them back to the
        # photo before cropping, otherwise the garment is sliced to a band.
        masks = None
        if result.masks is not None:
            raw = result.masks.data
            scaled = scale_masks(raw[:, None].float(), (H, W), padding=True)
            masks = scaled[:, 0].detach().cpu().numpy()

        kept = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            x1, y1, x2, y2 = self._pad_box(x1, y1, x2, y2, W, H)
            kept.append((i, (x1, y1, x2, y2), names[int(box.cls)], float(box.conf)))

        if not kept:
            return [{"crop": image, "label": None, "conf": 0.0,
                     "mask": None, "box": (0, 0, W, H)}]

        # One garment in the photo is a catalog shot. Keep the whole frame.
        if len(kept) == 1:
            i, box, label, conf = kept[0]
            mask = masks[i] if masks is not None and i < len(masks) else None
            crop, crop_mask = self._full_photo(image, mask)
            return [{
                "crop": crop,
                "label": label,
                "conf": conf,
                "mask": crop_mask,
                "box": box,
            }]

        segments = []
        for i, box, label, conf in kept:
            if masks is not None and i < len(masks):
                crop, crop_mask = self._masked_crop(image, masks[i], box)
            else:
                crop = image.crop(box)
                crop_mask = None
            segments.append({
                "crop": crop,
                "label": label,
                "conf": conf,
                "mask": crop_mask,
                "box": box,
            })
        return segments

    @staticmethod
    def _pad_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple:
        pad_x = int((x2 - x1) * BOX_PAD)
        pad_y = int((y2 - y1) * BOX_PAD)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    @staticmethod
    def _full_photo(image: Image.Image, mask) -> tuple:
        """
        Save a one-garment photo at its original size.

        A mask that only covers a slice is ignored. Whitening that slice
        would erase the rest of the garment, which is the waist on a pants shot.
        """
        if mask is None or mask.shape[:2] != (image.size[1], image.size[0]):
            return image, None

        region = mask > 0.5
        rows = np.any(region, axis=1)
        if not rows.any():
            return image, None
        r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        if (r1 - r0 + 1) / image.size[1] < 0.85:
            return image, None

        arr = np.array(image)
        white = np.full_like(arr, 255)
        composited = np.where(region[..., None], arr, white)
        mask_crop = (region * 255).astype(np.uint8)
        return Image.fromarray(composited), mask_crop

    @staticmethod
    def _masked_crop(
        image: Image.Image, mask: np.ndarray, bbox: tuple
    ) -> tuple:
        """
        Cut one item out of a mask that is already in photo coordinates.

        The crop stays the full detection box. White replaces only pixels the
        mask says are background. A mask that covers little of the box is
        ignored, so a bad mask cannot slice the garment down to a quarter.
        """
        x1, y1, x2, y2 = bbox
        box_h = max(1, y2 - y1)
        box_w = max(1, x2 - x1)
        region = mask[y1:y2, x1:x2] > 0.5

        if int(region.sum()) < MIN_MASK_COVER * box_h * box_w:
            return image.crop((x1, y1, x2, y2)), None

        arr = np.array(image)
        white = np.full_like(arr, 255)
        item_mask = np.zeros(mask.shape[:2], dtype=bool)
        item_mask[y1:y2, x1:x2] = region
        composited = np.where(item_mask[..., None], arr, white)

        crop = Image.fromarray(composited[y1:y2, x1:x2])
        mask_crop = (region * 255).astype(np.uint8)
        return crop, mask_crop
