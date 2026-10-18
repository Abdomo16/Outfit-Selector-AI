import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO, YOLOE
from ultralytics.utils.ops import scale_masks

from models import taxonomy

# Extra room around the detection box so hems and cuffs stay in the crop.
BOX_PAD = 0.04
# A mask covering less than this fraction of the box is misaligned. Keep the box.
MIN_MASK_COVER = 0.15
# Garment detections smaller than this share of the photo are prints, sleeves or noise.
MIN_AREA = 0.02
# Worn shoes, hats and bags are small in an outfit photo.
ACCESSORY_MIN_AREA = 0.01
ACCESSORY_CONF = 0.25
# A held bag sits inside the dress or coat box, but so do false hits on
# pockets and collars, so an accessory on a garment needs more confidence.
INSIDE_GARMENT = 0.7
ACCESSORY_CONF_ON_GARMENT = 0.4
# An item whose mask runs along a photo edge for this share of its width
# (top, bottom) or height (left, right) continues outside the photo.
EDGE_TOUCH = 0.25
# Pixels this close to an edge, as a share of the photo side, touch it.
EDGE_MARGIN = 0.01
# An item whose box is this much covered by other items is mostly hidden.
HIDDEN_SHARE = 0.3

ACCESSORY_MODEL_PATH = "weights/yoloe-26s-seg.pt"
ACCESSORY_PROMPTS_PATH = "weights/yoloe_wardrobe_prompts.pt"


class Segmenter:
    def __init__(
        self,
        model_path="weights/deepfashion2_yolov8s-seg.pt",
        accessory_model_path=ACCESSORY_MODEL_PATH,
    ):
        self.model_path = model_path
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_full_path = base_dir / self.model_path

        if os.path.exists(model_full_path):
            self.model = YOLO(str(model_full_path))
        else:
            self.model = YOLO("yolov8m-seg.pt")

        self.accessory_model = self._load_accessory_model(base_dir, accessory_model_path)

    @staticmethod
    def _load_accessory_model(base_dir: Path, model_path: str):
        """YOLOE prompted for shoes, bags and accessories. None when unavailable."""
        if os.environ.get("DISABLE_ACCESSORY_DETECTOR") == "1":
            return None
        names = list(taxonomy.YOLOE_PROMPTS)
        prompts_path = base_dir / ACCESSORY_PROMPTS_PATH
        try:
            model = YOLOE(str(base_dir / model_path))
            cached = torch.load(prompts_path) if prompts_path.exists() else None
            if cached is not None and cached["names"] == names:
                embeddings = cached["embeddings"]
            else:
                # Needs the YOLOE text encoder, downloaded on first use.
                embeddings = model.get_text_pe(names)
                torch.save({"names": names, "embeddings": embeddings}, prompts_path)
            model.set_classes(names, embeddings)
            return model
        except Exception as exc:
            print(f"[Segmenter] WARNING: accessory detector unavailable: {exc}")
            return None

    def segment(self, image: Image.Image) -> list:
        """
        Detect every wardrobe item in the photo and cut each one out.

        DeepFashion2 finds garments. YOLOE finds shoes, bags and accessories,
        which DeepFashion2 has no classes for. Instance masks whiten each
        item's background so the classifier and color detector only see it.

        Returns dicts with crop, label, conf, mask, box and partial. label is
        the detector class (e.g. "trousers", "sneaker") or None when nothing
        was detected and the full image is returned instead. partial is
        "cut_off", "hidden" or None, and is only set when the photo holds
        several items.
        """
        # 1. Normalise the photo: apply EXIF rotation, force RGB.
        image = ImageOps.exif_transpose(image).convert("RGB")
        W, H = image.size

        # 2. Class-aware NMS keeps overlapping different garments, e.g. a
        #    shirt worn over pants. YOLOE uses agnostic NMS because "shoe" and
        #    "sneaker" describe the same object.
        garments = self._filter(self._detect(self.model, image, conf=0.35), W, H, MIN_AREA)
        extras = []
        if self.accessory_model is not None:
            found = self._detect(self.accessory_model, image, conf=ACCESSORY_CONF, agnostic_nms=True)
            found = [
                d for d in found
                if d["conf"] >= ACCESSORY_CONF_ON_GARMENT
                or not any(self._inside(d["box"], g["box"]) >= INSIDE_GARMENT for g in garments)
            ]
            extras = self._filter(self._merge_shoe_pairs(found), W, H, ACCESSORY_MIN_AREA)
        kept = garments + extras

        if not kept:
            return [{"crop": image, "label": None, "conf": 0.0,
                     "mask": None, "box": (0, 0, W, H), "partial": None}]

        # One item in the photo is a catalog shot. Keep the whole frame.
        if len(kept) == 1:
            det = kept[0]
            crop, crop_mask = self._full_photo(image, det["mask"])
            return [{
                "crop": crop,
                "label": det["label"],
                "conf": det["conf"],
                "mask": crop_mask,
                "box": det["box"],
                "partial": None,
            }]

        segments = []
        for i, det in enumerate(kept):
            others = [d["mask"] for j, d in enumerate(kept) if j != i]
            box = det["box"]
            if det["mask"] is not None:
                box = self._mask_box(det["mask"], box, W, H)
                crop, crop_mask = self._masked_crop(image, det["mask"], box)
            else:
                crop = image.crop(box)
                crop_mask = None
            segments.append({
                "crop": crop,
                "label": det["label"],
                "conf": det["conf"],
                "mask": crop_mask,
                "box": box,
                "partial": self._partial_reason(det["mask"], det["box"], others),
            })
        return segments

    @staticmethod
    def _partial_reason(mask, box: tuple, other_masks: list):
        """
        "cut_off" when the item runs out of the photo, "hidden" when other
        items cover most of its box, else None. Needs photo-sized masks.
        """
        if mask is None:
            return None
        region = mask > 0.5
        if not region.any():
            return None
        H, W = region.shape
        my = max(2, int(EDGE_MARGIN * H))
        mx = max(2, int(EDGE_MARGIN * W))
        width = int(np.any(region, axis=0).sum())
        height = int(np.any(region, axis=1).sum())
        touches = (
            np.any(region[:my], axis=0).sum() / width,
            np.any(region[H - my:], axis=0).sum() / width,
            np.any(region[:, :mx], axis=1).sum() / height,
            np.any(region[:, W - mx:], axis=1).sum() / height,
        )
        if max(touches) >= EDGE_TOUCH:
            return "cut_off"

        x1, y1, x2, y2 = box
        covered = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        for other in other_masks:
            if other is not None and other.shape == region.shape:
                covered |= other[y1:y2, x1:x2] > 0.5
        covered &= ~region[y1:y2, x1:x2]
        if covered.sum() >= HIDDEN_SHARE * max(1, covered.size):
            return "hidden"
        return None

    @staticmethod
    def _mask_box(mask, box: tuple, width: int, height: int) -> tuple:
        """
        The mask's own bounds, padded, inside the detection box. The box is
        kept when the mask covers too little of it to be trusted.
        """
        x1, y1, x2, y2 = box
        region = mask[y1:y2, x1:x2] > 0.5
        if int(region.sum()) < MIN_MASK_COVER * max(1, region.size):
            return box
        rows = np.where(np.any(region, axis=1))[0]
        cols = np.where(np.any(region, axis=0))[0]
        mx1, mx2 = x1 + int(cols[0]), x1 + int(cols[-1]) + 1
        my1, my2 = y1 + int(rows[0]), y1 + int(rows[-1]) + 1
        px1, py1, px2, py2 = Segmenter._pad_box(mx1, my1, mx2, my2, width, height)
        return (max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2))

    @staticmethod
    def _detect(model, image: Image.Image, conf: float, agnostic_nms: bool = False) -> list:
        """Raw detections with boxes clamped to the photo and full-frame masks."""
        W, H = image.size
        results = model(
            image, conf=conf, iou=0.45, max_det=12, retina_masks=True,
            agnostic_nms=agnostic_nms, verbose=False,
        )
        if not results or not results[0].boxes:
            return []
        result = results[0]

        # Masks are in the letterboxed network frame. Scale them back to the
        # photo before cropping, otherwise the garment is sliced to a band.
        masks = None
        if result.masks is not None:
            raw = result.masks.data
            scaled = scale_masks(raw[:, None].float(), (H, W), padding=True)
            masks = scaled[:, 0].detach().cpu().numpy()

        found = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            found.append({
                "box": (max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))),
                "label": result.names[int(box.cls)],
                "conf": float(box.conf),
                "mask": masks[i] if masks is not None and i < len(masks) else None,
            })
        return found

    def _filter(self, detections: list, width: int, height: int, min_share: float) -> list:
        """Drop detections below min_share of the photo and pad the rest."""
        min_area = min_share * width * height
        kept = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            kept.append({**det, "box": self._pad_box(x1, y1, x2, y2, width, height)})
        return kept

    @staticmethod
    def _inside(inner: tuple, outer: tuple) -> float:
        """Share of the inner box covered by the outer box."""
        ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
        iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
        area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return ix * iy / area

    @staticmethod
    def _merge_shoe_pairs(detections: list) -> list:
        """
        Join left and right shoe boxes into one item.

        Shoe boxes whose rows overlap are one pair, however far apart the
        feet stand. A photo rarely shows two different pairs side by side.
        """
        shoes = [d for d in detections if taxonomy.YOLOE_PROMPTS.get(d["label"]) == "Shoes"]
        others = [d for d in detections if taxonomy.YOLOE_PROMPTS.get(d["label"]) != "Shoes"]
        merged = []
        for shoe in sorted(shoes, key=lambda d: d["box"][0]):
            if merged:
                last = merged[-1]
                a, b = last["box"], shoe["box"]
                if min(a[3], b[3]) > max(a[1], b[1]):
                    mask = None
                    if last["mask"] is not None and shoe["mask"] is not None:
                        mask = np.maximum(last["mask"], shoe["mask"])
                    best = last if last["conf"] >= shoe["conf"] else shoe
                    merged[-1] = {
                        "box": (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                        "label": best["label"],
                        "conf": best["conf"],
                        "mask": mask,
                    }
                    continue
            merged.append(dict(shoe))
        return others + merged

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
