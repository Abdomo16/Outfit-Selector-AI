import numpy as np
from PIL import Image, ImageOps

from models import colors, taxonomy
from models.color_detector.background import load_background_remover
from models.color_detector.color_detector import ColorDetector
from models.segmentation.segmenter import Segmenter
from models.attribute_extractor.attribute_extractor import FashionEncoder


def _photo_mask(seg: dict, width: int, height: int):
    """A segment's mask in photo coordinates, or None."""
    mask = seg.get("mask")
    if mask is None:
        return None
    m = np.asarray(mask) > 127
    if m.shape == (height, width):
        return m
    x1, y1, x2, y2 = seg["box"]
    if m.shape != (y2 - y1, x2 - x1):
        return None
    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = m
    return full


def read_item_color(color_detector, encoder, image: Image.Image, item: dict,
                    det_mask=None, exclude_mask=None) -> dict:
    """
    Measure the garment's own pixels, then let the fashion model name the
    color when the measurement agrees with it. item needs type, pattern and
    embedding from the encoder.
    """
    read = color_detector.read(
        image, det_mask, exclude_mask, patterned=item.get("pattern") not in (None, "Plain"),
    )
    scores = encoder.color_scores(item["embedding"], item["type"])
    name = colors.choose_name(read["color"], read["lab"], scores)
    secondary = read["secondary_color"]
    if secondary == name:
        secondary = None
    return {
        "color": name,
        "hex": read["hex"],
        "secondary_color": secondary,
        "secondary_hex": read["secondary_hex"] if secondary else None,
    }


class InferencePipeline:
    def __init__(self):
        self.segmenter = Segmenter()
        self.color_detector = ColorDetector(background_remover=load_background_remover())
        self.encoder = FashionEncoder()

    def _classify_crop(self, image: Image.Image, seg_label: str) -> dict:
        """
        Classify one crop with the fashion encoder.

        The detector label favours its categories, so a DeepFashion2
        "trousers" box leans towards bottoms and a YOLOE "sneaker" box towards
        shoes. No label means every type starts equal.
        """
        return self.encoder.predict(image, taxonomy.hint_for_label(seg_label))

    def _color(self, image: Image.Image, item: dict, det_mask=None, exclude_mask=None) -> dict:
        return read_item_color(self.color_detector, self.encoder, image, item, det_mask, exclude_mask)

    def process_single_item(self, image: Image.Image, seg_label: str = "", seg_mask=None) -> dict:
        """Classification, color and attributes for one already-cropped item."""
        result = self._classify_crop(image, seg_label)
        return {**result, **self._color(image, result, det_mask=seg_mask)}

    def process_upload(self, image: Image.Image) -> dict:
        """
        Full photo → segment → N cropped items → classify each item.
        """
        # 1. Normalise the photo once (EXIF rotation) — segmenter does the
        #    same internally, so segment boxes map onto this image.
        source = ImageOps.exif_transpose(image).convert("RGB")
        width, height = source.size

        # 2. Segment the photo into individual items. Each entry has
        #    {"crop", "label", "conf", "mask", "box"}, box in `source` pixels.
        segments = self.segmenter.segment(source)
        photo_masks = [_photo_mask(seg, width, height) for seg in segments]

        # 3. Classify each crop, then read its color from the original photo
        #    (crops have a whitened background) with the other items' pixels
        #    removed, so a top hanging over trousers does not tint them.
        items = []
        for i, seg in enumerate(segments):
            result = self._classify_crop(seg["crop"], seg["label"] or "")

            others = [m for j, m in enumerate(photo_masks) if j != i and m is not None]
            exclude = np.logical_or.reduce(others) if others else None
            if seg["crop"].size == source.size:
                color_image, det_mask = source, photo_masks[i]
            else:
                x1, y1, x2, y2 = seg["box"]
                color_image = source.crop(seg["box"])
                det_mask = photo_masks[i][y1:y2, x1:x2] if photo_masks[i] is not None else None
                exclude = exclude[y1:y2, x1:x2] if exclude is not None else None

            items.append({
                "crop": seg["crop"],
                "partial": seg.get("partial"),
                **result,
                **self._color(color_image, result, det_mask, exclude),
            })

        return {"items": items}
