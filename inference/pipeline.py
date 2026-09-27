from PIL import Image, ImageOps

from models.classifier.classifier import ClothingClassifier
from models.color_detector.color_detector import ColorDetector
from models.segmentation.segmenter import Segmenter
from models.attribute_extractor.attribute_extractor import AttributeExtractor

# DeepFashion2 class → app taxonomy (23 classes).
# None = no equivalent app class; classification falls back to CLIP.
DF2_TYPE_MAP = {
    "short_sleeved_shirt": "Tshirts",
    "long_sleeved_shirt": "Shirts",
    "short_sleeved_outwear": None,
    "long_sleeved_outwear": None,
    "vest": None,
    "sling": "Tops",
    "shorts": "Shorts",
    "trousers": "Trousers",
    "skirt": None,
    "short_sleeved_dress": None,
    "long_sleeved_dress": None,
    "vest_dress": None,
    "sling_dress": None,
}


class InferencePipeline:
    def __init__(self):
        self.segmenter = Segmenter()
        self.classifier = ClothingClassifier()
        self.color_detector = ColorDetector()
        self.attribute_extractor = AttributeExtractor()

    def _classify_crop(self, image: Image.Image, seg_label: str, seg_conf: float) -> dict:
        """
        Classify one segmented crop.

        The DeepFashion2 segmenter already knows the garment category (trained
        on 491K fashion images), so its label is the most reliable signal for
        clothing. keras/CLIP only handle accessories (no detection) and the
        DF2 categories the app taxonomy doesn't have (outerwear, dresses...).
        """
        mapped = DF2_TYPE_MAP.get(seg_label)

        if mapped is not None:
            item_type = mapped
            # Refine generic trousers/shorts into jeans/chinos/leggings etc.
            # only when CLIP is reasonably confident — otherwise keep the
            # reliable segmenter label.
            if mapped in ("Trousers", "Shorts"):
                refined, ref_conf = self.attribute_extractor.get_pants_type(image)
                if ref_conf >= 0.5 and refined in ("jeans", "chinos", "leggings"):
                    item_type = refined.capitalize()
            return {"type": item_type, "confidence": round(seg_conf, 4)}

        if seg_label:  # detected clothing with no mapped app class (dress, vest...)
            clip_type, clip_conf = self.attribute_extractor.classify_type(image)
            return {"type": clip_type, "confidence": round(clip_conf, 4)}

        # No segmentation detection (accessories, single items) → keras path.
        class_res = self.classifier.predict(image)
        if class_res["confidence"] < 0.5:
            clip_type, clip_conf = self.attribute_extractor.classify_type(image)
            if clip_conf > class_res["confidence"]:
                class_res = {"type": clip_type, "confidence": round(clip_conf, 4)}

        item_type = class_res["type"]
        if item_type in ("trousers", "shorts"):
            refined, ref_conf = self.attribute_extractor.get_pants_type(image)
            if ref_conf >= 0.5 and refined in ("jeans", "chinos", "leggings"):
                item_type = refined.capitalize()
                class_res = {"type": item_type, "confidence": round(ref_conf, 4)}
        return class_res

    def process_single_item(self, image: Image.Image, seg_label: str = "", seg_conf: float = 0.0, seg_mask=None) -> dict:
        """Run classification + color + attributes on a single cropped item."""
        class_res = self._classify_crop(image, seg_label, seg_conf)

        color_res = self.color_detector.predict(image, mask=seg_mask)
        attr_res = self.attribute_extractor.predict(image)

        return {
            "type":       class_res["type"],
            "confidence": class_res["confidence"],
            "color":      color_res["color"],
            "hex":        color_res["hex"],
            "pattern":    attr_res["pattern"],
            "style":      attr_res["style"],
            "occasion":   attr_res["occasion"],
            "season":     attr_res["season"]
        }

    def process_upload(self, image: Image.Image) -> dict:
        """
        Full photo → segment → N cropped items → classify each item.
        """
        # 1. Normalise the photo once (EXIF rotation) — segmenter does the
        #    same internally, so segment boxes map onto this image.
        source = ImageOps.exif_transpose(image).convert("RGB")

        # 2. Segment the photo into individual clothing article crops
        #    (masked background removal happens inside).
        #    Each entry: {"crop", "label", "conf", "mask", "box"} where box
        #    is in `source` coordinates.
        segments = self.segmenter.segment(source)

        # 3. Classify each item and read its color from the BOX region of the
        #    original photo. Overlay debugging showed the DF2 masks can land
        #    on the wrong fabric when garments occlude each other, while the
        #    boxes track the visible garment region reliably.
        results = []
        for seg in segments:
            crop = seg["crop"]
            label = seg["label"] or ""
            conf = seg["conf"]
            box = seg["box"]

            class_res = self._classify_crop(crop, label, conf)
            attr_res = self.attribute_extractor.predict(crop)

            band = (0.4, 1.0) if label in ("trousers", "shorts", "skirt") else None
            box_crop = source.crop(box)
            ranked = self.color_detector.predict_ranked(box_crop, mask=None, row_band=band)

            results.append({
                "crop": crop,
                **class_res,
                "ranked_colors": ranked,
                "pattern": attr_res["pattern"],
                "style": attr_res["style"],
                "occasion": attr_res["occasion"],
                "season": attr_res["season"],
            })

        # 4. Occlusion-aware re-read for lower-body garments: a long top
        #    hanging over trousers leaks its own fabric color into the
        #    trousers' box read. Re-read the bottom band of the box while
        #    excluding the overlapping item's fabric colors, with the tighter
        #    background filter (real-photo backgrounds aren't pure white).
        lower_body = ("trousers", "shorts", "skirt")

        def _hex_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[j:j + 2], 16) for j in (0, 2, 4))

        def _boxes_overlap(a, b):
            ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ox * oy > 0

        for i, seg in enumerate(segments):
            label = seg["label"] or ""
            if label not in lower_body:
                continue
            box = seg["box"]
            exclude_rgbs = []
            for j, other in enumerate(segments):
                if j == i or not (other["label"] or ""):
                    continue
                if not _boxes_overlap(box, other["box"]):
                    continue
                exclude_rgbs.extend(
                    _hex_rgb(c["hex"])
                    for c in results[j]["ranked_colors"] if c["share"] >= 0.15
                )
            if not exclude_rgbs:
                continue
            box_crop = source.crop(box)
            results[i]["ranked_colors"] = self.color_detector.predict_ranked(
                box_crop, mask=None, row_band=(0.7, 1.0),
                exclude_rgbs=exclude_rgbs, strict_bg=True, k=8,
            )

        # 5. Cross-item color dedup: garments in the same photo are usually
        #    different colors. When an item's dominant cluster was already
        #    claimed by an earlier item (e.g. a shirt hanging over trousers
        #    leaks into the trousers mask), fall back to the item's runner-up
        #    cluster — choosing the candidate most dissimilar to the claimed
        #    colors (a warm-lit background gap often shares the claimed color's
        #    hue, so hue-similar fallbacks are avoided). Items are assigned
        #    most-certain-first (highest dominant-cluster share).
        order = sorted(
            range(len(results)),
            key=lambda i: results[i]["ranked_colors"][0]["share"],
            reverse=True,
        )
        claimed = set()          # claimed color names
        claimed_rgbs = []        # claimed color RGB values
        picks = {}
        for i in order:
            ranked = [
                c for c in results[i]["ranked_colors"] if c["share"] >= 0.15
            ] or results[i]["ranked_colors"]

            top = ranked[0]
            if top["color"] not in claimed:
                pick = top
            else:
                pick = max(
                    ranked,
                    key=lambda c: min(
                        sum((a - b) ** 2 for a, b in zip(_hex_rgb(c["hex"]), cr))
                        for cr in claimed_rgbs
                    ),
                )
            claimed.add(pick["color"])
            claimed_rgbs.append(_hex_rgb(pick["hex"]))
            picks[i] = pick

        items = []
        for i, res in enumerate(results):
            pick = picks[i]
            items.append({
                "crop":       res["crop"],
                "type":       res["type"],
                "confidence": res["confidence"],
                "color":      pick["color"],
                "hex":        pick["hex"],
                "pattern":    res["pattern"],
                "style":      res["style"],
                "occasion":   res["occasion"],
                "season":     res["season"],
            })

        # Return the items in an array
        return {"items": items}
