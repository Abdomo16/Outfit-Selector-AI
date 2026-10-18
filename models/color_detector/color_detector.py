import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from models import colors

# rembg alpha above this is garment.
ALPHA_THRESHOLD = 0.5
# A foreground smaller than this share of the crop means rembg found nothing.
MIN_FOREGROUND = 0.05
# Use the rembg and detector overlap only when it keeps this much of the rembg
# foreground; detector masks sometimes land on the wrong fabric.
MIN_OVERLAP_KEPT = 0.4
# Shadows and shine are not the fabric color.
DARKEST_DROPPED = 5
BRIGHTEST_DROPPED = 2
# Clusters closer than this (CIEDE2000) are shades of one color.
SAME_COLOR = 12.0
# A second color must be this different from the first and cover this share.
# Plain items need a large share: a smaller one is usually another garment
# showing in the crop, like a shirt above trousers.
SECONDARY_DIFFERENCE = 20.0
SECONDARY_SHARE = 0.35
SECONDARY_SHARE_PATTERNED = 0.10
MAX_PIXELS = 20000


class ColorDetector:
    def __init__(self, k: int = 4, background_remover=None):
        self.k = k
        self.background_remover = background_remover

    def fabric_mask(self, image: Image.Image, det_mask=None, exclude_mask=None) -> np.ndarray:
        """Boolean mask of the garment's own pixels, with fallbacks."""
        width, height = image.size
        detector = _as_bool(det_mask, (height, width))
        mask = None

        if self.background_remover is not None:
            alpha = self.background_remover(image) > ALPHA_THRESHOLD
            if alpha.mean() >= MIN_FOREGROUND:
                mask = alpha
                if detector is not None:
                    overlap = alpha & detector
                    if overlap.sum() >= MIN_OVERLAP_KEPT * alpha.sum():
                        mask = overlap

        if mask is None and detector is not None and detector.mean() >= MIN_FOREGROUND:
            mask = detector
        if mask is None:
            mask = np.zeros((height, width), dtype=bool)
            mask[int(height * 0.2):int(height * 0.8), int(width * 0.2):int(width * 0.8)] = True

        excluded = _as_bool(exclude_mask, (height, width))
        if excluded is not None:
            kept = mask & ~excluded
            if kept.sum() >= 0.1 * mask.sum():
                mask = kept

        # Edge pixels blend fabric with the background.
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        if eroded.sum() >= 0.3 * mask.sum():
            mask = eroded
        return mask

    def clusters(self, image: Image.Image, mask: np.ndarray) -> list:
        """Fabric color clusters as [{"lab", "share"}], largest first."""
        pixels = np.asarray(image.convert("RGB"))[mask].reshape(-1, 3)
        if len(pixels) < 50:
            pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
        if len(pixels) > MAX_PIXELS:
            pixels = pixels[np.random.default_rng(0).choice(len(pixels), MAX_PIXELS, replace=False)]

        lab = colors.rgb_to_lab(pixels.astype(np.float32))
        low, high = np.percentile(lab[:, 0], [DARKEST_DROPPED, 100 - BRIGHTEST_DROPPED])
        trimmed = lab[(lab[:, 0] >= low) & (lab[:, 0] <= high)]
        if len(trimmed) >= 50:
            lab = trimmed

        k = max(1, min(self.k, len(np.unique(lab.round(1), axis=0))))
        labels = KMeans(n_clusters=k, random_state=42, n_init=4).fit_predict(lab)
        found = [
            {"lab": np.median(lab[labels == i], axis=0), "share": float((labels == i).mean())}
            for i in range(k)
        ]
        found.sort(key=lambda c: c["share"], reverse=True)

        merged = []
        for cluster in found:
            same = next(
                (m for m in merged if colors.ciede2000(m["lab"], cluster["lab"]) < SAME_COLOR),
                None,
            )
            if same is None:
                merged.append(dict(cluster))
            else:
                same["share"] += cluster["share"]
        merged.sort(key=lambda c: c["share"], reverse=True)
        return merged

    def read(self, image: Image.Image, det_mask=None, exclude_mask=None, patterned: bool = False) -> dict:
        """
        Primary color (name, hex, Lab) and, when the fabric clearly has one,
        a secondary color. Names come from the pixels only; the pipeline may
        replace the primary name with a fashion-model vote.
        """
        found = self.clusters(image, self.fabric_mask(image, det_mask, exclude_mask))
        primary = found[0]
        min_share = SECONDARY_SHARE_PATTERNED if patterned else SECONDARY_SHARE
        secondary = next(
            (
                c for c in found[1:]
                if c["share"] >= min_share
                and colors.ciede2000(primary["lab"], c["lab"]) >= SECONDARY_DIFFERENCE
            ),
            None,
        )
        result = {
            "color": colors.name_for_lab(primary["lab"]),
            "hex": colors.rgb_to_hex(colors.lab_to_rgb(primary["lab"])),
            "lab": [float(v) for v in primary["lab"]],
            "secondary_color": None,
            "secondary_hex": None,
        }
        if secondary is not None:
            result["secondary_color"] = colors.name_for_lab(secondary["lab"])
            result["secondary_hex"] = colors.rgb_to_hex(colors.lab_to_rgb(secondary["lab"]))
        return result

    def predict(self, image: Image.Image, mask: np.ndarray = None) -> dict:
        read = self.read(image, det_mask=mask)
        return {"color": read["color"], "hex": read["hex"]}


def _as_bool(mask, shape: tuple):
    """A uint8, float or bool mask resized to shape, or None."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != bool:
        m = m > (127 if m.max() > 1 else 0.5)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m
