import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Extended color map — covers all colors referenced in fashion_rules.json
COLOR_MAP = {
    "black":    (0,   0,   0),
    "white":    (255, 255, 255),
    "grey":     (128, 128, 128),
    "red":      (220, 20,  60),
    "green":    (34,  139, 34),
    "blue":     (30,  144, 255),
    "yellow":   (255, 215, 0),
    "orange":   (255, 140, 0),
    "pink":     (255, 105, 180),
    "purple":   (128, 0,   128),
    "navy":     (0,   0,   128),
    "beige":    (245, 245, 220),
    "brown":    (139, 69,  19),
    "olive":    (107, 142, 35),
    "teal":     (0,   128, 128),
    "maroon":   (128, 0,   0),
    "burgundy": (128, 0,   32),
    "khaki":    (195, 176, 145),
    "cream":    (255, 253, 208),
    "cyan":     (0,   255, 255),
}


class ColorDetector:
    def __init__(self, k: int = 6):
        self.k = k

    def _nearest_color_name(self, rgb: tuple) -> str:
        min_dist = float("inf")
        closest = "unknown"
        for name, known_rgb in COLOR_MAP.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, known_rgb))
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def predict_ranked(
        self, image: Image.Image, mask: np.ndarray = None, row_band: tuple = None,
        exclude_rgbs: list = None, strict_bg: bool = False, k: int = None,
    ) -> list:
        """
        Return the image's color clusters ranked by size:
        [(name, hex, share), ...] with share in 0..1.

        Used by the pipeline for cross-item color dedup: when two garments in
        one photo claim the same dominant color, the second item can fall back
        to its runner-up cluster.

        row_band: optional (start_frac, end_frac) restricting the pixels to a
        vertical band of the image — used for lower-body garments whose crop
        includes the occluding top hanging over them (the legs at the bottom
        show the garment's true color).

        exclude_rgbs: optional list of RGB tuples to drop from the read —
        used to remove an occluding garment's fabric colors before clustering.

        strict_bg: use a tighter background filter (drops shaded walls too) —
        for box crops of real photos where the background isn't pure white.
        """
        img_rgb = np.array(image.convert("RGB"))

        if mask is not None:
            m = np.asarray(mask)
            if m.ndim == 3:
                m = m[..., 0]
            m = m > 127
            if m.shape[:2] != img_rgb.shape[:2]:
                m = cv2.resize(
                    m.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 127
            if m.any():
                if row_band is not None:
                    h = m.shape[0]
                    r0 = int(h * row_band[0])
                    banded = np.zeros_like(m)
                    banded[r0:int(h * row_band[1]), :] = m[r0:int(h * row_band[1]), :]
                    if banded.sum() >= 0.1 * m.sum():
                        m = banded
                # Erode the mask boundary: edge pixels blend the garment with
                # the background and skew the clusters.
                kernel = np.ones((5, 5), np.uint8)
                eroded = cv2.erode(m.astype(np.uint8), kernel, iterations=2) > 0
                if eroded.sum() >= 0.1 * m.sum():
                    m = eroded
                img_rgb = img_rgb[m]
        elif row_band is not None:
            # No mask — restrict the crop's pixels to the vertical band.
            h = img_rgb.shape[0]
            r0 = int(h * row_band[0])
            img_rgb = img_rgb[r0:int(h * row_band[1]), :]

        img_rgb = img_rgb.reshape(-1, 3).astype(np.float32)

        # Drop pixels close to the occluding garment's colors.
        if exclude_rgbs:
            keep = np.ones(len(img_rgb), dtype=bool)
            for ref in exclude_rgbs:
                dist = np.sqrt(((img_rgb - np.asarray(ref, dtype=np.float32)) ** 2).sum(axis=1))
                keep &= dist > 60
            if keep.sum() >= 0.1 * len(img_rgb):
                img_rgb = img_rgb[keep]

        # Drop low-saturation bright pixels — background walls and gaps
        # between garments, not garment fabric. strict_bg drops ALL
        # low-saturation pixels (walls, even shaded ones, are far less
        # saturated than garment fabric under warm lighting).
        sel_hsv = cv2.cvtColor(
            img_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)
        if strict_bg:
            not_bg = sel_hsv[:, 1] >= 90
        else:
            not_bg = ~((sel_hsv[:, 1] < 40) & (sel_hsv[:, 2] > 210))
        if not_bg.sum() >= 0.1 * len(sel_hsv):
            img_rgb = img_rgb[not_bg]

        img_hsv = cv2.cvtColor(
            img_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)
        pixels = img_hsv.astype(np.float32)

        kmeans = KMeans(n_clusters=k or self.k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        n_k = k or self.k
        counts = np.bincount(kmeans.labels_, minlength=n_k)
        order = np.argsort(counts)[::-1]
        total = counts.sum()

        ranked = []
        for idx in order:
            hsv = kmeans.cluster_centers_[idx].astype(np.uint8)
            rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0][0]
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            ranked.append({
                "color": self._nearest_color_name((r, g, b)),
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "share": float(counts[idx]) / total,
            })
        return ranked

    def predict(self, image: Image.Image, mask: np.ndarray = None) -> dict:
        dominant = self.predict_ranked(image, mask=mask)[0]
        return {"color": dominant["color"], "hex": dominant["hex"]}
