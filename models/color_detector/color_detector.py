import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Extended color map — covers all colors referenced in fashion_rules.json
COLOR_MAP = {
    "black":       (0,   0,   0),
    "white":       (255, 255, 255),
    "grey":        (128, 128, 128),
    "red":         (220, 20,  60),
    "green":       (34,  139, 34),
    "blue":        (30,  144, 255),
    "light_blue":  (173, 216, 230),   # covers pale / sky blue
    "denim":       (93,  140, 174),   # faded denim / washed jeans
    "yellow":      (255, 215, 0),
    "orange":      (255, 140, 0),
    "pink":        (255, 105, 180),
    "purple":      (128, 0,   128),
    "navy":        (0,   0,   128),
    "beige":       (245, 245, 220),
    "brown":       (139, 69,  19),
    "olive":       (107, 142, 35),
    "teal":        (0,   128, 128),
    "maroon":      (128, 0,   0),
    "burgundy":    (128, 0,   32),
    "khaki":       (195, 176, 145),
    "cream":       (255, 253, 208),
    "cyan":        (0,   255, 255),
}


class ColorDetector:
    def __init__(self, k: int = 4):
        self.k = k

    def _nearest_color_name(self, rgb: tuple) -> str:
        """Find the nearest named color using CIE Lab distance (perceptually uniform)."""
        # Convert query RGB → Lab
        query_img = np.array([[list(rgb)]], dtype=np.uint8)
        query_lab = cv2.cvtColor(query_img, cv2.COLOR_RGB2Lab)[0][0].astype(float)

        min_dist = float("inf")
        closest = "unknown"
        for name, known_rgb in COLOR_MAP.items():
            known_img = np.array([[list(known_rgb)]], dtype=np.uint8)
            known_lab = cv2.cvtColor(known_img, cv2.COLOR_RGB2Lab)[0][0].astype(float)
            dist = float(np.sum((query_lab - known_lab) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def predict(self, image: Image.Image) -> dict:
        img_rgba = image.convert("RGBA")
        img_rgb = np.array(img_rgba.convert("RGB"))
        alpha = np.array(img_rgba.split()[3])  # alpha channel

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # If the image has transparent pixels (from segmentation mask),
        # only cluster pixels that belong to the clothing item itself.
        foreground_mask = alpha > 0
        if foreground_mask.any():
            pixels = img_hsv[foreground_mask].astype(np.float32)
        else:
            # Fully opaque image (no mask) — use all pixels
            pixels = img_hsv.reshape(-1, 3).astype(np.float32)

        # Guard: need at least k pixels to cluster
        k = min(self.k, len(pixels))
        if k < 1:
            return {"color": "unknown", "hex": "#000000"}

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Pick the dominant cluster (largest)
        counts = np.bincount(kmeans.labels_)
        dominant_hsv = kmeans.cluster_centers_[np.argmax(counts)].astype(np.uint8)

        # Convert HSV → RGB for naming and hex
        dominant_rgb = cv2.cvtColor(dominant_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0][0]
        r, g, b = int(dominant_rgb[0]), int(dominant_rgb[1]), int(dominant_rgb[2])

        return {
            "color": self._nearest_color_name((r, g, b)),
            "hex":   f"#{r:02x}{g:02x}{b:02x}"
        }
