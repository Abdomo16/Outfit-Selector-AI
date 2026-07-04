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
    def __init__(self, k: int = 4):
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

    def predict(self, image: Image.Image) -> dict:
        img_rgb = np.array(image.convert("RGB"))
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        pixels = img_hsv.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
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
