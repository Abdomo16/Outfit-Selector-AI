import os

import numpy as np
from PIL import Image

# About 0.7 s per item on CPU. birefnet-general-lite is sharper on crops
# holding two garments but takes about 9 s.
DEFAULT_MODEL = "isnet-general-use"


class BackgroundRemover:
    """Foreground alpha from rembg, so colors are read from the garment only."""

    def __init__(self, model_name: str | None = None):
        from rembg import new_session

        self.model_name = model_name or os.environ.get("REMBG_MODEL", DEFAULT_MODEL)
        print(f"[BackgroundRemover] Loading {self.model_name}...")
        self.session = new_session(self.model_name)

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Alpha in 0..1 with the image's height and width."""
        from rembg import remove

        mask = remove(image.convert("RGB"), session=self.session, only_mask=True)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.BILINEAR)
        return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def load_background_remover():
    """None when disabled or when rembg is not installed."""
    if os.environ.get("DISABLE_BACKGROUND_REMOVAL") == "1":
        return None
    try:
        return BackgroundRemover()
    except Exception as exc:
        print(f"[BackgroundRemover] WARNING: unavailable, colors use detector masks: {exc}")
        return None
