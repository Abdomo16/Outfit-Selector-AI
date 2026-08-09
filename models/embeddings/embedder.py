import numpy as np
from PIL import Image

class Embedder:
    def __init__(self):
        pass

    def predict(self, image: Image.Image):
        """
        Takes an image and returns a dummy 512-dim embedding vector.
        """
        #  normalized random 512-dim numpy array
        vec = np.random.randn(512)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
