from PIL import Image

from models.classifier.classifier import ClothingClassifier
from models.color_detector.color_detector import ColorDetector


class InferencePipeline:
    def __init__(self):
        self.classifier = ClothingClassifier()
        self.color_detector = ColorDetector()

    def process_single_item(self, image: Image.Image) -> dict:
        """Run classifier + color detector on a single cropped clothing image."""
        class_res = self.classifier.predict(image)
        color_res = self.color_detector.predict(image)
        return {
            "type":       class_res["type"],
            "confidence": class_res["confidence"],
            "color":      color_res["color"],
            "hex":        color_res["hex"]
        }

    def process_upload(self, image: Image.Image) -> dict:
        """
        Phase 2: single-item upload wrapper.
        Returns the Flutter-expected shape: { "items": [...] }.
        Phase 3 will replace this with segmentation → N cropped items.
        """
        item = self.process_single_item(image)
        return {"items": [item]}
