from PIL import Image

from models.classifier.classifier import ClothingClassifier
from models.color_detector.color_detector import ColorDetector
from models.segmentation.segmenter import Segmenter
from models.attribute_extractor.attribute_extractor import AttributeExtractor


class InferencePipeline:
    def __init__(self):
        self.segmenter = Segmenter()
        self.classifier = ClothingClassifier()
        self.color_detector = ColorDetector()
        self.attribute_extractor = AttributeExtractor()

    def process_single_item(self, image: Image.Image) -> dict:
        """Run classifier + color detector on a single cropped clothing image."""
        class_res = self.classifier.predict(image)
        color_res = self.color_detector.predict(image)
        attr_res = self.attribute_extractor.predict(image)
        return {
            "type":       class_res["type"],
            "confidence": class_res["confidence"],
            "color":      color_res["color"],
            "hex":        color_res["hex"],
            "pattern":    attr_res["pattern"],
            "style":      attr_res["style"],
            "season":     attr_res["season"]
        }

    def process_upload(self, image: Image.Image) -> dict:
        """
        Phase 3: full photo → segment → N cropped items → classify each item.
        """
        # 1. Segment the photo into individual clothing article crops
        crops = self.segmenter.segment(image)

        # 2. Process each cropped item
        items = []
        for crop in crops:
            item_data = self.process_single_item(crop)
            items.append(item_data)

        # Return the items in an array
        return {"items": items}
