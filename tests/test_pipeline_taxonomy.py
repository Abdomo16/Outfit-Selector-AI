import numpy as np
from PIL import Image

from inference.pipeline import InferencePipeline
from models.color_detector.color_detector import ColorDetector


class FakeSegmenter:
    def __init__(self, segments):
        self.segments = segments

    def segment(self, image):
        return self.segments


class FakeEncoder:
    """Records the hint for each crop and answers from the first hinted category."""

    def __init__(self, color_vote=None):
        self.hints = []
        self.color_vote = color_vote

    def color_scores(self, image_embed, type_name):
        return [self.color_vote] if self.color_vote else []

    def predict(self, image, hint=()):
        self.hints.append(hint)
        category = hint[0] if hint else "Tops"
        type_name = {"Bottoms": "jeans", "Shoes": "sneakers", "Tops": "t-shirt"}[category]
        return {
            "category": category,
            "type": type_name,
            "confidence": 0.9,
            "pattern": "Plain",
            "style": "Minimal",
            "occasion": "Travel",
            "season": "Summer",
            "embedding": [0.1, 0.2],
        }


def _pipeline(segments, color_vote=None):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.segmenter = FakeSegmenter(segments)
    pipeline.color_detector = ColorDetector()
    pipeline.encoder = FakeEncoder(color_vote)
    return pipeline


def _segment(label, box, color):
    crop = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), color)
    return {"crop": crop, "label": label, "conf": 0.8, "mask": None, "box": box}


def test_upload_returns_category_type_and_embedding_per_item():
    photo = Image.new("RGB", (200, 400), (240, 240, 240))
    photo.paste((20, 40, 160), (20, 150, 180, 330))
    photo.paste((250, 250, 250), (40, 340, 160, 390))
    pipeline = _pipeline([
        _segment("trousers", (20, 150, 180, 330), (20, 40, 160)),
        _segment("sneaker", (40, 340, 160, 390), (250, 250, 250)),
    ])

    items = pipeline.process_upload(photo)["items"]

    assert pipeline.encoder.hints == [("Bottoms",), ("Shoes",)]
    assert [(i["category"], i["type"]) for i in items] == [("Bottoms", "jeans"), ("Shoes", "sneakers")]
    for item in items:
        assert item["embedding"] == [0.1, 0.2]
        assert item["hex"].startswith("#")


def test_photo_without_detections_is_classified_without_a_hint():
    photo = Image.new("RGB", (100, 100), (200, 30, 30))
    pipeline = _pipeline([{"crop": photo, "label": None, "conf": 0.0, "mask": None, "box": (0, 0, 100, 100)}])

    items = pipeline.process_upload(photo)["items"]

    assert pipeline.encoder.hints == [()]
    assert items[0]["category"] == "Tops"
    assert items[0]["color"] == "red"


def _single(photo):
    return [{"crop": photo, "label": None, "conf": 0.0, "mask": None, "box": (0, 0, *photo.size)}]


def test_confident_vote_that_fits_the_fabric_names_the_color():
    photo = Image.new("RGB", (100, 100), (165, 165, 170))
    pipeline = _pipeline(_single(photo), color_vote=("light grey", 0.8))

    assert pipeline.process_upload(photo)["items"][0]["color"] == "light grey"


def test_vote_that_contradicts_the_fabric_is_ignored():
    photo = Image.new("RGB", (100, 100), (18, 23, 29))
    pipeline = _pipeline(_single(photo), color_vote=("olive", 0.9))

    item = pipeline.process_upload(photo)["items"][0]
    assert item["color"] == "black"
    assert item["secondary_color"] is None


def test_other_items_pixels_do_not_tint_the_trousers():
    photo = Image.new("RGB", (200, 400), (255, 255, 255))
    photo.paste((20, 20, 20), (40, 150, 160, 390))       # black trousers
    photo.paste((220, 30, 40), (30, 100, 170, 260))      # red top hanging over them
    top_mask = np.zeros((400, 200), dtype=np.uint8)
    top_mask[100:260, 30:170] = 255
    trousers = {"crop": photo.crop((30, 140, 170, 395)), "label": "trousers", "conf": 0.9,
                "mask": None, "box": (30, 140, 170, 395)}
    top = {"crop": photo.crop((30, 100, 170, 260)), "label": "short_sleeved_shirt", "conf": 0.9,
           "mask": top_mask[100:260, 30:170], "box": (30, 100, 170, 260)}
    pipeline = _pipeline([trousers, top])

    items = pipeline.process_upload(photo)["items"]
    assert items[0]["color"] == "black"
    assert items[1]["color"] == "red"
