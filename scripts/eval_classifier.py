"""
Measure category and type accuracy on your own labeled photos.

Put photos in eval_data/<Category>/<type>/, using the names in
models/taxonomy.py. Spaces in a type may be written as underscores:

    eval_data/Shoes/sneakers/1.jpg
    eval_data/Outerwear/puffer_jacket/coat_photo.png

    python scripts/eval_classifier.py                      # encoder only
    python scripts/eval_classifier.py --detect             # detector + encoder
    python scripts/eval_classifier.py --model Marqo/marqo-fashionSigLIP
"""
import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import taxonomy  # noqa: E402
from models.attribute_extractor.attribute_extractor import FashionEncoder  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _samples(root: Path) -> list:
    samples = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name
        if category not in taxonomy.CATEGORIES:
            print(f"Skipping unknown category folder: {category}")
            continue
        for type_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            type_name = type_dir.name.replace("_", " ").lower()
            if taxonomy.category_for_type(type_name) != category:
                print(f"Skipping {category}/{type_dir.name}: not a {category} type")
                continue
            for path in sorted(type_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append((path, category, type_name))
    return samples


def _predict_upload(encoder, segmenter, image: Image.Image) -> list:
    """Every item a real upload of this photo would save."""
    return [
        encoder.predict(seg["crop"], taxonomy.hint_for_label(seg["label"]))
        for seg in segmenter.segment(image)
    ]


def _accuracy(rows: list) -> str:
    hits = sum(1 for ok in rows if ok)
    return f"{hits}/{len(rows)} ({100 * hits / len(rows):.0f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="eval_data", help="folder of labeled photos")
    parser.add_argument("--model", help="fashion model id, defaults to FASHION_MODEL_ID")
    parser.add_argument("--detect", action="store_true", help="run the detectors first, like a real upload")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"No folder at {root}. See the docstring at the top of this script for the layout.")
        return 1
    samples = _samples(root)
    if not samples:
        print(f"No labeled photos found in {root}.")
        return 1

    encoder = FashionEncoder(args.model)
    segmenter = None
    if args.detect:
        from models.segmentation.segmenter import Segmenter
        segmenter = Segmenter()

    by_category = defaultdict(list)
    by_type = defaultdict(list)
    mistakes = Counter()
    seconds = 0.0
    items_saved = 0
    for path, category, type_name in samples:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        start = time.perf_counter()
        if segmenter is not None:
            results = _predict_upload(encoder, segmenter, image)
        else:
            results = [encoder.predict(image)]
        seconds += time.perf_counter() - start
        items_saved += len(results)

        # With detectors a photo can save several items; it counts as right
        # when one of them carries the expected label.
        by_category[category].append(any(r["category"] == category for r in results))
        type_hit = any(r["type"] == type_name for r in results)
        by_type[type_name].append(type_hit)
        if not type_hit:
            mistakes[(type_name, results[0]["type"])] += 1

    all_categories = [ok for rows in by_category.values() for ok in rows]
    all_types = [ok for rows in by_type.values() for ok in rows]
    print(f"\nModel: {encoder.model_id}{' with detectors' if segmenter else ''}")
    print(f"Photos: {len(samples)}, {seconds / len(samples):.2f} s per photo")
    if segmenter is not None:
        print(f"Items saved per photo: {items_saved / len(samples):.2f}")
    print(f"Category accuracy: {_accuracy(all_categories)}")
    print(f"Type accuracy:     {_accuracy(all_types)}")

    print("\nPer category:")
    for category in taxonomy.CATEGORIES:
        if category in by_category:
            print(f"  {category:<12} {_accuracy(by_category[category])}")

    print("\nPer type:")
    for type_name, rows in sorted(by_type.items()):
        print(f"  {type_name:<16} {_accuracy(rows)}")

    if mistakes:
        print("\nMost common mistakes (expected -> predicted):")
        for (expected, predicted), count in mistakes.most_common(10):
            print(f"  {expected} -> {predicted}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
