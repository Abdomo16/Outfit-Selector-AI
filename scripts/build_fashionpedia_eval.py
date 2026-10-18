"""
Build an evaluation set from Fashionpedia validation photos.

Crops every labeled item that is large enough and saves it as
<out>/<Category>/<type>/<image>_<box>.jpg for scripts/eval_classifier.py.
Fashionpedia only labels broad classes ("shoe", "bag, wallet"), so each crop
goes under one representative type. Read category accuracy from this set;
type accuracy is only meaningful for classes that map to a single type.

    python scripts/build_fashionpedia_eval.py --out eval_data_fashionpedia --per-class 15
"""
import argparse
import io
import sys
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "detection-datasets/fashionpedia"

# Fashionpedia class -> (category, representative type). Garment parts
# (sleeve, collar, pocket...) and classes outside the taxonomy are skipped.
CLASS_MAP = {
    "shirt, blouse": ("Tops", "shirt"),
    "top, t-shirt, sweatshirt": ("Tops", "t-shirt"),
    "sweater": ("Tops", "sweater"),
    "cardigan": ("Tops", "cardigan"),
    "jacket": ("Outerwear", "jacket"),
    "coat": ("Outerwear", "coat"),
    "pants": ("Bottoms", "trousers"),
    "shorts": ("Bottoms", "shorts"),
    "skirt": ("Bottoms", "skirt"),
    "dress": ("Dresses", "dress"),
    "jumpsuit": ("Dresses", "jumpsuit"),
    "shoe": ("Shoes", "sneakers"),
    "bag, wallet": ("Bags", "handbag"),
    "glasses": ("Accessories", "sunglasses"),
    "hat": ("Accessories", "hat"),
    "tie": ("Accessories", "tie"),
    "watch": ("Accessories", "watch"),
    "belt": ("Accessories", "belt"),
    "sock": ("Accessories", "socks"),
    "scarf": ("Accessories", "scarf"),
}
MIN_SIDE = 64
MIN_SHARE = 0.01


def _rows(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": "default", "split": "val", "offset": offset, "length": length}
    response = requests.get(ROWS_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="eval_data_fashionpedia")
    parser.add_argument("--per-class", type=int, default=15, help="crops per Fashionpedia class")
    parser.add_argument("--max-photos", type=int, default=400)
    args = parser.parse_args()

    out = Path(args.out)
    first = _rows(0, 1)
    objects = next(f for f in first["features"] if f["name"] == "objects")
    names = objects["type"]["category"]["feature"]["names"]
    saved = Counter()
    offset = 0
    while offset < args.max_photos and any(saved[c] < args.per_class for c in CLASS_MAP):
        page = _rows(offset, 100)["rows"]
        if not page:
            break
        for entry in page:
            row = entry["row"]
            wanted = [
                (names[c], box)
                for c, box in zip(row["objects"]["category"], row["objects"]["bbox"])
                if names[c] in CLASS_MAP and saved[names[c]] < args.per_class
            ]
            if not wanted:
                continue
            try:
                photo = requests.get(row["image"]["src"], timeout=60)
                photo.raise_for_status()
                image = Image.open(io.BytesIO(photo.content)).convert("RGB")
            except Exception as exc:
                print(f"photo {row['image_id']} failed: {exc}")
                continue
            width, height = image.size
            for index, (name, (x1, y1, x2, y2)) in enumerate(wanted):
                w, h = x2 - x1, y2 - y1
                if min(w, h) < MIN_SIDE or w * h < MIN_SHARE * width * height:
                    continue
                if saved[name] >= args.per_class:
                    continue
                category, type_name = CLASS_MAP[name]
                folder = out / category / type_name.replace(" ", "_")
                folder.mkdir(parents=True, exist_ok=True)
                image.crop((int(x1), int(y1), int(x2), int(y2))).save(
                    folder / f"{row['image_id']}_{index}_{name.split(',')[0]}.jpg", quality=92,
                )
                saved[name] += 1
        offset += len(page)
        print(f"{offset} photos read, {sum(saved.values())} crops saved")

    for name in CLASS_MAP:
        print(f"  {name:<26} {saved[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
