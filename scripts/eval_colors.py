"""
Measure color accuracy against hand-checked labels.

The labels file is a CSV with columns id, expected, note. id is a
wardrobe_items id; expected is one of the names in models/colors.py.
Each item's saved image is read again and compared with both the color
saved in the database (before) and the current detector (now).

Exact means the same name; family means the outfit rules treat them alike,
e.g. light grey and grey.

    python scripts/eval_colors.py                          # scripts/color_labels.csv
    python scripts/eval_colors.py --labels my_labels.csv
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.supabase_client import get_supabase  # noqa: E402
from inference.pipeline import read_item_color  # noqa: E402
from models import colors  # noqa: E402
from models.attribute_extractor.attribute_extractor import FashionEncoder  # noqa: E402
from models.color_detector.background import load_background_remover  # noqa: E402
from models.color_detector.color_detector import ColorDetector  # noqa: E402

DEFAULT_LABELS = Path(__file__).resolve().parent / "color_labels.csv"


def _score(pairs: list) -> str:
    exact = sum(1 for got, want in pairs if got == want)
    fam = sum(1 for got, want in pairs if colors.family(got) == colors.family(want))
    n = len(pairs)
    return f"exact {exact}/{n} ({100 * exact / n:.0f}%), family {fam}/{n} ({100 * fam / n:.0f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    with open(args.labels, newline="", encoding="utf-8") as f:
        labels = [row for row in csv.DictReader(f) if row.get("id")]
    unknown = [row["expected"] for row in labels if row["expected"] not in colors.COLOR_NAMES]
    if unknown:
        print(f"Unknown color names in labels: {sorted(set(unknown))}")
        return 1

    load_dotenv()
    rows = {
        str(r["id"]): r
        for r in get_supabase().table("wardrobe_items")
        .select("id,color,image_path").in_("id", [row["id"] for row in labels]).execute().data
    }
    encoder = FashionEncoder()
    detector = ColorDetector(background_remover=load_background_remover())

    before, now = [], []
    for label in labels:
        row = rows.get(label["id"])
        if row is None or not row.get("image_path"):
            print(f"{label['id'][:8]}  not found, skipped")
            continue
        response = requests.get(row["image_path"].rstrip("?"), timeout=30)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        item = encoder.predict(image)
        got = read_item_color(detector, encoder, image, item)
        saved = colors.family(row.get("color")) if row.get("color") else ""
        before.append(((row.get("color") or "").lower(), label["expected"]))
        now.append((got["color"], label["expected"]))
        mark = "ok " if colors.family(got["color"]) == colors.family(label["expected"]) else "BAD"
        second = f" & {got['secondary_color']}" if got["secondary_color"] else ""
        print(f"{mark} {label['id'][:8]}  expected {label['expected']:<10} saved {saved:<10} "
              f"now {got['color']}{second} {got['hex']}  {label.get('note', '')}")

    if not now:
        print("No labeled items found.")
        return 1
    print(f"\nBefore (saved in database): {_score(before)}")
    print(f"Now (current detector):     {_score(now)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
