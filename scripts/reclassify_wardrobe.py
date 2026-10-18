"""
Re-label existing wardrobe items with the current fashion encoder.

Each row's saved crop is downloaded and classified again. Type, category,
style, pattern, season and confidence are updated. With --colors the color,
hex and secondary color are re-read too; otherwise color is left alone.

Dry run by default: prints every old and new label without writing.

    python scripts/reclassify_wardrobe.py                    # preview types
    python scripts/reclassify_wardrobe.py --colors           # preview types and colors
    python scripts/reclassify_wardrobe.py --colors --apply   # write changes
    python scripts/reclassify_wardrobe.py --user <id>        # one user only
"""
import argparse
import io
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.supabase_client import get_supabase  # noqa: E402
from inference.pipeline import read_item_color  # noqa: E402
from models.attribute_extractor.attribute_extractor import FashionEncoder  # noqa: E402
from models.color_detector.background import load_background_remover  # noqa: E402
from models.color_detector.color_detector import ColorDetector  # noqa: E402

FIELDS = ("type", "category", "style", "pattern", "season", "confidence")
COLOR_FIELDS = ("color", "hex", "secondary_color", "secondary_hex")


def _load_image(url: str) -> Image.Image:
    response = requests.get(url.rstrip("?"), timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the new labels")
    parser.add_argument("--colors", action="store_true", help="re-read colors as well")
    parser.add_argument("--user", help="only re-label this user's items")
    args = parser.parse_args()

    load_dotenv()
    fields = FIELDS + (COLOR_FIELDS if args.colors else ())
    table = get_supabase().table("wardrobe_items")
    query = table.select("id,image_path," + ",".join(fields))
    if args.user:
        query = query.eq("user_id", args.user)
    rows = query.execute().data or []
    print(f"{len(rows)} items. {'Writing changes.' if args.apply else 'Dry run, nothing is written.'}")

    encoder = FashionEncoder()
    color_detector = ColorDetector(background_remover=load_background_remover()) if args.colors else None
    changed = failed = 0
    for row in rows:
        item_id = str(row["id"])
        if not row.get("image_path"):
            print(f"{item_id[:8]}  skipped: no image")
            continue
        try:
            image = _load_image(row["image_path"])
            result = encoder.predict(image)
            if color_detector is not None:
                result.update(read_item_color(color_detector, encoder, image, result))
        except Exception as exc:
            failed += 1
            print(f"{item_id[:8]}  failed: {exc}")
            continue

        update = {field: result[field] for field in fields}
        old = f"{row.get('type')} / {row.get('category')}"
        new = f"{update['type']} / {update['category']}"
        line = f"{item_id[:8]}  {old:<32} -> {new:<28} ({update['confidence']:.2f})"
        if args.colors:
            new_color = update["color"] + (f" & {update['secondary_color']}" if update["secondary_color"] else "")
            line += f"   color {row.get('color')} -> {new_color} {update['hex']}"
        print(line)
        if any(row.get(field) != update[field] for field in fields if field not in ("confidence", "hex", "secondary_hex")):
            changed += 1
        if args.apply:
            table.update(update).eq("id", item_id).execute()

    print(f"\n{changed} items get a different label, {failed} failed.")
    if not args.apply:
        print("Run again with --apply to save.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
