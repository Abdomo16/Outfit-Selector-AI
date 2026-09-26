import requests
import json

BASE = "http://127.0.0.1:8000"

def upload_image(image_path: str) -> list:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE}/wardrobe/upload",
            files={"file": (image_path, f, "image/jpeg")},
        )
    resp.raise_for_status()
    return resp.json()["items"]

def recommend(wardrobe: list, occasion: str, season: str):
    payload = {
        "wardrobe": [
            {
                "id": i,
                "type": item["type"],
                "confidence": item.get("confidence", 0.9),
                "color": item.get("color", ""),
                "hex": item.get("hex", ""),
                "pattern": item.get("pattern"),
                "style": item.get("style"),
                "occasion": "",
                "season": item.get("season"),
                "embedding": item.get("embedding", [0.5, 0.5, 0.5]),
                "image_path": item.get("image_path"),
            }
            for i, item in enumerate(wardrobe)
        ],
        "occasion": occasion,
        "season": season,
    }
    resp = requests.post(f"{BASE}/recommend/", json=payload)
    return resp.status_code, resp.json()

if __name__ == "__main__":
    # 1) Real items from a photo
    items = upload_image("sample_test.jpg")
    print(f"Uploaded {len(items)} item(s) from photo:")
    for it in items:
        print(f"  - {it['type']} ({it['color']})")

    if not items:
        print("No items detected in photo - using a hand-built wardrobe instead.")
        items = []

    # 2) Make sure we have a top + bottom (photo may only contain one item)
    if not any(it["type"] in ("tops", "t-shirt", "t_shirt", "shirt", "hoodie", "polo", "sweater") for it in items):
        items.append({
            "type": "t-shirt", "confidence": 0.98, "color": "white",
            "hex": "#FFFFFF", "pattern": "Plain", "style": "Minimal",
            "occasion": "Casual", "season": "Summer",
            "embedding": [0.9, 0.1, 0.2],
        })
    if not any(it["type"] in ("jeans", "trousers", "chinos", "shorts", "skirt", "leggings") for it in items):
        items.append({
            "type": "jeans", "confidence": 0.98, "color": "blue",
            "hex": "#0000FF", "pattern": "Plain", "style": "Minimal",
            "occasion": "Casual", "season": "Summer",
            "embedding": [0.1, 0.9, 0.1],
        })
    if not any(it["type"] in ("sneakers", "shoes", "loafers", "boots") for it in items):
        items.append({
            "type": "sneakers", "confidence": 0.98, "color": "white",
            "hex": "#FFFFFF", "pattern": "Plain", "style": "Casual",
            "occasion": "Casual", "season": "Summer",
            "embedding": [0.1, 0.1, 0.9],
        })

    # 3) Ask for a recommendation
    status, data = recommend(items, occasion="casual", season="summer")
    print(f"\nHTTP {status}")
    if status == 200:
        print("Recommended outfit:")
        for piece in data["outfit"]:
            print(f"  - {piece['category']} ({piece['color']})")
        print(f"Score: {data['score']:.3f}")
    else:
        print("Response:", json.dumps(data, indent=2))
